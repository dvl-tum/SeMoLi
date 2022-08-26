import os
from abc import ABC, abstractmethod
import logging
import functools
from functools import partial
import numpy as np
import torch
import torch_geometric
from torch_geometric.transforms import Compose, FixedPoints
import copy

from torch_points3d.models import model_interface
from torch_points3d.core.data_transform import instantiate_transforms
from torch_points3d.datasets.batch import SimpleBatch

from torch.utils.data._utils.collate import default_collate

# A logger for this file
log = logging.getLogger(__name__)


def explode_transform(transforms):
    """ Returns a flattened list of transform
    Arguments:
        transforms {[list | T.Compose]} -- Contains list of transform to be added
    Returns:
        [list] -- [List of transforms]
    """
    out = []
    if transforms is not None:
        if isinstance(transforms, Compose):
            out = copy.deepcopy(transforms.transforms)
        elif isinstance(transforms, list):
            out = copy.deepcopy(transforms)
        else:
            raise Exception("Transforms should be provided either within a list or a Compose")
    return out


class BaseDataset:
    def __init__(self, dataset_opt):
        self.dataset_opt = dataset_opt

        # Default dataset path
        self._data_path = os.path.join(dataset_opt.dataroot, dataset_opt.dataset_name)

        self._batch_size = None
        self.strategies = {}

        self.train_sampler = None
        self.test_sampler = None
        self.val_sampler = None

        self._train_dataset = None
        self._test_dataset = None
        self._val_dataset = None

        self.train_pre_batch_collate_transform = None
        self.val_pre_batch_collate_transform = None
        self.test_pre_batch_collate_transform = None

        BaseDataset.set_transform(self, dataset_opt)

        self.used_properties = {}

    @staticmethod
    def set_transform(obj, dataset_opt):
        """This function create and set the transform to the obj as attributes
        """
        obj.pre_transform = None
        obj.test_transform = None
        obj.train_transform = None
        obj.val_transform = None
        obj.inference_transform = None

        for key_name in dataset_opt.keys():
            if "transform" in key_name:
                new_name = key_name.replace("transforms", "transform")
                try:
                    transform = instantiate_transforms(getattr(dataset_opt, key_name))
                except Exception:
                    log.exception("Error trying to create {}, {}".format(new_name, getattr(dataset_opt, key_name)))
                    continue
                setattr(obj, new_name, transform)

        inference_transform = explode_transform(obj.pre_transform)
        inference_transform += explode_transform(obj.test_transform)
        obj.inference_transform = Compose(inference_transform) if len(inference_transform) > 0 else None

    def create_dataloaders(
        self,
        model: model_interface.DatasetInterface,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
    ):
        """ Creates the data loaders. Must be called in order to complete the setup of the Dataset
        """
        conv_type = model.conv_type
        self._batch_size = batch_size

        if self.train_sampler:
            log.info(self.train_sampler)

        if self.train_dataset:
            self._train_loader = self._dataloader(
                self.train_dataset,
                self.train_pre_batch_collate_transform,
                conv_type,
                batch_size=batch_size,
                shuffle=shuffle and not self.train_sampler,
                num_workers=num_workers,
                sampler=self.train_sampler,
            )

        if self.test_dataset:
            self._test_loaders = [
                self._dataloader(
                    dataset,
                    self.test_pre_batch_collate_transform,
                    conv_type,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    sampler=self.test_sampler,
                )
                for dataset in self.test_dataset
            ]

        if self.val_dataset:
            self._val_loader = self._dataloader(
                self.val_dataset,
                self.val_pre_batch_collate_transform,
                conv_type,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                sampler=self.val_sampler,
            )

    def _dataloader(self, dataset, pre_batch_collate_transform, conv_type, **kwargs):

        batch_collate_function = self.__class__._get_collate_function(
            conv_type, pre_batch_collate_transform
        )
        num_workers = kwargs.get("num_workers", 0)
        persistent_workers = num_workers > 0
        dataloader = partial(
            torch.utils.data.DataLoader,
            collate_fn=batch_collate_function,
            worker_init_fn=np.random.seed,
            persistent_workers=persistent_workers,
        )
        return dataloader(dataset, **kwargs)
    
    @staticmethod
    def _collate_fn(batch, collate_fn=None, pre_collate_transform=None):
        if pre_collate_transform:
            batch = pre_collate_transform(batch)
        return collate_fn(batch)

    @staticmethod
    def _get_collate_function(conv_type, pre_collate_transform=None):
        if conv_type == 'dense':
            fn = default_collate
        else:
            raise Exception("Collate function ot implemented yet for non-dense dataset types.")
        return partial(BaseDataset._collate_fn, collate_fn=fn, pre_collate_transform=pre_collate_transform)

    @property
    def _loaders(self):
        loaders = []
        if self.has_train_loader:
            loaders += [self.train_dataloader]
        if self.has_val_loader:
            loaders += [self.val_dataloader]
        if self.has_test_loaders:
            loaders += self.test_dataloaders
        return loaders

    def get_raw_data(self, stage, idx, **kwargs):
        assert stage in self.available_dataset_names
        dataset = self.get_dataset(stage)
        if hasattr(dataset, "get_raw_data"):
            return dataset.get_raw_data(idx, **kwargs)
        else:
            raise Exception("Dataset {} doesn t have a get_raw_data function implemented".format(dataset))

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_batches(self):
        out = {
            self.train_dataset.name: len(self._train_loader),
            "val": len(self._val_loader) if self.has_val_loader else 0,
        }
        if self.test_dataset:
            for loader in self._test_loaders:
                stage_name = loader.dataset.name
                out[stage_name] = len(loader)
        return out

    def get_dataset(self, name):
        """ Get a dataset by name. Raises an exception if no dataset was found

        Parameters
        ----------
        name : str
        """
        all_datasets = [self.train_dataset, self.val_dataset]
        if self.test_dataset:
            all_datasets += self.test_dataset
        for dataset in all_datasets:
            if dataset is not None and dataset.name == name:
                return dataset
        raise ValueError("No dataset with name %s was found." % name)

    def add_weights(self, dataset_name="train", class_weight_method="sqrt"):
        """ Add class weights to a given dataset that are then accessible using the `class_weights` attribute
        """
        L = self.num_classes
        weights = torch.ones(L)
        dataset = self.get_dataset(dataset_name)
        idx_classes, counts = torch.unique(dataset.data.y, return_counts=True)

        dataset.idx_classes = torch.arange(L).long()
        weights[idx_classes] = counts.float()
        weights = weights.float()
        weights = weights.mean() / weights
        if class_weight_method == "sqrt":
            weights = torch.sqrt(weights)
        elif str(class_weight_method).startswith("log"):
            weights = torch.log(1.1 + weights / weights.sum())
        else:
            raise ValueError("Method %s not supported" % class_weight_method)

        weights /= torch.sum(weights)
        log.info("CLASS WEIGHT : {}".format([np.round(weight.item(), 4) for weight in weights]))
        setattr(dataset, "weight_classes", weights)

        return dataset
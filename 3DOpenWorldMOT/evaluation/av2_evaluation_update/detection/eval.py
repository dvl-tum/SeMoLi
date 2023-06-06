# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Argoverse 3D object detection evaluation.

Evaluation:

    Precision/Recall

        1. Average Precision: Standard VOC-style average precision calculation
            except a true positive requires a 3D Euclidean center distance of less
            than a predefined threshold.

    True Positive Errors

        All true positive errors accumulate error solely when an object is a true positive match
        to a ground truth detection. The matching criterion is represented by `tp_thresh` in the DetectionCfg class.
        In our challenge, we use a `tp_thresh` of 2.0 meters.

        1. Average Translation Error: The average Euclidean distance (center-based) between a
            detection and its ground truth assignment.
        2. Average Scale Error: The average intersection over union (IoU) after the prediction
            and assigned ground truth's pose has been aligned.
        3. Average Orientation Error: The average angular distance between the detection and
            the assigned ground truth. We choose the smallest angle between the two different
            headings when calculating the error.

    Composite Scores

        1. Composite Detection Score: The ranking metric for the detection leaderboard. This
            is computed as the product of mAP with the sum of the complements of the true positive
            errors (after normalization), i.e.:
                - Average Translation Measure (ATM): ATE / TP_THRESHOLD; 0 <= 1 - ATE / TP_THRESHOLD <= 1.
                - Average Scaling Measure (ASM): 1 - ASE / 1;  0 <= 1 - ASE / 1 <= 1.
                - Average Orientation Measure (AOM): 1 - AOE / PI; 0 <= 1 - AOE / PI <= 1.

            These (as well as AP) are averaged over each detection class to produce:
                - mAP
                - mATM
                - mASM
                - mAOM

            Lastly, the Composite Detection Score is computed as:
                CDS = mAP * (mATE + mASE + mAOE); 0 <= mAP * (mATE + mASE + mAOE) <= 1.

        ** In the case of no true positives under the specified threshold, the true positive measures
            will assume their upper bounds of 1.0. respectively.

Results:

    The results are represented as a (C + 1, P) table, where C + 1 represents the number of evaluation classes
    in addition to the mean statistics average across all classes, and P refers to the number of included statistics,
    e.g. AP, ATE, ASE, AOE, CDS by default.
"""
import logging
from multiprocessing import get_context
from typing import Dict, Final, List, Optional, Tuple

import numpy as np
import pandas as pd

from av2.evaluation.detection.constants import NUM_DECIMALS, MetricNames, TruePositiveErrorNames
from av2.evaluation.detection.utils import (
    DetectionCfg,
    accumulate,
    compute_average_precision,
    groupby,
    load_mapped_avm_and_egoposes,
)
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.cuboid import ORDERED_CUBOID_COL_NAMES
from av2.utils.io import TimestampedCitySE3EgoPoses
from av2.utils.typing import NDArrayBool, NDArrayFloat

import matplotlib.pyplot as plt

TP_ERROR_COLUMNS: Final[Tuple[str, ...]] = tuple(x.value for x in TruePositiveErrorNames)
DTS_COLUMN_NAMES: Final[Tuple[str, ...]] = tuple(ORDERED_CUBOID_COL_NAMES) + ("score",)
GTS_COLUMN_NAMES: Final[Tuple[str, ...]] = tuple(ORDERED_CUBOID_COL_NAMES) + ("category_int",) + ("num_interior_pts",) + ("filter_moving",)
UUID_COLUMN_NAMES: Final[Tuple[str, ...]] = (
    "log_id",
    "timestamp_ns",
    "category",
)

logger = logging.getLogger(__name__)


def evaluate(
    dts: pd.DataFrame,
    gts: pd.DataFrame,
    cfg: DetectionCfg,
    n_jobs: int = 8,
    min_points: int = 0,
    max_points: int = 10000,
    filter_class: str = 'REGULAR_VEHICLE',
    eval_only_machted: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate a set of detections against the ground truth annotations.

    Each sweep is processed independently, computing assignment between detections and ground truth annotations.

    Args:
        dts: (N,14) Table of detections.
        gts: (M,16) Table of ground truth annotations.
        cfg: Detection configuration.
        n_jobs: Number of jobs running concurrently during evaluation.

    Returns:
        (C+1,K) Table of evaluation metrics where C is the number of classes. Plus a row for their means.
        K refers to the number of evaluation metrics.

    Raises:
        RuntimeError: If accumulation fails.
        ValueError: If ROI pruning is enabled but a dataset directory is not specified.
    """
    if cfg.eval_only_roi_instances and cfg.dataset_dir is None:
        raise ValueError(
            "ROI pruning has been enabled, but the dataset directory has not be specified. "
            "Please set `dataset_directory` to the split root, e.g. av2/sensor/val."
        )

    # Sort both the detections and annotations by lexicographic order for grouping.
    dts = dts.sort_values(list(UUID_COLUMN_NAMES))
    gts = gts.sort_values(list(UUID_COLUMN_NAMES))

    dts_npy: NDArrayFloat = dts[list(DTS_COLUMN_NAMES)].to_numpy().astype(float)
    gts_npy: NDArrayFloat = gts[list(GTS_COLUMN_NAMES)].to_numpy().astype(float)
    dts_uuids: List[str] = dts[list(UUID_COLUMN_NAMES)].to_numpy().tolist()
    gts_uuids: List[str] = gts[list(UUID_COLUMN_NAMES)].to_numpy().tolist()

    # We merge the unique identifier -- the tuple of ("log_id", "timestamp_ns", "category")
    # into a single string to optimize the subsequent grouping operation.
    # `groupby_mapping` produces a mapping from the uuid to the group of detections / annotations
    # which fall into that group.
    uuid_to_dts = groupby([":".join(map(str, x)) for x in dts_uuids], dts_npy)
    uuid_to_gts = groupby([":".join(map(str, x)) for x in gts_uuids], gts_npy)

    log_id_to_avm: Optional[Dict[str, ArgoverseStaticMap]] = None
    log_id_to_timestamped_poses: Optional[Dict[str, TimestampedCitySE3EgoPoses]] = None
    
    # Load maps and egoposes if roi-pruning is enabled.
    if cfg.eval_only_roi_instances and cfg.dataset_dir is not None:
        logger.info("Loading maps and egoposes ...")
        log_ids: List[str] = gts.loc[:, "log_id"].unique().tolist()
        log_id_to_avm, log_id_to_timestamped_poses = load_mapped_avm_and_egoposes(log_ids, cfg.dataset_dir)

    args_list: List[Tuple[NDArrayFloat, NDArrayFloat, DetectionCfg, Optional[ArgoverseStaticMap], Optional[SE3]]] = []
    uuids = sorted(uuid_to_dts.keys() | uuid_to_gts.keys())
    for uuid in uuids:
        log_id, timestamp_ns, _ = uuid.split(":")
        args: Tuple[NDArrayFloat, NDArrayFloat, DetectionCfg, Optional[ArgoverseStaticMap], Optional[SE3]]

        sweep_dts: NDArrayFloat = np.zeros((0, 10))
        sweep_gts: NDArrayFloat = np.zeros((0, 10))
        if uuid in uuid_to_dts:
            sweep_dts = uuid_to_dts[uuid]
        if uuid in uuid_to_gts:
            sweep_gts = uuid_to_gts[uuid]

        args = sweep_dts, sweep_gts, cfg, None, None, min_points, max_points, int(timestamp_ns), filter_class, eval_only_machted
        if log_id_to_avm is not None and log_id_to_timestamped_poses is not None:
            avm = log_id_to_avm[log_id]
            city_SE3_ego = log_id_to_timestamped_poses[log_id][int(timestamp_ns)]
            args = sweep_dts, sweep_gts, cfg, avm, city_SE3_ego, min_points, max_points, int(timestamp_ns), filter_class, eval_only_machted
        args_list.append(args)

    logger.info("Starting evaluation ...")
    with get_context("spawn").Pool(processes=n_jobs) as p:
        outputs: Optional[List[Tuple[NDArrayFloat, NDArrayFloat]]] = p.starmap(accumulate, args_list)

    if outputs is None:
        raise RuntimeError("Accumulation has failed! Please check the integrity of your detections and annotations.")
    dts_list, gts_list, num_points_tps, num_points_fns = zip(*outputs)

    METRIC_COLUMN_NAMES = cfg.affinity_thresholds_m + TP_ERROR_COLUMNS + ("is_evaluated",)
    dts_metrics: NDArrayFloat = np.concatenate(dts_list)  # type: ignore
    gts_metrics: NDArrayFloat = np.concatenate(gts_list)  # type: ignore
    dts.loc[:, METRIC_COLUMN_NAMES] = dts_metrics
    gts.loc[:, METRIC_COLUMN_NAMES] = gts_metrics
    if len([t for t in num_points_tps if t is not None]):
        num_points_tps = np.concatenate([t for t in num_points_tps if t is not None])
    else:
        num_points_tps = None
    if [f for f in num_points_fns if f is not None]:
        num_points_fns = np.concatenate([f for f in num_points_fns if f is not None])
    else:
        num_points_fns = None
    '''if len([t for t in to_remove_gts if t is not None]):
        to_remove_gts = np.concatenate([t for t in to_remove_gts if t is not None])
        # remove
        gts = gts[~to_remove_gts]
    else:
        to_remove_gts = None
    if len([t for t in to_remove_dts if t is not None]):
        to_remove_dts = np.concatenate([t for t in to_remove_dts if t is not None])
        # remove
        dts = dts[~to_remove_dts]
    else:
        to_remove_dts = None'''

    # Compute summary metrics.
    metrics, fps = summarize_metrics(dts, gts, cfg)
    metrics.loc["AVERAGE_METRICS"] = metrics.mean()
    metrics = metrics.round(NUM_DECIMALS)
    return dts, gts, metrics, num_points_tps, num_points_fns, fps


def summarize_metrics(
    dts: pd.DataFrame,
    gts: pd.DataFrame,
    cfg: DetectionCfg,
) -> pd.DataFrame:
    """Calculate and print the 3D object detection metrics.

    Args:
        dts: (N,14) Table of detections.
        gts: (M,15) Table of ground truth annotations.
        cfg: Detection configuration.

    Returns:
        The summary metrics.
    """
    # Sample recall values in the [0, 1] interval.
    recall_interpolated: NDArrayFloat = np.linspace(0, 1, cfg.num_recall_samples, endpoint=True)

    # Initialize the summary metrics.
    summary = pd.DataFrame(
        {s.value: cfg.metrics_defaults[i] for i, s in enumerate(tuple(MetricNames))}, index=cfg.categories
    )

    average_precisions = pd.DataFrame({t: 0.0 for t in cfg.affinity_thresholds_m}, index=cfg.categories)
    precisions = pd.DataFrame({t: 0.0 for t in cfg.affinity_thresholds_m}, index=cfg.categories)
    for category in cfg.categories:
        # Find detections that have the current category.
        is_category_dts = dts["category"] == category

        # Only keep detections if they match the category and have NOT been filtered.
        is_valid_dts = np.logical_and(is_category_dts, dts["is_evaluated"])

        # Get valid detections and sort them in descending order.
        category_dts = dts.loc[is_valid_dts].sort_values(by="score", ascending=False).reset_index(drop=True)

        # Find annotations that have the current category.
        is_category_gts = gts["category"] == category
        is_valid_gts = np.logical_and(is_category_gts, gts["is_evaluated"])

        # Compute number of ground truth annotations.
        num_gts = gts.loc[is_category_gts, "is_evaluated"].sum()

        # Get matched and valid ground truth and sort them in descending order.
        category_gts = gts.loc[is_valid_gts]

        # Cannot evaluate without ground truth information.
        if num_gts == 0:
            continue

        for affinity_threshold_m in cfg.affinity_thresholds_m:
            true_positives: NDArrayBool = category_dts[affinity_threshold_m].astype(bool).to_numpy()
            if affinity_threshold_m == cfg.tp_threshold_m:
                fps = ~true_positives
            
            '''
            ## GET STATS
            fp_dets = category_dts[~true_positives]
            tp_dets = category_dts[true_positives]
            
            num_pts_fp = fp_dets['num_interior_pts']
            counts, bins = np.histogram(num_pts_fp, bins=np.linspace(0, 2000, 100))
            plt.hist(bins[:-1], bins, weights=counts)

            num_pts_tp = tp_dets['num_interior_pts']
            counts, bins = np.histogram(num_pts_tp, bins=np.linspace(0, 2000, 100))
            plt.hist(bins[:-1], bins, weights=counts)
            plt.title("Number of points")
            plt.legend(['num_pts_fp', 'num_pts_tp'])
            plt.savefig(f'/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/stats/{affinity_threshold_m}_num_pts.png')
            plt.close()

            size_fp = np.abs(fp_dets[['length_m', 'width_m', 'height_m']].values)
            counts, bins = np.histogram(size_fp[:, 0], bins=np.linspace(0, 10, 100))
            plt.hist(bins[:-1], bins, weights=counts)
            counts, bins = np.histogram(size_fp[:, 1], bins=np.linspace(0, 10, 100))
            plt.hist(bins[:-1], bins, weights=counts)
            counts, bins = np.histogram(size_fp[:, 2], bins=np.linspace(0, 5, 100))
            plt.hist(bins[:-1], bins, weights=counts)
            plt.title("Length, Width, Height False Positives")
            plt.legend(['length_m', 'width_m', 'height_m'])
            plt.savefig(f'/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/stats/{affinity_threshold_m}_size_fp.png')
            plt.close()

            size_tp = np.abs(tp_dets[['length_m', 'width_m', 'height_m']].values)
            counts, bins = np.histogram(size_tp[:, 0], bins=np.linspace(0, 10, 100))
            plt.hist(bins[:-1], bins, weights=counts)
            counts, bins = np.histogram(size_tp[:, 1], bins=np.linspace(0, 10, 100))
            plt.hist(bins[:-1], bins, weights=counts)
            counts, bins = np.histogram(size_tp[:, 2], bins=np.linspace(0, 5, 100))
            plt.hist(bins[:-1], bins, weights=counts)
            plt.title("Length, Width, Height True Positives")
            plt.legend(['length_m', 'width_m', 'height_m'])
            plt.savefig(f'/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/stats/{affinity_threshold_m}_size_tp.png')
            plt.close()

            if 'pts_density' in category_dts.columns.values.tolist():
                density_tp = (size_fp[:, 0] * size_fp[:, 1] * size_fp[:, 2]) / num_pts_fp
                counts, bins = np.histogram(density_tp, bins=np.linspace(0, 0.5, 100))
                plt.hist(bins[:-1], bins, weights=counts)

                density_tp = (size_tp[:, 0] * size_tp[:, 1] * size_tp[:, 2]) / num_pts_tp
                counts, bins = np.histogram(density_tp, bins=np.linspace(0, 0.5, 100))
                plt.hist(bins[:-1], bins, weights=counts)
                plt.title(['Point Density'])
                plt.legend(['density_fp', 'density_tp'])
                plt.savefig(f'/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/stats/{affinity_threshold_m}_density_tp.png')
                plt.close()
            
            all_ATE = list()
            all_ASE = list()
            all_AOE = list()
            category_gts_thresh = category_gts[category_gts[affinity_threshold_m] == 0]
            for track_uuid in category_gts_thresh['track_uuid'].unique():
                track_category_gts_thresh = category_gts_thresh[
                    category_gts_thresh['track_uuid'] == track_uuid]
                all_ATE.append(np.var(
                    track_category_gts_thresh['ATE'].values[1:] - \
                        track_category_gts_thresh['ATE'].values[:1]))
                all_ASE.append(np.var(
                    track_category_gts_thresh['ASE'].values[1:] - \
                        track_category_gts_thresh['ASE'].values[:1]))
                all_AOE.append(np.var(
                    track_category_gts_thresh['AOE'].values[1:] - \
                        track_category_gts_thresh['AOE'].values[:1]))

            counts, bins = np.histogram(np.asarray(all_ATE), bins=np.linspace(0, 5, 100))
            plt.hist(bins[:-1], bins, weights=counts)
            plt.title(['ATE'])
            plt.savefig(f'/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/stats/{affinity_threshold_m}_all_ATE.png')
            plt.close()

            counts, bins = np.histogram(np.asarray(all_ASE), bins=np.linspace(0, 5, 100))
            plt.hist(bins[:-1], bins, weights=counts)
            plt.title(['ASE'])
            plt.savefig(f'/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/stats/{affinity_threshold_m}_all_ASE.png')
            plt.close()
            
            counts, bins = np.histogram(np.asarray(all_AOE), bins=np.linspace(0, 5, 100))
            plt.hist(bins[:-1], bins, weights=counts)
            plt.title(['AOE'])
            plt.savefig(f'/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/stats/{affinity_threshold_m}_all_AOE.png')
            plt.close()
            '''
            # print("FPS", category_dts[~true_positives][['log_id','timestamp_ns', 'tx_m', 'ty_m', 'tz_m']])
            # a = gts[np.logical_and(is_category_gts, gts["is_evaluated"])]
            # print("FNS", a[~a[affinity_threshold_m].values.astype(bool)][['log_id','timestamp_ns', 'tx_m', 'ty_m', 'tz_m','is_evaluated', 'num_interior_pts', 'track_uuid']])

            # Continue if there aren't any true positives.
            if len(true_positives) == 0:
                continue
            print(affinity_threshold_m, "TPS: ", true_positives.sum(), "FPS: ", (~true_positives).sum(), "FNS: ", num_gts-(true_positives.sum()), "Num GT: ", num_gts)
            # Compute average precision for the current threshold.
            threshold_average_precision, _ = compute_average_precision(true_positives, recall_interpolated, num_gts)
            # Record the average precision.
            average_precisions.loc[category, affinity_threshold_m] = threshold_average_precision
            precisions.loc[category, affinity_threshold_m] = true_positives.sum()/(true_positives.sum()+(~true_positives).sum())
        print("AVERAGE PRECISIONS \n", average_precisions.loc[category])
        print("PRECISIONS \n", precisions.loc[category])
        mean_average_precisions: NDArrayFloat = average_precisions.loc[category].to_numpy().mean()

        # Select only the true positives for each instance.
        middle_idx = len(cfg.affinity_thresholds_m) // 2
        middle_threshold = cfg.affinity_thresholds_m[middle_idx]
        is_tp_t = category_dts[middle_threshold].to_numpy().astype(bool)

        # Initialize true positive metrics.
        tp_errors: NDArrayFloat = np.array(cfg.tp_normalization_terms)

        # Check whether any true positives exist under the current threshold.
        has_true_positives = np.any(is_tp_t)

        # If true positives exist, compute the metrics.
        if has_true_positives:
            tp_error_cols = [str(x.value) for x in TruePositiveErrorNames]
            tp_errors = category_dts.loc[is_tp_t, tp_error_cols].to_numpy().mean(axis=0)

        # Convert errors to scores.
        tp_scores = 1 - np.divide(tp_errors, cfg.tp_normalization_terms)

        # Compute Composite Detection Score (CDS).
        cds = mean_average_precisions * np.mean(tp_scores)
        summary.loc[category] = np.array([mean_average_precisions, *tp_errors, cds])

    # Return the summary.
    return summary, fps

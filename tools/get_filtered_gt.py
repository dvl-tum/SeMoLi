from SeMoLi.evaluation import eval_detection
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from pathlib import Path
import os
import hydra
from omegaconf import OmegaConf
import shutil


@hydra.main(config_path="../SeMoLi/conf", config_name="conf", version_base=None)
def main(cfg):
    for split in ['train', 'val']:
        gt_folder = os.path.join(cfg.data.data_dir, split)
        loader = AV2SensorDataLoader(data_dir=Path(
                gt_folder), labels_dir=Path(gt_folder))

        eval_detection.get_feather_files(gt_folder,
                is_gt=True,
                remove_far=True,
                remove_non_drive=False,
                remove_non_move=True,
                remove_non_move_strategy='per_frame',
                remove_non_move_thresh=1,
                seq_list=None,
                loader=loader,
                gt_folder=gt_folder,
                root_dir=cfg.root_dir,
                filtered_file_path=cfg.data.filtered_file_path,
                is_waymo='Waymo' in gt_folder)

    shutil.rmtree(f'{cfg.root_dir}/outputs', ignore_errors=True)


if __name__ == "__main__":
    main()

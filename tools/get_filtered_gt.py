from evaluation import eval_detection
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from pathlib import Path
import os


def main(split, gt_folder):
    gt_folder = os.path.join(gt_folder, split)
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
            gt_folder=gt_folder)


if __name__ == "__main__":
    for split in ['train', 'val']:
        gt_folder = f'/workspace/Waymo_Converted_{split}/Waymo_Converted'
        main(split, gt_folder)

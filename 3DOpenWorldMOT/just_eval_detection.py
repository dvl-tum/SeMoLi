from detection_evaluation import eval_detection
import os

'''
_, metric = eval_detection.eval_detection(
                    gt_folder=os.path.join(os.getcwd(), cfg.data.data_dir),
                    trackers_folder=tracker_dir,
                    seq_to_eval=seq_list,
                    remove_far='80' in cfg.data.trajectory_dir,
                    remove_non_drive='non_drive' in cfg.data.trajectory_dir,
                    remove_non_move=cfg.data.remove_static,
                    remove_non_move_strategy=cfg.data.remove_static_strategy,
                    remove_non_move_thresh=3000/3600,
                    classes_to_eval='REGULAR_VEHICLE'
                ) 
'''

def main(
        tracker_folder,
        remove_far,
        remove_non_drive,
        remove_non_move,
        remove_non_move_strategy,
        ):

    seq_list = os.listdir(tracker_folder)

    _, metric = eval_detection.eval_detection(
                    gt_folder='data/argoverse2',
                    trackers_folder=tracker_folder,
                    seq_to_eval=seq_list,
                    remove_far=remove_far,
                    remove_non_drive=remove_non_drive,
                    remove_non_move=remove_non_move,
                    remove_non_move_strategy=remove_non_move_strategy,
                    remove_non_move_thresh=3000/3600,
                    classes_to_eval='REGULAR_VEHICLE',
                    just_eval=True
                ) 

if __name__ == "__main__":
    # DBSCAN
    # out/traj_pos_7.0_5/val/ --> 0.105  0.732  0.626  1.374  0.055
    # out/traj_0.75_5/val/ --> 0.044  0.806  0.688  1.406  0.021
    # out/pos_1.5_5/val/ --> 0.110  0.735  0.625  1.380  0.057
    main(
        'out/traj_0.75_5/val/',
        remove_far=True,
        remove_non_drive=True,
        remove_non_move=True,
        remove_non_move_strategy='per_frame')
import os
from TrackEvalOpenWorld.scripts.run_av2_ow import evaluate_av2_ow_MOT

tracker_dir = '/usr/wiss/seidensc/Documents/3DOpenWorldMOT/3DOpenWorldMOT/outputs/2022-10-18/09-17-33/out/tracker_out'
# seq_list = os.listdir(tracker_dir + '/DBSCAN_TRAJ_POS_all_points_eval_5_1/val')

tracker_dir = '/usr/wiss/seidensc/Documents/3DOpenWorldMOT/3DOpenWorldMOT/outputs/2022-10-18/09-20-32/out/tracker_out'
# seq_list = os.listdir(tracker_dir + '/DBSCAN_TRAJ_POS_all_points_eval_5_6/val')

# tracker_dir = '/usr/wiss/seidensc/Documents/3DOpenWorldMOT/3DOpenWorldMOT/outputs/2022-10-18/09-22-32/out/tracker_out'
# seq_list = os.listdir(tracker_dir + '/DBSCAN_TRAJ_all_points_eval_5_1/val')

# tracker_dir = '/usr/wiss/seidensc/Documents/3DOpenWorldMOT/3DOpenWorldMOT/outputs/2022-10-18/09-23-32/out/tracker_out'
# seq_list = os.listdir(tracker_dir + '/DBSCAN_TRAJ_all_points_eval_5_2/val')

# tracker_dir = '/usr/wiss/seidensc/Documents/3DOpenWorldMOT/3DOpenWorldMOT/outputs/2022-10-18/09-23-41/out/tracker_out'
# seq_list = os.listdir(tracker_dir + '/DBSCAN_TRAJ_all_points_eval_5_3/val')

# DIFFPOS TRAJ
tracker_dir = '/usr/wiss/seidensc/Documents/3DOpenWorldMOT/3DOpenWorldMOT/outputs/2022-10-19/00-25-32/out/tracker_out'
# seq_list = os.listdir(tracker_dir + '/SimpleGraph_TRAJ_POS/val' )

# DIFFTRAJ_DIFFPOS TRAJ
tracker_dir = '/usr/wiss/seidensc/Documents/3DOpenWorldMOT/3DOpenWorldMOT/outputs/2022-10-19/00-30-32/out/tracker_out'

# DIFFTRAJ_DIFFPOS POS
tracker_dir = '/usr/wiss/seidensc/Documents/3DOpenWorldMOT/3DOpenWorldMOT/outputs/2022-10-19/00-36-32/out/tracker_out'

# DIFFTRAJ_DIFFPOS TRAJ_POS
tracker_dir = '/usr/wiss/seidensc/Documents/3DOpenWorldMOT/3DOpenWorldMOT/outputs/2022-10-19/00-49-32/out/tracker_out'

# DIFFTRAJ_DIFFPOS TRAJ_POS
tracker_dir = '/usr/wiss/seidensc/Documents/3DOpenWorldMOT/3DOpenWorldMOT/outputs/2022-10-19/00-59-32/out/tracker_out'


tracker_dir = '/usr/wiss/seidensc/Documents/3DOpenWorldMOT/3DOpenWorldMOT/outputs/2022-10-19/01-52-40/out/tracker_out'

# FIX SEQ LIST
seq_list = os.listdir('/usr/wiss/seidensc/Documents/3DOpenWorldMOT/3DOpenWorldMOT/outputs/2022-10-19/00-25-32/out/tracker_out' + '/SimpleGraph_TRAJ_POS/val' )
seq_list.remove('e1d68dde-22a9-3918-a526-0850b21ff2eb')

gt_folder = '/storage/user/seidensc/datasets/argoverse2'
print(tracker_dir)
print(seq_list)
print(len(seq_list))

output_res, _ = evaluate_av2_ow_MOT(
    gt_folder=gt_folder,
    trackers_folder=tracker_dir,
    seq_to_eval=seq_list,
    remove_far=True,
    remove_non_drive=True,
    do_print=True
    )

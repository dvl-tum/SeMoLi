from pyarrow import feather
import random
import pickle
import os
import shutil
import pandas as pd


def get_samples(file_path='data_utils/new_seq_splits_Waymo_Converted_fixed_val/', split='0.1_val_detector', argo=False):
    file_path = 'data_utils/new_seq_splits_Waymo_Converted_fixed_val/' if not argo else 'data_utils/new_seq_splits_AV2_fixed_val/'
    _argo = '_argo' if argo else ''
    file_path = f'{file_path}{split}.txt'
    sampled_path = os.path.join(f'sampled_for_visualization{_argo}', os.path.basename(file_path)[:-3]+'pkl')
    if os.path.isfile(sampled_path):
        with open(sampled_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
    random.seed = 10
    os.makedirs(f'sampled_for_visualization{_argo}', exist_ok=True)
    with open(file_path, 'r') as f:
        seqs = f.readlines()
        seqs = [s.strip('\n') for s in seqs]
    
    if not argo:
        train_data = feather.read_feather('/workspace/ExchangeWorkspace/Waymo_Converted_filtered/train_1.0_per_frame_remove_non_move_remove_far_filtered_version_city.feather')
    else:
        train_data = feather.read_feather('Argoverse2_filtered/train_1.0_per_frame_remove_non_move_remove_far_filtered_version.feather')
    train_data = train_data[train_data['log_id'].isin(seqs)]
    sampled = dict()
    for seq in seqs:
        timestamps = train_data[train_data['log_id']==seq]['timestamp_ns'].unique().tolist()
        times = random.choices(timestamps, k=3)
        sampled[seq] = times
    with open(sampled_path, 'wb') as save_file: 
        pickle.dump(sampled, save_file)

    return sampled

def get_point_clouds_from_feather_files(sampled, split, argo=False):
    _argo = '_argo' if argo else ''
    if not argo:
        pc_path_whole = '/workspace/Waymo_Converted_train/train/'
    else:
        pc_path_whole = '/workspace/Argoverse2/train/'
    pc_path_whole_filtered = '/workspace/all_egocomp_margin0.6_width25/all_egocomp_margin0.6_width25_train/'
    path_to_store_whole = f'sampled_for_visualization{_argo}/{split}/point_clouds_whole/'
    path_to_store_filtered = f'sampled_for_visualization{_argo}/{split}/point_clouds_filtered/'
    os.makedirs(path_to_store_whole, exist_ok=True)
    os.makedirs(path_to_store_filtered, exist_ok=True)
    # get lidar point clouds
    for seq, timestamps in sampled.items():
        os.makedirs(path_to_store_whole+seq, exist_ok=True)
        os.makedirs(path_to_store_filtered+seq, exist_ok=True)
        for time in timestamps:
            if not os.path.isfile(f'{path_to_store_whole}{seq}/{time}.feather'):
                shutil.copyfile(f'{pc_path_whole}{seq}/sensors/lidar/{time}.feather', f'{path_to_store_whole}{seq}/{time}.feather')
            if not os.path.isfile(f'{path_to_store_filtered}{seq}/{time}.pt'):
                shutil.copyfile(f'{pc_path_whole_filtered}{seq}/{time}.pt', f'{path_to_store_filtered}{seq}/{time}.pt')

def get_detections_from_feather_files(detection_set, sampled, split, argo=False):
    _argo = '_argo' if argo else ''
    data = None
    is_file = os.path.isfile(detection_set)
    if is_file:
        data = feather.read_feather(detection_set)
        save_name = os.path.basename(os.path.dirname(detection_set))
    else:
        save_name = os.path.basename(os.path.dirname(os.path.dirname(detection_set)))
    
    os.makedirs(f'sampled_for_visualization{_argo}/{split}/detections/{save_name}', exist_ok=True)
    sampled_data = None
    for seq, timestamps in sampled.items():
        print(seq, timestamps)
        try:
            if data is None:
                data = feather.read_feather(f'{detection_set}/{seq}/annotations.feather')
            for time in timestamps:
                time_data = data[data['timestamp_ns'] == time]
                if sampled_data is None:
                    sampled_data = time_data
                else:
                    sampled_data = pd.concat([sampled_data, time_data])
        except:
            continue
        if not is_file:
            data = None
    print(f'Store to sampled_for_visualization{_argo}/{split}/detections/{save_name}/detections.feather...')
    feather.write_feather(sampled_data, f'sampled_for_visualization{_argo}/{split}/detections/{save_name}/detections.feather')


def main(detection_set, split, argo=False):
    samples = get_samples(split=split, argo=argo)
    get_point_clouds_from_feather_files(samples, split, argo=argo)
    get_detections_from_feather_files(detection_set, samples, split, argo=argo)


if __name__ == "__main__":
    detection_sets = [
        '/workspace/ExchangeWorkspace/detections_from_pp_sv2_format/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car_feather_fake_gt_three_anchors_0.9_0.1_True_True_train_detector/0.1_val_detector.feather',
        #'/workspace/ExchangeWorkspace/detections_from_pp_sv2_format/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car_feather_fake_gt_three_anchors_0.9_0.1_False_True_train_detector/0.1_val_detector.feather',
        #'/workspace/ExchangeWorkspace/detections_from_pp_sv2_format/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car_GNN_10_feather_ALL_three_anchors_pos_based_0.9_0.1_False_False_train_detector/0.1_val_detector.feather',
        #'/workspace/ExchangeWorkspace/detections_from_pp_sv2_format/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car_GNN_10_feather_ALL_three_anchors_final_vel_0.9_0.1_False_False_train_detector/0.1_val_detector.feather',
        #'/workspace/ExchangeWorkspace/detections_from_pp_sv2_format/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car_feather_fake_gt_class_specific_0.9_0.1_False_False_train_detector/0.1_val_detector.feather'
    ]
    detection_sets = [
            '/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/out/detections_val_gnn/DEBUG_GNN_0.5_0.1_all_egocomp_margin0.6_width25_nooracle_64_3_True_64_3_True_0.5_3.5_0.5_4_0.0001_0.01_16000_16000__NS_MG_32_LN___P___MMMDPTT___PT_/val_gnn/',
            '/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/out/detections_val_gnn/DEBUG0.05_0.5_0.1_all_egocomp_margin0.6_width25_pos_1.0_10/val_gnn/',
            '/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/out/detections_val_gnn/DEBUG_DBSCAN_0.5_0.1_all_egocomp_margin0.6_width25_pos_1.0_10/val_gnn/',
            '/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/out/detections_val_gnn/DEBUG_DBSCAN_0.5_0.1_all_egocomp_margin0.6_width25_pos_1.0_10/val_gnn/',
'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/out/detections_val_gnn/DEBUG_DBSCAN_RESAMPLE_0.5_0.1_all_egocomp_margin0.6_width25_pos_1.0_10/val_gnn/',
'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/out/detections_val_gnn/DEBUG_DBSCAN_RESAMPLE_FLOWMASK_0.5_0.1_all_egocomp_margin0.6_width25_pos_1.0_10/val_gnn/',
'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/out/detections_val_gnn/DEBUG_DBSCAN_RESAMPLE_RADIUS5_0.5_0.1_all_egocomp_margin0.6_width25_pos_1.0_10/val_gnn/',
'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/out/detections_val_gnn/DEBUG_DBSCAN_RESAMPLE_RADIUS5_FLOWMASK_0.5_0.1_all_egocomp_margin0.6_width25_pos_1.0_10/val_gnn/',
'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/out/detections_val_gnn/DEBUG_DBSCAN_RESAMPLE_RADIUS5_MINAREA_0.5_0.1_all_egocomp_margin0.6_width25_pos_1.0_10/val_gnn/',
'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/out/detections_val_gnn/DEBUG_DBSCAN_RESAMPLE_RESAMPLE_0.005_0.5_0.1_all_egocomp_margin0.6_width25_pos_1.0_10/val_gnn/',
'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/out/detections_val_gnn/DEBUG_GNN_ALL_PTS_0.5_0.1_all_egocomp_margin0.6_width25_nooracle_64_3_True_64_3_True_0.5_3.5_0.5_4_0.0001_0.01_16000__NS_MG_32_LN___P___MMMDPTT___PT_/val_gnn/',
'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/out/detections_val_gnn/DEBUG_GNN_ALL_PTS_RESAMPLE_0.005_0.5_0.1_all_egocomp_margin0.6_width25_nooracle_64_3_True_64_3_True_0.5_3.5_0.5_4_0.0001_0.01_16000__NS_MG_32_LN___P___MMMDPTT___PT_/val_gnn/',
'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/out/detections_val_gnn/DEBUG_GNN_ALL_PTS_RESAMPLE_0.5_0.1_all_egocomp_margin0.6_width25_nooracle_64_3_True_64_3_True_0.5_3.5_0.5_4_0.0001_0.01_16000__NS_MG_32_LN___P___MMMDPTT___PT_/val_gnn/',
'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/out/detections_val_gnn/DEBUG_GNN_ALL_PTS_RESAMPLE_FLOWMASK_0.5_0.1_all_egocomp_margin0.6_width25_nooracle_64_3_True_64_3_True_0.5_3.5_0.5_4_0.0001_0.01_16000__NS_MG_32_LN___P___MMMDPTT___PT_/val_gnn/'
            ]
    detection_sets = ['/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/out/detections_val_gnn/ABLATION_AV2_0.1_0.1_all_egocomp_margin0.6_width25_nooracle_64_3_True_64_3_True_0.5_3.5_0.5_4_3.162277660168379e-06_0.0031622776601683794_16000_16000__NS_MG_32_LN___P___MMMDPTT___PT_/val_gnn/']
    detection_sets = [#'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/tracks/tracks_for_eval/registered_dets/DEBUG_GNN_PRECOMP_DETS_0.5_0.1_all_egocomp_margin0.6_width25_nooracle_64_3_True_64_3_True_0.5_3.5_0.5_4_0.0001_0.01_16000_16000__NS_MG_32_LN___P___MMMDPTT___PT_/1000_0.5_0_5_False_10_10_5_False_0.5_10/train/',
            '/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/tracks/tracks_for_eval/tracked_dets/DEBUG_GNN_PRECOMP_DETS_0.5_0.1_all_egocomp_margin0.6_width25_nooracle_64_3_True_64_3_True_0.5_3.5_0.5_4_0.0001_0.01_16000_16000__NS_MG_32_LN___P___MMMDPTT___PT_/1000_0.5_0_5_False_10_10_5_False_0.5_10/train/']
    argo = False
    for detection_set in detection_sets:
        main(split='0.1_val_gnn', detection_set=detection_set, argo=argo)
    




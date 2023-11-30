from pyarrow import feather
import random
import pickle
import os
import shutil
import pandas as pd
import hydra
from omegaconf import OmegaConf


def get_samples(file_path='data_utils/new_seq_splits_Waymo_Converted_fixed_val/', split='0.1_val_detector', argo=False, cfg=None):
    file_path = 'PseudoDetection3D/data_utils/new_seq_splits_Waymo_Converted_fixed_val/' if not argo else 'PseudoDetection3D/data_utils/new_seq_splits_AV2_fixed_val/'
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
        train_data = feather.read_feather('{root_dir}/{filtered_file_path}/train_1.0_per_frame_remove_non_move_remove_far_filtered_version_city.feather')
    else:
        train_data = feather.read_feather('{root_dir}/{filtered_file_path}/train_1.0_per_frame_remove_non_move_remove_far_filtered_version.feather')
    train_data = train_data[train_data['log_id'].isin(seqs)]
    sampled = dict()
    count = 10
    print(train_data['log_id'].unique().shape)
    for i, seq in enumerate(seqs):
        if i == count:
            break
        timestamps = train_data[train_data['log_id']==seq]['timestamp_ns'].unique().tolist()
        print( train_data[train_data['log_id']==seq].shape[0])
        if train_data[train_data['log_id']==seq].shape[0] == 0:
            continue
        times = random.choices(timestamps, k=3)
        sampled[seq] = times
    quit()
    with open(sampled_path, 'wb') as save_file: 
        pickle.dump(sampled, save_file)

    return sampled

def get_point_clouds_from_feather_files(sampled, split, argo=False, cfg=None):
    _argo = '_argo' if argo else ''
    pc_path_whole = cfg.data.data_dir + '_train/train'
    pc_path_whole_filtered = cfg.data.processed_dir + '_train/'
    path_to_store_whole = f'{cfg.root_dir}/sampled_for_visualization{_argo}/{split}/point_clouds_whole/'
    path_to_store_filtered = f'{cfg.root_dir}/sampled_for_visualization{_argo}/{split}/point_clouds_filtered/'
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

def get_detections_from_feather_files(detection_set, sampled, split, argo=False, cfg=None):
    _argo = '_argo' if argo else ''
    data = None
    is_file = os.path.isfile(detection_set)
    if is_file:
        data = feather.read_feather(detection_set)
        save_name = os.path.basename(os.path.dirname(detection_set))
    else:
        save_name = os.path.basename(os.path.dirname(os.path.dirname(detection_set)))
    
    os.makedirs(f'{cfg.root_dir}/sampled_for_visualization{_argo}/{split}/detections/{save_name}', exist_ok=True)
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
    print(f'Store to {cfg.root_dir}/sampled_for_visualization{_argo}/{split}/detections/{save_name}/detections.feather...')
    feather.write_feather(sampled_data, f'{cfg.root_dir}/sampled_for_visualization{_argo}/{split}/detections/{save_name}/detections.feather')


@hydra.main(config_path="PseudoDetection3D/conf", config_name="conf")
def main(cfg):
    split = f'{cfg.data.percentage_data_val}_{cfg.data.detection_set}'
    argo = cfg.data.datset_name == 'av2_traj'
    samples = get_samples(split=split, argo=argo, cfg=cfg)
    get_point_clouds_from_feather_files(samples, split, argo=argo, cfg=cfg)
    get_detections_from_feather_files(cfg.sample_vis.detection_path, samples, split, argo=argo)


if __name__ == "__main__":
    main()




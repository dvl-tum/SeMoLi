from multiprocessing.pool import Pool
import os
import pandas as pd 
import glob
from pyarrow import feather
import torch
from av2.structures.cuboid import CuboidList


def just_get_interior_pts():
    data_feather = None
    paths = glob.glob('/workspace/all_egocomp_margin0.6_width25/all_egocomp_margin0.6_width25_train/*/*')
    for i, p in enumerate(paths):
        print(f'{i}/{len(paths)}')
        if i % 100 == 0:
            print(f'{i}/{len(paths)}')
            if os.path.isfile('/workspace/ExchangeWorkspace/Waymo_Converted_filtered/train_1.0_per_frame_remove_non_move_remove_far_filtered_version_city_w0_withwaymovel_w_num_interior_filtered.feather'):
                _data_feather = feather.read_feather('/workspace/ExchangeWorkspace/Waymo_Converted_filtered/train_1.0_per_frame_remove_non_move_remove_far_filtered_version_city_w0_withwaymovel_w_num_interior_filtered.feather')
                print(data_feather.shape, _data_feather.shape)
                data_feather = pd.concat([data_feather, _data_feather])
                print(data_feather.shape)
            if data_feather is not None:
                feather.write_feather(data_feather, '/workspace/ExchangeWorkspace/Waymo_Converted_filtered/train_1.0_per_frame_remove_non_move_remove_far_filtered_version_city_w0_withwaymovel_w_num_interior_filtered.feather')
                data_feather = None

        data = torch.load(p)
        lidar_timestamp_ns = int(data['timestamps'][0])
        log_id = data['log_id']
        annotations_feather_path = '/workspace/ExchangeWorkspace/Waymo_Converted_filtered/train_1.0_per_frame_remove_non_move_remove_far_filtered_version_city_w0_withwaymovel.feather'
        _data = CuboidList._get_num_interior_from_feather_all( annotations_feather_path, log_id, lidar_timestamp_ns, get_moving=False, pc=data['pc_list'].numpy())
        if data_feather is None:
            data_feather = _data
        else:
            data_feather = pd.concat([data_feather, _data])
    feather.write_feather(data_feather, '/workspace/ExchangeWorkspace/Waymo_Converted_filtered/train_1.0_per_frame_remove_non_move_remove_far_filtered_version_city_w0_withwaymovel_w_num_interior_filtered.feather')
    
    data_feather = None
    paths = glob.glob('/workspace/all_egocomp_margin0.6_width25/all_egocomp_margin0.6_width25_val/*/*')
    for i, p in enumerate(paths):
        if i % 100 == 0:
            print(f'{i}/{len(paths)}')
            if os.path.isfile('/workspace/ExchangeWorkspace/Waymo_Converted_filtered/val_1.0_per_frame_remove_non_move_remove_far_filtered_version_city_w0_withwaymovel_w_num_interior_filtered.feather'):
                _data_feather = feather.read_feather('/workspace/ExchangeWorkspace/Waymo_Converted_filtered/val_1.0_per_frame_remove_non_move_remove_far_filtered_version_city_w0_withwaymovel_w_num_interior_filtered.feather')
                data_feather = pd.concat([data_feather, _data_feather])
            if data_feather is not None:
                feather.write_feather(data_feather, '/workspace/ExchangeWorkspace/Waymo_Converted_filtered/val_1.0_per_frame_remove_non_move_remove_far_filtered_version_city_w0_withwaymovel_w_num_interior_filtered.feather')
                data_feather = None

        data = torch.load(p)
        lidar_timestamp_ns = int(data['timestamps'][0])
        log_id = data['log_id']
        annotations_feather_path = '/workspace/ExchangeWorkspace/Waymo_Converted_filtered/val_1.0_per_frame_remove_non_move_remove_far_filtered_version_city_w0_withwaymovel.feather'
        _data = CuboidList._get_num_interior_from_feather_all( annotations_feather_path, log_id, lidar_timestamp_ns, get_moving=False, pc=data['pc_list'].numpy())
        if data_feather is None:
            data_feather = _data
        else:
            data_feather = pd.append([data_feather, _data])
    feather.write_feather(data_feather, '/workspace/ExchangeWorkspace/Waymo_Converted_filtered/val_1.0_per_frame_remove_non_move_remove_far_filtered_version_city_w0_withwaymovel_w_num_interior_filtered.feather')


if __name__ == "__main__":
    just_get_interior_pts()

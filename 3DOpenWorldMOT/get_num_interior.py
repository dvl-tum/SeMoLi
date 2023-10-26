from multiprocessing.pool import Pool
import os
import pandas as pd 
import glob
from pyarrow import feather
import torch
from av2.structures.cuboid import CuboidList
from functools import partial


def just_get_interior_pts(path, out_path, out_file, annotations_feather_path):
    seqs = glob.glob(f'{path}/*')
    
    annotations_feather_path = feather.read_feather(annotations_feather_path)
    print('Loaded...', annotations_feather_path)
    data_loader = [(i, seq, path, out_path, annotations_feather_path[annotations_feather_path['log_id']==seq]) for i, seq in enumerate(seqs)]
    with Pool() as pool:
        _get_interior_one_seq = partial(get_interior_one_seq)
        pool.map(_get_interior_one_seq, data_loader)
    
    files = glob.glob(f'{out_path}/num_interior/workspace/all_egocomp_margin0.6_width25/all_egocomp_margin0.6_width25_train/*/with_num_interior.feather')
    data_feather = None
    for p in files:
        _data = feather.read_feather(p)
        if data_feather is None:
            data_feather = _data
        else:
            data_feather = pd.concat([data_feather, _data])
    print(f'{out_path}/{out_file}')
    print()
    feather.write_feather(data_feather, f'{out_path}/{out_file}')
    print('wrote :)')

def get_interior_one_seq(inp):
    i, seq, path, out_path, annotations_feather_path = inp
    if os.path.isfile(f'{out_path}/num_interior/{seq}/with_num_interior.feather'):
        return
    print(i)
    if i % 50 == 0:
        print(i)
    timestamps = glob.glob(f'{seq}/*')
    data_feather = None
    for j, time_p in enumerate(timestamps):
        if j % 20 and i % 50 == 0:
            print(i, j, len(timestamps))
        data = torch.load(time_p)
        lidar_timestamp_ns = int(data['timestamps'][0])
        log_id = data['log_id']
        _data = CuboidList._get_num_interior_from_feather_all( annotations_feather_path, log_id, lidar_timestamp_ns, get_moving_only=False, pc=data['pc_list'].numpy())
        if data_feather is None:
            data_feather = _data
        else:
            data_feather = pd.concat([data_feather, _data])
    os.makedirs(f'{out_path}/num_interior/{seq}', exist_ok=True)
    print('write to', f'{out_path}/num_interior/{seq}/with_num_interior.feather')
    feather.write_feather(data_feather, f'{out_path}/num_interior/{seq}/with_num_interior.feather')

if __name__ == "__main__":
    for split in ['train', 'val']:
        path = f'/workspace/all_egocomp_margin0.6_width25/all_egocomp_margin0.6_width25_{split}'
        out_path = f'/workspace/ExchangeWorkspace/Waymo_Converted_filtered'
        out_file = f'{split}_1.0_per_frame_remove_non_move_remove_far_filtered_version_city_w0_withwaymovel_w_num_interior_filtered.feather'
        annotations_feather_path = f'/workspace/ExchangeWorkspace/Waymo_Converted_filtered/{split}_1.0_per_frame_remove_non_move_remove_far_filtered_version_city_w0_withwaymovel.feather'
        just_get_interior_pts(path, out_path, out_file, annotations_feather_path)

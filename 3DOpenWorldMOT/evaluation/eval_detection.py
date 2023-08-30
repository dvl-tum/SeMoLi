# FOR DETECTION EVALUATION
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from av2.evaluation.detection.eval import evaluate
from av2.evaluation.detection.utils import DetectionCfg
from pathlib import Path
from av2.utils.io import read_feather, read_all_annotations
import numpy as np
import os
from collections import defaultdict
from pathlib import Path
from av2.map.map_api import ArgoverseStaticMap, RasterLayerType
from pyarrow import feather
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib as mpl
from torch import multiprocessing as mp
import pandas as pd
from scipy.spatial.transform import Rotation


_class_dict = {
    1: 'REGULAR_VEHICLE',
    2: 'PEDESTRIAN',
    3: 'BOLLARD',
    4: 'CONSTRUCTION_CONE',
    5: 'CONSTRUCTION_BARREL',
    6: 'STOP_SIGN',
    7: 'BICYCLE',
    8: 'LARGE_VEHICLE',
    9: 'WHEELED_DEVICE',
    10: 'BUS',
    11: 'BOX_TRUCK',
    12: 'SIGN',
    13: 'TRUCK',
    14: 'MOTORCYCLE',
    15: 'BICYCLIST',
    16: 'VEHICULAR_TRAILER',
    17: 'TRUCK_CAB',
    18: 'MOTORCYCLIST',
    19: 'DOG',
    20: 'SCHOOL_BUS',
    21: 'WHEELED_RIDER',
    22: 'STROLLER',
    23: 'ARTICULATED_BUS',
    24: 'MESSAGE_BOARD_TRAILER',
    25: 'MOBILE_PEDESTRIAN_SIGN',
    26: 'WHEELCHAIR',
    27: 'RAILED_VEHICLE',
    28: 'OFFICIAL_SIGNALER',
    29: 'TRAFFIC_LIGHT_TRAILER',
    30: 'ANIMAL',
    31: 'MOBILE_PEDESTRIAN_CROSSING_SIGN'}

class_dict = {v: k for k, v in _class_dict.items()}

WAYMO_CLASSES = {'TYPE_UNKNOWN': 0, 'TYPE_VECHICLE': 1,
                 'TYPE_PEDESTRIAN': 2, 'TYPE_SIGN': 12, 'TYPE_CYCLIST': 15}


def quat_to_mat(quat_wxyz):
    """Convert a quaternion to a 3D rotation matrix.

    NOTE: SciPy uses the scalar last quaternion notation. Throughout this repository,
        we use the scalar FIRST convention.

    Args:
        quat_wxyz: (...,4) array of quaternions in scalar first order.

    Returns:
        (...,3,3) 3D rotation matrix.
    """
    # Convert quaternion from scalar first to scalar last.
    quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
    mat = Rotation.from_quat(quat_xyzw).as_matrix()
    return mat


def get_feather_files(
        paths,
        is_gt=True,
        remove_far=True,
        remove_non_drive=True,
        remove_non_move=True,
        remove_non_move_strategy='per_frame',
        remove_non_move_thresh=3000/3600,
        seq_list=None,
        loader=None,
        gt_folder=None,
        classes_to_eval='all'):
    
    if is_gt:
        # get file name
        split_dir = paths
        split = os.path.basename(paths)
        split_dir = os.path.dirname(os.path.dirname(paths)) + '_filtered'
        split_dir = os.path.dirname(paths) + '_filtered'

        file = 'filtered_version.feather'
        file = 'remove_non_drive_' + file if remove_non_drive else file
        file = 'remove_far_' + file if remove_far else file
        file = 'remove_non_move_' + file if remove_non_move else file
        file = remove_non_move_strategy + '_' + file if remove_non_move else file
        file = str(remove_non_move_thresh)[
            :5] + '_' + file if remove_non_move else file
        file = split + '_' + file

        path_filtered = os.path.join(split_dir, file)
        path_filtered = '/dvlresearch/jenny/Waymo_Converted_filtered_val/val_1.0_per_frame_remove_non_move_remove_far_filtered_version.feather'
        # path_filtered = "/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/Waymo_Converted_filtered_val/val_1.0_per_frame_remove_non_move_remove_far_filtered_version.feather"
    if not is_gt or not os.path.isfile(path_filtered):
        df = None
        for i, path in enumerate(os.listdir(paths)):
            if i % 50 == 0:
                print(f'load {i}/{len(os.listdir(paths))}')
            if seq_list is not None:
                if path not in seq_list and not is_gt:
                    continue
            data = feather.read_feather(
                os.path.join(paths, path, 'annotations.feather'))

            if 'Argo' in paths or not is_gt:
                def convert2int(x): return class_dict[x]
                data['category'] = data['category'].apply(convert2int)
            else:
                def str2ing(x): return int(x)
                data['category'] = data['category'].apply(str2ing)

            if classes_to_eval != 'all':
                data = data[data['category'] == class_dict[classes_to_eval]]
            data['log_id'] = [path] * data.shape[0]

            if df is None:
                df = data
            else:
                df = df.append(data)

        df = df.astype({'num_interior_pts': 'int64'})

        if not is_gt:
            if 'score' not in df.columns:
                # np.max(df['num_interior_pts'].values)/df['num_interior_pts'].values # [1] * df.shape[0]
                df['score'] = [1] * df.shape[0]

    # filter out objects in non-drivable area or far away
    if is_gt:
        # check if filtered version already exists
        if os.path.isfile(path_filtered):
            df = feather.read_feather(path_filtered)

            if 'seq' in df.columns:
                df.rename(columns={'seq': 'log_id'}, inplace=True)

            if classes_to_eval != 'all':
                df = df[df['category'] == class_dict[classes_to_eval]]

            if seq_list is not None:
                df = df[df['log_id'].isin(seq_list)]

            df = df.astype({'num_interior_pts': 'int64'})
            return df

        # iterate over sequences
        num_seqs = df['log_id'].unique().shape[0]
        data_loader = [
            [log_id, num_seqs, remove_non_move, remove_non_move_strategy, loader, remove_non_move_thresh, remove_non_drive, paths, remove_far, df] for log_id in df['log_id'].unique()]

        data_loader = enumerate(data_loader)

        all_filtered = list()
        mp.set_start_method('forkserver')
        with mp.Pool() as pool:
            filtered = pool.map(filter_seq, data_loader, chunksize=None)
            all_filtered.append(filtered)

        all_filtered = all_filtered[0]
        filtered = None

        for seq_df in all_filtered:
            if filtered is None and seq_df.shape[0]:
                filtered = seq_df
            elif filtered is not None:
                filtered = filtered.append(seq_df)
        
        # store filtered df
        df = filtered

        with open(path_filtered, 'wb') as f:
            feather.write_feather(df, f)
    
        if seq_list is not None:
            df = df[df['log_id'].isin(seq_list)]

    return df

def filter_seq(data, width=25):
    # generate filtered version
    filtered = None
    map_dict = dict()
    # unpack input
    m, (seq, num_seqs, remove_non_move, remove_non_move_strategy, loader, remove_non_move_thresh, remove_non_drive, path, remove_far, df) = data
    print(f'Sequence {m}/{num_seqs}...')
    seq_df = df[df['log_id'] == seq]
    timestamps = sorted(seq_df['timestamp_ns'].unique().tolist())

    if remove_non_move and remove_non_move_strategy == 'per_seq':
        centroid_dict = defaultdict(list)
        movers = list()
        centroid_time = defaultdict(list)
        for i, t in enumerate(sorted(seq_df['timestamp_ns'].unique().tolist())):
            labels = loader.get_labels_at_lidar_timestamp(
                log_id=seq, lidar_timestamp_ns=int(t))

            city_SE3_ego = loader.get_city_SE3_ego(seq, int(t))

            labels = labels.transform(city_SE3_ego)
            for label in labels:
                centroid_dict[label.track_id].append(
                    label.dst_SE3_object.translation)
                centroid_time[label.track_id].append(int(t))

        for track, centroids in centroid_dict.items():
            diff_cent = np.linalg.norm(np.asarray(
                centroids)[:-1, :-1] - np.asarray(centroids)[1:, :-1], axis=1)
            diff_time = np.asarray(centroid_time[track])[
                1:] - np.asarray(centroid_time[track])[:-1]
            diff_time = diff_time / np.power(10, 9)
            if np.mean(diff_cent/diff_time) > remove_non_move_thresh:
                movers.append(track)

    # iterate over timesyeps to get argoverse map and labels
    timestamp_list = sorted(seq_df['timestamp_ns'].unique().tolist())
    for i, t in enumerate(timestamp_list):
        bool_labels = list()
        if i > len(timestamp_list) - width:
            break
        track_ids = list()
        time_df = seq_df[seq_df['timestamp_ns'] == t]

        # get labels
        labels = loader.get_labels_at_lidar_timestamp(
            log_id=seq, lidar_timestamp_ns=int(t))

        # remove labels that are non in dirvable area
        if remove_non_drive and 'Argo' in path:
            if seq not in map_dict.keys():
                log_map_dirpath = Path(gt_folder) / seq / "map"
                map_dict[seq] = ArgoverseStaticMap.from_map_dir(
                    log_map_dirpath, build_raster=True)
            centroids_ego = np.asarray(
                [label.dst_SE3_object.translation for label in labels])
            city_SE3_ego = loader.get_city_SE3_ego(seq, int(t))
            centroids_city = city_SE3_ego.transform_point_cloud(
                centroids_ego)
            bool_labels = map_dict[seq].get_raster_layer_points_boolean(
                centroids_city, layer_name=RasterLayerType.DRIVABLE_AREA)
            labels = [l for i, l in enumerate(
                labels) if bool_labels[i]]

        if remove_non_move:
            if remove_non_move_strategy == 'per_seq':
                labels = [
                    label for label in labels if label.track_id in movers]
            elif remove_non_move_strategy == 'per_frame':

                if i < len(timestamps) - 1:
                    # labels at t+1
                    labels_t2 = loader.get_labels_at_lidar_timestamp(
                        log_id=seq, lidar_timestamp_ns=int(timestamps[i+1]))
                    city_SE3_t2 = loader.get_city_SE3_ego(
                        seq, int(timestamps[i+1]))
                    ids_t2 = [label.track_id for label in labels_t2]
                else:
                    labels_t2 = list()
                    ids_t2 = list()

                city_SE3_t1 = loader.get_city_SE3_ego(
                    seq, int(timestamps[i]))
                vels = list()

                ego_traj_SE3_ego_ref = city_SE3_t2.inverse().compose(city_SE3_t1)

                for m, label in enumerate(labels):
                    center = label.dst_SE3_object.translation
                    if len(labels_t2) and label.track_id in ids_t2:
                        # Pose of the object in the destination reference frame.
                        # ego_SE3_object --> from object to ego   
                        ego_traj_SE3_obj_traj = labels_t2[ids_t2.index(
                            label.track_id)].dst_SE3_object
                        ego_ref_SE3_obj_ref = label.dst_SE3_object

                            # transform points belonging to object in t1 into ego t2 coordinate system
                        obj_ref_ego_traj = ego_traj_SE3_ego_ref.transform_point_cloud(
                            center)
                        
                        # transform points belonging to object in t1 into obj and from obj to ego t2 (assmue obj same points)
                        obj_traj_ego_traj = ego_traj_SE3_obj_traj.compose(
                            ego_ref_SE3_obj_ref.inverse()).transform_point_cloud(center)

                        # get flow
                        translation = obj_traj_ego_traj - obj_ref_ego_traj
                        dist = np.linalg.norm(translation)
                        if 'Argo' in path:
                            diff_time = (
                                timestamps[i+1]-t) / np.power(10, 9)
                        else:
                            diff_time = (
                                timestamps[i+1]-t) / np.power(10, 6)
                        vel = dist/diff_time
                        vels.append(vel)
                        bool_labels.append(
                            vel > remove_non_move_thresh)
                    else:
                        vels.append(None)
                        bool_labels.append(False)
                # filter labels
                # labels = [l for i, l in enumerate(
                #     labels) if bool_labels[i]]
            else:
                assert remove_non_move_strategy in [
                    'per_frame', 'per_seq'], 'remove strategy for static objects not defined'

        # remove points that are far away (>80,)
        if remove_far and len(labels):
            all_centroids = np.asarray(
                [label.dst_SE3_object.translation for label in labels])
            all_centroids = np.atleast_2d(all_centroids)
            dists_to_center = np.sqrt(np.sum(all_centroids ** 2, 1))
            ind = np.where(dists_to_center <= 80)[0]
            labels = [labels[i] for i in ind]
            bool_labels = [bool_labels[i] for i in ind]

        # get track id of remaining objects
        track_ids = [l.track_id for l in labels]
        bool_labels = {l.track_id: b for l, b in zip(labels, bool_labels)}

        # filter time df by track ids
        time_df = time_df[time_df['track_uuid'].isin(track_ids)]
        time_df['filter_moving'] = [bool_labels[t] for t in time_df['track_uuid'].values]

        if filtered is None and time_df.shape[0]:
            filtered = time_df
        elif filtered is not None:
            filtered = filtered.append(time_df)

    return filtered


def visualize_whole(df, gf, name, base_dir='../../../'):
    split_dir = Path('/dvlresearch/jenny/Waymo_Converted_GT/val')
    split_dir = Path('/workspace/Waymo_Converted_val/val')
    loader = AV2SensorDataLoader(data_dir=split_dir, labels_dir=split_dir)
    for seq in df['log_id'].unique():

        ddf = df[df['log_id'] == seq]
        gdf = gf[gf['log_id'] == seq]

        os.makedirs(f'{base_dir}Visualization_Whole_DETS/{seq}', exist_ok=True)
        
        # get lims of visualization in city coords
        lims_mins = list()
        lims_maxs = list()
        if gdf.shape[0] == 0:
            to_use = ddf
        else:
            to_use = gdf
        
        '''for i, timestamp in enumerate(sorted(to_use['timestamp_ns'].unique())):
            city_SE3_ego = loader.get_city_SE3_ego(seq, int(timestamp))
            time_gdf = to_use[to_use['timestamp_ns'] == timestamp]
            gdf_city = city_SE3_ego.transform_point_cloud(
                        time_gdf[['tx_m', 'ty_m', 'tz_m']].values)
            if not gdf_city.shape[0]:
                continue
            mins = np.min(gdf_city, axis=0)
            maxs = np.max(gdf_city, axis=0)
            lims_mins.append(mins)
            lims_maxs.append(maxs)
        lims_mins = np.vstack(lims_mins)
        lims_maxs = np.vstack(lims_maxs)
        mins = np.min(lims_mins, axis=0)
        maxs = np.max(lims_maxs, axis=0)
        x_lim = (mins[0] - 10,
                 maxs[0] + 10)
        y_lim = (mins[1] - 10,
                 maxs[1] + 10)'''
        # visualize for every timestamp
        for i, timestamp in enumerate(set(sorted(gdf['timestamp_ns'].unique().tolist()+ddf['timestamp_ns'].unique().tolist()))):
            if i % 10 == 0:
                print(i, timestamp)
            city_SE3_ego = loader.get_city_SE3_ego(seq, int(timestamp))
            
            # transform timestamp into city coords
            time_ddf = ddf[ddf['timestamp_ns'] == timestamp]
            time_gdf = gdf[gdf['timestamp_ns'] == timestamp]
            ddf_ego = time_ddf[['tx_m', 'ty_m', 'tz_m']].values
            gdf_ego = time_gdf[['tx_m', 'ty_m', 'tz_m']].values
            '''ddf_city = city_SE3_ego.transform_point_cloud(
                        time_ddf[['tx_m', 'ty_m', 'tz_m']].values)
            gdf_city = city_SE3_ego.transform_point_cloud(
                        time_gdf[['tx_m', 'ty_m', 'tz_m']].values)'''

            # visualize detections
            fig, ax = plt.subplots()
            j = 0
            for i, row in time_ddf.iterrows():
                plt.scatter(ddf_ego[j, 0], ddf_ego[j, 1], color='black', marker='o', s=2)
                x_0 = ddf_ego[j, 0]-0.5*row['length_m']
                y_0 = ddf_ego[j, 1]-0.5*row['width_m']

                mat = quat_to_mat(np.array([row['qw'], row['qx'], row['qy'], row['qz']]))
                alpha = np.arccos(mat[0, 0])
                rect = patches.Rectangle(
                    (x_0, y_0),
                    row['length_m'],
                    row['width_m'],
                    linewidth=1,
                    edgecolor='black',
                    facecolor='none',
                    angle=alpha)

                ax.add_patch(rect)
                j += 1
            
            # visualize ground truth
            j = 0
            for i, row in time_gdf.iterrows(): 
                if row['num_interior_pts'] < 5:
                    color = 'green'
                elif row['num_interior_pts'] < 10 and  row['num_interior_pts'] >= 5:
                    color = 'blue'
                elif row['num_interior_pts'] < 15 and  row['num_interior_pts'] >= 10:
                    color = 'pink'
                elif row['num_interior_pts'] < 20 and  row['num_interior_pts'] >= 15:
                    color = 'red'
                elif row['num_interior_pts'] < 25 and  row['num_interior_pts'] >= 20:
                    color = 'orange'
                else:
                    color = 'grey'

                plt.scatter(gdf_ego[j, 0], gdf_ego[j, 1],
                            color=color, marker='*', s=2)
                x_0 = gdf_ego[j, 0]-0.5*row['length_m']
                y_0 = gdf_ego[j, 1]-0.5*row['width_m']

                mat = quat_to_mat(np.array([row['qw'], row['qx'], row['qy'], row['qz']]))
                alpha = np.arccos(mat[0, 0])
                rect = patches.Rectangle(
                    (x_0, y_0),
                    row['length_m'],
                    row['width_m'],
                    linewidth=1,
                    edgecolor=color,
                    facecolor='none',
                    angle=alpha)

                ax.add_patch(rect)
                j += 1
            mins_mas = np.vstack([gdf_ego, ddf_ego])
            mins = np.min(mins_mas, axis=0)
            maxs = np.max(mins_mas, axis=0)
            x_lim = [mins[0]- 10, maxs[0]+10]
            y_lim = [mins[1]-10, maxs[1]+10]
            # plt.axis('off')
            ax.axis('equal')
            plt.xlim(x_lim)
            plt.ylim(y_lim)
            plt.savefig(
                f'{base_dir}Visualization_Whole_DETS/{seq}/frame_{timestamp}_{name}.jpg', dpi=1000)
            plt.close()


def eval_detection(
        gt_folder,
        trackers_folder,
        seq_to_eval,
        remove_far=True,
        remove_non_drive=True,
        remove_non_move=True,
        remove_non_move_strategy='per_frame',
        remove_non_move_thresh=3000/3600,
        split='val',
        classes_to_eval='all',
        visualize=False,
        debug=False,
        name='General',
        just_eval=False,
        min_points=-1,
        max_points=1000000,
        base_dir='../../../',
        print_detail=False,
        filter_class=None,
        eval_only_machted=False,
        filter_moving_before=False):

    if not len(seq_to_eval):
        return None, np.array([0, 2, 1, 3.142, 0])

    gt_folder = os.path.join(gt_folder, split)
    loader = AV2SensorDataLoader(data_dir=Path(
        gt_folder), labels_dir=Path(gt_folder))
    dataset_dir = Path(gt_folder)
    eval_only_roi_instances = False if 'waymo' in gt_folder or 'Waymo' in gt_folder else True

    if just_eval:
        print("Loading data...")

    gts = get_feather_files(
        gt_folder,
        is_gt=True,
        remove_far=remove_far,
        remove_non_drive=remove_non_drive,
        remove_non_move=remove_non_move,
        remove_non_move_strategy=remove_non_move_strategy,
        remove_non_move_thresh=remove_non_move_thresh,
        seq_list=seq_to_eval,
        loader=loader,
        gt_folder=gt_folder,
        classes_to_eval=classes_to_eval)

    if just_eval:
        print("Loaded ground truth...")

    dts = get_feather_files(
        trackers_folder,
        seq_list=seq_to_eval,
        is_gt=False,
        loader=loader,
        gt_folder=gt_folder,
        classes_to_eval=classes_to_eval)

    dts = dts[np.logical_and(dts['height_m'] > 0.1, 
                             np.logical_and(dts['length_m'] > 0.1, dts['width_m'] > 0.1))]

    # dts = dts[dts['num_interior_pts'] > 50]
    # dts = dts[dts['height_m'] < 3]
    # dts = dts[dts['width_m'] < 4]
    # dts = dts[dts['length_m'] < 7]
    print('num dts', dts.shape)

    if just_eval:
        print("Loaded detections...")

    if dts is None:
            return None, np.array([0, 2, 1, 3.142, 0])
    print('filter class', filter_class)
    if filter_class is None:
        gts['category_int'] = [1] * gts.shape[0] # gts['category']
        filter_class = 'REGULAR_VEHICLE'
    else:
        gts['category_int'] = gts['category']
    
    gts['category'] = [_class_dict[c] for c in gts['category']]
    dts['category'] = [_class_dict[c] for c in dts['category']]
    gts['category'] = ['REGULAR_VEHICLE'] * gts.shape[0]

    print(f'Min points {min_points}, Max points {max_points}')

    if just_eval:
        print("Loaded ground truth...")

    if just_eval:
        print("Evaluate now...")

    if filter_moving_before:
       gts = gts[gts['filter_moving']]

    gts_orig = gts
    dts_orig = dts
    for affinity, tp_thresh, threshs in zip(['CENTER', 'IoU3D'], [4.0, 0.6], [(0.5, 1.0, 2.0, 4.0), (0.2, 0.4, 0.6, 0.8)]):
        # Evaluate instances.
        # Defaults to competition parameters.
        competition_cfg = DetectionCfg(
            dataset_dir=dataset_dir, 
            eval_only_roi_instances=eval_only_roi_instances, 
            tp_threshold_m=tp_thresh,
            affinity_type=affinity,
            affinity_thresholds_m=threshs
            )
        print(f"{affinity}\n")
        dts, gts, metrics, num_points_tps, num_points_fns, _ = evaluate(
            dts_orig, gts_orig, cfg=competition_cfg, min_points=min_points, max_points=max_points, filter_class=class_dict[filter_class], eval_only_machted=eval_only_machted)

        dts = dts[dts['is_evaluated']==1]
        gts = gts[gts['is_evaluated']==1]
        print('shapes after', gts.shape, dts.shape)

        if print_detail:
            # remove gt objects without lidar points inside
            print('All GT objects: ', gts.shape[0])
            print('GT objects with 0 points: ', gts[gts['num_interior_pts'] == 0].shape[0])
            print('GT objects with less than 5 points and more than 0: ', gts[np.logical_and(gts['num_interior_pts'] < 5, gts['num_interior_pts'] >= 0)].shape[0])
            print('GT objects with less than 10 points and more than 5: ', gts[np.logical_and(gts['num_interior_pts'] < 10, gts['num_interior_pts'] >= 5)].shape[0])
            print('GT objects with less than 15 points and more than 10: ', gts[np.logical_and(gts['num_interior_pts'] < 15, gts['num_interior_pts'] >= 10)].shape[0])
            print('GT objects with less than 20 points and more than 15: ', gts[np.logical_and(gts['num_interior_pts'] < 20, gts['num_interior_pts'] >= 15)].shape[0])
            print('GT objects with less than 25 points and more than 20: ', gts[np.logical_and(gts['num_interior_pts'] < 25, gts['num_interior_pts'] >= 20)].shape[0])
            print('GT objects with more than 25 points: ', gts[gts['num_interior_pts'] >= 25].shape[0])

        if visualize and dts.shape[0] and gts.shape[0] and affinity == 'CENTER':
            visualize_whole(dts, gts[gts['num_interior_pts']>0], name, base_dir)

        if print_detail:
            print(f'Less than 5 points and more than 0: TP {num_points_tps[np.logical_and(num_points_tps < 5, num_points_tps >= 0)].shape[0]}, FN {num_points_fns[np.logical_and(num_points_fns < 5, num_points_fns >= 0)].shape[0]}')
            print(f'Less than 10 points and more than 5: TP {num_points_tps[np.logical_and(num_points_tps < 10, num_points_tps >= 5)].shape[0]}, FN {num_points_fns[np.logical_and(num_points_fns < 10, num_points_fns >= 5)].shape[0]}')
            print(f'Less than 15 points and more than 10: TP {num_points_tps[np.logical_and(num_points_tps < 15, num_points_tps >= 10)].shape[0]}, FN {num_points_fns[np.logical_and(num_points_fns < 15, num_points_fns >= 10)].shape[0]}')
            print(f'Less than 20 points and more than 15: TP {num_points_tps[np.logical_and(num_points_tps < 20, num_points_tps >= 15)].shape[0]}, FN {num_points_fns[np.logical_and(num_points_fns < 20, num_points_fns >= 15)].shape[0]}')
            print(f'Less than 25 points and more than 20: TP {num_points_tps[np.logical_and(num_points_tps < 25, num_points_tps >= 20)].shape[0]}, FN {num_points_fns[np.logical_and(num_points_fns < 25, num_points_fns >= 20)].shape[0]}')
            print(f'More than 25 points: TP {num_points_tps[num_points_tps >= 25].shape[0]}, FN {num_points_fns[num_points_fns >= 25].shape[0]}')

        # AP    ATE    ASE    AOE    CDS
        classes_to_eval = 'REGULAR_VEHICLE'
        if classes_to_eval == 'all':
            metric = metrics.loc['AVERAGE_METRICS'].values
        else:
            print(metrics.loc[classes_to_eval])
            metric = metrics.loc[classes_to_eval].values

        # break
  
    return metrics, metric


if __name__ == '__main__':
    name ='just_eval'
    tracker_dir =  '4449931/out/gt_all_egocomp_margin0.6_width25_nooracle_4096_8000_mean_dist_over_time_min_mean_max_diffpostrajtime_min_mean_max_vel_nodescore_correlation_mygraph/val'
    # tracker_dir = 'out/gt_all_egocomp_margin0.6_width25_oraclenode_oracleedge_4096_8000_mean_dist_over_time_min_mean_max_diffpostrajtime_min_mean_max_vel_nodescore_correlation_mygraph/val'
    # tracker_dir = '4495651/out/gt_all_egocomp_margin0.6_width25_oraclenode_oracleedge_4096_8000_mean_dist_over_time_min_mean_max_diffpostrajtime_min_mean_max_vel_nodescore_correlation_mygraph/val'
    # tracker_dir = '../../3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/out/all_egocomp_margin0.6_width25_oraclenode_oracleedge_4096_8000_mean_dist_over_time_min_mean_max_diffpostrajtime_min_mean_max_vel_nodescore_correlation_mygraph/val/'
    # tracker_dir = 'out/gt_all_egocomp_margin0.6_width25_oraclenode_oracleedge_4096_8000_mean_dist_over_time_min_mean_max_diffpostrajtime_min_mean_max_vel_nodescore_correlation_mygraph/val/'
    tracker_dir = 'results/4540531/out/all_egocomp_margin0.6_width25_nooracle_4096_32000_mean_dist_over_time_min_mean_max_diffpostrajtime_min_mean_max_vel_nodescore_correlation_mygraph/val'
    tracker_dir = '/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/out/trajectories_removestatic1_numcd44_removegroundrc1_removefar1_removeheight1_len25_nooracle_4096_8000_mean_dist_over_time_min_mean_max_diffpostrajtime_min_mean_max_vel_nodescore_correlation_mygraph_16_5.0/val'
    tracker_dir = 'output_not_registered/val'
    # OLD FLOW
    tracker_dir = '4571635/out/all_egocomp_margin0.6_width25_nooracle_4096_32000_mean_dist_over_time_min_mean_max_diffpostrajtime_min_mean_max_vel_nodescore_correlation_mygraph_32_5.0/val'
    # NEW FLOW
    tracker_dir = 'out/trajectories_removestatic1_numcd44_removegroundrc0_remove_ground_pts_patch1_removefar1_removeheight1_len25_nooracle_32_32_32_32_0.7_3.5_1.5e-06_0.05_4096_8000_/val'
    
    tracker_dir = '/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/out/trajectories_removestatic1_numcd44_removegroundrc1_removefar1_removeheight1_len25_oraclenode_oracleedge_32_32_32_32_0.75_3.5_1.5e-06_0.05_4096_8000__NS_MG_16_5.0/val/'
    tracker_dir = '/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/out/trajectories_removestatic1_numcd44_removegroundrc1_removefar1_removeheight1_len25_nooracle_32_32_32_32_0.75_3.5_1.5e-06_0.05_4096_8000__NS_MG_16_5.0/val'
    tracker_dir = 'out/trajectories_removestatic1_numcd44_removegroundrc0_remove_ground_pts_patch1_removefar1_removeheight1_len25_nooracle_32_64_32_64_0.929_1.902_0.9221111_3.155_8.11e-08_0.0011_16000_16000__NS_MG_32_2.0_LN__AU_/val'
    tracker_dir = '../../4673658/out/all_egocomp_margin0.6_width25_nooracle_64_64_3.5900203994472646e-07_0.06176295901709523_16000_16000__NS_MG_32_2.0_LN_/val'
    tracker_dir = '4676686/out/all_egocomp_margin0.6_width25_nooracle_64_64_3.5900203994472646e-07_0.06176295901709523_16000_16000__NS_MG_32_2.0_LN_/val/'
    tracker_dir = 'collapsed/val'
    tracker_dir = 'out/trajectories2_nooracle_64_64_3.5900203994472646e-07_0.06176295901709523_16000_16000__NS_MG_32_2.0_LN_/val'
    # tracker_dir = 'collapsed_heuristic_mean/val'
    tracker_dir = '/workspace/result/out/all_egocomp_margin0.6_width25_nooracle_64_64_3.5900203994472646e-07_0.06176295901709523_16000_16000__NS_MG_32_2.0_LN_/val'
    tracker_dir = '/workspace/result/out/all_egocomp_margin0.6_width25_oraclenode_oracleedge_64_64_3.5900203994472646e-07_0.06176295901709523_16000_16000__NS_MG_32_2.0_LN_/val/'
    tracker_dir = '/workspace/result/out/detections/GNN_RealData_ValTrainOLD_all_egocomp_margin0.6_width25_oraclenode_64_64_3.5900203994472646e-07_0.06176295901709523_16000_16000__NS_MG_32_2.0_LN_/val/'
    tracker_dir = 'out/detections/val'
    tracker_dir = 'out/detections/GNN_GT_FLOW_FLOW_ONLY_EVAL/val/'

    print(tracker_dir)
    gt_folder = 'data/waymo_converted'
    gt_folder = '/dvlresearch/jenny/Waymo_Converted_GT'
    gt_folder = '/dvlresearch/jenny/debug_Waymo_Converted_val/Waymo_Converted'
    gt_folder = '/workspace/Waymo_Converted_val'
    gt_folder = '../../../../datasets/Argoverse2'
    seq_list = os.listdir(tracker_dir)
    min_points = -1
    max_points = 1000000
    # for m in [-1]: # [-1, 0, 5, 10, 15, 20, 25]:
    m = -1
    c = None
    for t in ['GNN_GT_FLOW_ORACLE_EDGE_FLOW_ONLY_EVAL']: # os.listdir('out/detections/'): #class_dict.keys():
        tracker_dir = f'out/detections/{t}/val'
        print(tracker_dir)
        # seq = '16473613811052081539'
        # seq_list = [seq]
        print(f'Class: {c}')
        min_points = m
        max_points = m+5 if m <= 25 and m != -1 else 1000000
        _, detection_metric = eval_detection(
            gt_folder=gt_folder,
            trackers_folder=tracker_dir,
            seq_to_eval=seq_list,
            remove_far=True,
            remove_non_drive=False,
            remove_non_move=True,
            remove_non_move_strategy='per_frame',
            remove_non_move_thresh=1.0,
            classes_to_eval='all',
            debug=False,
            visualize=False,
            name=name,
            min_points=min_points,
            max_points=max_points,
            base_dir='',
            print_detail=False,
            filter_class=c,
            eval_only_machted=False,
            filter_moving_before=False)
        
        print(detection_metric, '\n')
        print()

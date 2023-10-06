# FOR DETECTION EVALUATION
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from av2.evaluation.detection.eval import evaluate
from av2.evaluation.detection.utils import DetectionCfg
from av2.evaluation.detection.constants import CompetitionCategories, CompetitionCategoriesWaymo 
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
import matplotlib


_class_dict_argo = {
    -1: 'UNMATCHED',
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

class_dict_argo = {v: k for k, v in _class_dict_argo.items()}

WAYMO_CLASSES = {'TYPE_UNKNOWN': -1, 'TYPE_VECHICLE': 1,
                 'TYPE_PEDESTRIAN': 2, 'TYPE_SIGN': 12, 'TYPE_CYCLIST': 15}
_class_dict_waymo = {
        -1: 'TYPE_UNKNOWN', 
        1: 'TYPE_VECHICLE',
        2: 'TYPE_PEDESTRIAN', 
        3: 'TYPE_SIGN', 
        4: 'TYPE_CYCLIST'}

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
        gt_folder=None):
    
    if is_gt:
        # get file name
        split = os.path.basename(paths)
        file = 'filtered_version.feather'
        file = 'remove_non_drive_' + file if remove_non_drive else file
        file = 'remove_far_' + file if remove_far else file
        file = 'remove_non_move_' + file if remove_non_move else file
        file = remove_non_move_strategy + '_' + file if remove_non_move else file
        file = str(remove_non_move_thresh)[
            :5] + '_' + file if remove_non_move else file
        file = split + '_' + file
        
        if 'Waymo' in paths:
            path_filtered = os.path.join('/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/Waymo_Converted_filtered', file)
        else:
            path_filtered = os.path.join('/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/Argoverse2_filtered', file)
    
    if not is_gt or not os.path.isfile(path_filtered):
        df = None
        for i, path in enumerate(os.listdir(paths)):
            if seq_list is not None:
                if path not in seq_list and not is_gt:
                    continue
            data = feather.read_feather(
                os.path.join(paths, path, 'annotations.feather'))

            if 'Argo' in paths or not is_gt:
                def convert2int(x): return class_dict_argo[x]
                data['category'] = data['category'].apply(convert2int)
            else:
                def str2ing(x): return int(x)
                data['category'] = data['category'].apply(str2ing)

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

            # visualize detections
            fig, ax = plt.subplots()
            j = 0
            for i, row in time_ddf.iterrows():
                plt.scatter(ddf_ego[j, 0], ddf_ego[j, 1], color='black', marker='o', s=2)
                x_0 = ddf_ego[j, 0]-0.5*row['length_m']
                y_0 = ddf_ego[j, 1]-0.5*row['width_m']

                mat = quat_to_mat(np.array([row['qw'], row['qx'], row['qy'], row['qz']]))
                alpha = Rotation.from_matrix(mat).as_euler('xyz')[2]
                t = matplotlib.transforms.Affine2D().rotate_around(ddf_ego[j, 0], ddf_ego[j, 1], alpha) + ax.transData
                # alpha = row['rot']
                rect = patches.Rectangle(
                    (x_0, y_0),
                    row['length_m'],
                    row['width_m'],
                    linewidth=1,
                    edgecolor='black',
                    facecolor='none',
                    transform=t)

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
                # alpha = np.arccos(mat[0, 0])
                alpha = Rotation.from_matrix(mat).as_euler('xyz')[2]
                t = matplotlib.transforms.Affine2D().rotate_around(gdf_ego[j, 0], gdf_ego[j, 1], alpha) + ax.transData
                rect = patches.Rectangle(
                    (x_0, y_0),
                    row['length_m'],
                    row['width_m'],
                    linewidth=1,
                    edgecolor=color,
                    facecolor='none',
                    transform=t)

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
        visualize=False,
        debug=False,
        name='General',
        just_eval=False,
        min_points=-1,
        max_points=1000000,
        base_dir='../../../',
        print_detail=False,
        filter_class=-2,
        only_matched_gt=False,
        filter_moving_first=False,
        use_matched_category=False,
        filter_moving=True):

    if not len(seq_to_eval):
        return None, np.array([0, 2, 1, 3.142, 0]), None

    is_waymo = 'waymo' in gt_folder or 'Waymo' in gt_folder
    gt_folder = os.path.join(gt_folder, split)
    loader = AV2SensorDataLoader(data_dir=Path(
        gt_folder), labels_dir=Path(gt_folder))
    dataset_dir = Path(gt_folder)
    eval_only_roi_instances = False if is_waymo else True

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
        gt_folder=gt_folder)

    if just_eval:
        print("Loaded ground truth...")
    
    dts = get_feather_files(
        trackers_folder,
        seq_list=seq_to_eval,
        is_gt=False,
        loader=loader,
        gt_folder=gt_folder)
    dts = dts.drop_duplicates()
    # dts = dts[np.logical_and(dts['height_m'] > 0.1, 
    #                          np.logical_and(dts['length_m'] > 0.1, dts['width_m'] > 0.1))]

    # dts = dts[dts['num_interior_pts'] > 50]
    # dts = dts[dts['height_m'] < 3]
    # dts = dts[dts['width_m'] < 4]
    # dts = dts[dts['length_m'] < 7]
    if print_detail:
        print(f'\t Num dts: {dts.shape}, Num gts: {gts.shape}')

    if just_eval:
        print("Loaded detections...")

    if dts is None:
            return None, np.array([0, 2, 1, 3.142, 0]), None

    if use_matched_category:
        filter_class = -1
    
    if filter_class == -2:
        gts['category_int'] = [1] * gts.shape[0]    
        gts['category'] = [1] * gts.shape[0]
        filter_class = 1 # 'REGULAR_VEHICLE'
    else:                                                                                                                                                           
         gts['category_int'] = gts['category'] 
    
    if is_waymo:
        _class_dict = _class_dict_waymo
    else:
        _class_dict = _class_dict_argo

    gts['category'] = [_class_dict[c] for c in gts['category']]
    dts['category'] = [_class_dict[c] for c in dts['category']]
    if print_detail:
        print(f' \t Min points {min_points}, Max points {max_points}')

    if just_eval:
        print("Loaded ground truth...")

    if just_eval:
        print("Evaluate now...")

    gts_orig = gts
    dts_orig = dts
    for affinity, tp_thresh, threshs, n_jobs in zip(
        ['CENTER', 'IoU3D'], [2.0, 0.6], [(0.5, 1.0, 2.0, 4.0), (0.2, 0.4, 0.6, 0.8)], [8, 1]):

        # Evaluate instances.
        # Defaults to competition parameters.
        if print_detail:
            print(f' \t Setting categories to Waymo categories {is_waymo}')
        if is_waymo:
            categories : Tuple[str, ...] = tuple(x.value for x in CompetitionCategoriesWaymo)
        else:
            categories : Tuple[str, ...] = tuple(x.value for x in CompetitionCategories)
        competition_cfg = DetectionCfg(
            dataset_dir=dataset_dir, 
            eval_only_roi_instances=eval_only_roi_instances, 
            tp_threshold_m=tp_thresh,
            affinity_type=affinity,
            affinity_thresholds_m=threshs,
            categories=categories
            )

        print(f"\t {affinity}\n")
        dts, gts, metrics, np_tps, np_fns, _, all_results_df = evaluate(
            dts_orig,
            gts_orig,
            cfg=competition_cfg,
            min_points=min_points,
            max_points=max_points,
            filter_class=filter_class,
            only_matched_gt=only_matched_gt,
            filter_moving_first=filter_moving_first,
            use_matched_category=use_matched_category,
            _class_dict=_class_dict,
            n_jobs=n_jobs,
            filter_moving=filter_moving)
        
        if print_detail:
            print(f"\t Writing macthed detections to matched_{trackers_folder}/annotations_{affinity}.feather...")
        os.makedirs(f'matched_{trackers_folder}', exist_ok=True)
        feather.write_feather(dts, f'matched_{trackers_folder}/annotations_{affinity}.feather')
        
        dts = dts[dts['is_evaluated']==1]
        gts = gts[gts['is_evaluated']==1]
        if print_detail:
            print('\t Shapes after', dts.shape, gts.shape)

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
            print(f'Less than 5 points and more than 0: TP {np_tps[np.logical_and(np_tps < 5, np_tps >= 0)].shape[0]}, FN {np_fns[np.logical_and(np_fns < 5, np_fns >= 0)].shape[0]}')
            print(f'Less than 10 points and more than 5: TP {np_tps[np.logical_and(np_tps < 10, np_tps >= 5)].shape[0]}, FN {np_fns[np.logical_and(np_fns < 10, np_fns >= 5)].shape[0]}')
            print(f'Less than 15 points and more than 10: TP {np_tps[np.logical_and(np_tps < 15, np_tps >= 10)].shape[0]}, FN {np_fns[np.logical_and(np_fns < 15, np_fns >= 10)].shape[0]}')
            print(f'Less than 20 points and more than 15: TP {np_tps[np.logical_and(np_tps < 20, np_tps >= 15)].shape[0]}, FN {np_fns[np.logical_and(np_fns < 20, np_fns >= 15)].shape[0]}')
            print(f'Less than 25 points and more than 20: TP {np_tps[np.logical_and(np_tps < 25, np_tps >= 20)].shape[0]}, FN {np_fns[np.logical_and(np_fns < 25, np_fns >= 20)].shape[0]}')
            print(f'More than 25 points: TP {np_tps[np_tps >= 25].shape[0]}, FN {np_fns[np_fns >= 25].shape[0]}')
            
        # AP    ATE    ASE    AOE    CDS
        _filter_class = filter_class if filter_class == -1 else _class_dict[filter_class]
        if _filter_class == -1:
            metric = metrics.loc['AVERAGE_METRICS'].values
        else:
            print('\tDetection metrics: ', metrics.loc[_filter_class].values)
            metric = metrics.loc[_filter_class].values

        # break
  
    return metrics, metric, all_results_df


if __name__ == '__main__':
    name ='just_eval'
    gt_folder = '/dvlresearch/jenny/Waymo_Converted_GT'
    gt_folder = '/workspace/Waymo_Converted_train/Waymo_Converted'
    gt_folder = '/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/download_for_vis/Waymo_Converted'
    # gt_folder = '../../../../datasets/Argoverse2'

    min_points = -1
    max_points = 1000000
    # for m in [-1]: # [-1, 0, 5, 10, 15, 20, 25]:
    m = -1
    c = -2 # -1 = dont filter, -2 = set everything to 'REGULAR_VEHICLE', else set to class
    split = 'train'
    detections = 'detector'
    orig_split = 'train' if detections != 'evaluation' else 'val'
    for t in os.listdir(f'tracks/tracks_for_eval/initial_dets'): #class_dict.keys():
        print(t)
        tracker_dir = f'/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/m59iURZnSJmGWVJ5lmqE1Q/3DOpenWorldMOT/3DOpenWorldMOT/out/detections_val_gnn/INITIAL_DETS_OLD_MODEL_ORACLE_0.1_0.1_all_egocomp_margin0.6_width25_oraclenode_oracleedge_64_64_64_64_0.5_3.5_0.5_4_3.162277660168379e-06_0.0031622776601683794_16000_16000__NS_MG_32_2.0_LN___P___MMMDPTT___MMMV_/val_gnn'
        print(tracker_dir)
        # seq = '16473613811052081539'
        # seq_list = [seq]
        seq_list = os.listdir(tracker_dir)
        min_points = m
        max_points = m+5 if m <= 25 and m != -1 else 1000000
        _, detection_metric, _ = eval_detection(
            gt_folder=gt_folder,
            trackers_folder=tracker_dir,
            split=orig_split,
            seq_to_eval=seq_list,
            remove_far=True,
            remove_non_drive=False,
            remove_non_move=True,
            remove_non_move_strategy='per_frame',
            remove_non_move_thresh=1.0,
            debug=False,
            just_eval=True,
            visualize=True,
            name=name,
            min_points=min_points,
            max_points=max_points,
            base_dir='',
            print_detail=False,
            filter_class=c,
            only_matched_gt=False,
            filter_moving_first=False,
            use_matched_category=False,
            filter_moving=True)
        
        print(detection_metric, '\n')
        print()

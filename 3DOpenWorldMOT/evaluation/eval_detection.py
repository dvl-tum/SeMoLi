# FOR DETECTION EVALUATION
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

    df = None
    try:
        for i, path in enumerate(os.listdir(paths)):
            if seq_list is not None:
                if path not in seq_list and not is_gt:
                    continue
            data = feather.read_feather(
                os.path.join(paths, path, 'annotations.feather'))
            if 'argo' in paths or not is_gt:
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
    except:
        return None

    df = df.astype({'num_interior_pts': 'int64'})

    if not is_gt:
        if 'score' not in df.columns:
            # np.max(df['num_interior_pts'].values)/df['num_interior_pts'].values # [1] * df.shape[0]
            df['score'] = [1] * df.shape[0]

    # filter out objects in non-drivable area or far away
    if is_gt:
        # get file name
        split_dir = paths
        split = os.path.basename(paths)
        split_dir = os.path.dirname(paths)

        file = 'filtered_version.feather'
        file = 'remove_non_drive_' + file if remove_non_drive else file
        file = 'remove_far_' + file if remove_far else file
        file = 'remove_non_move_' + file if remove_non_move else file
        file = remove_non_move_strategy + '_' + file if remove_non_move else file
        file = str(remove_non_move_thresh)[
            :5] + '_' + file if remove_non_move else file
        file = split + '_' + file if remove_far else file

        path_filtered = \
            os.path.join(split_dir, file)

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

        # generate filtered version
        filtered = None
        map_dict = dict()
        # iterate over sequences
        num_seqs = df['log_id'].unique().shape[0]
        for m, seq in enumerate(df['log_id'].unique()):
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
            for i, t in enumerate(sorted(seq_df['timestamp_ns'].unique().tolist())):
                track_ids = list()
                time_df = seq_df[seq_df['timestamp_ns'] == t]

                # get labels
                labels = loader.get_labels_at_lidar_timestamp(
                    log_id=seq, lidar_timestamp_ns=int(t))

                # remove labels that are non in dirvable area
                if remove_non_drive and 'argo' in paths:
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

                        other_trans = list()
                        bool_labels = list()
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

                                if 'argo' in path:
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
                        labels = [l for i, l in enumerate(
                            labels) if bool_labels[i]]
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
                
                # get track id of remaining objects
                track_ids = [l.track_id for l in labels]

                # filter time df by track ids
                time_df = time_df[time_df['track_uuid'].isin(track_ids)]
                if filtered is None and time_df.shape[0]:
                    filtered = time_df
                elif filtered is not None:
                    filtered = filtered.append(time_df)

        # store filtered df
        df = filtered
        with open(path_filtered, 'wb') as f:
            feather.write_feather(df, f)

    return df


def visualize_whole(df, gf, name, base_dir='../../../'):
    split_dir = Path('/storage/user/seidensc/datasets/Waymo_Converted/val')
    loader = AV2SensorDataLoader(data_dir=split_dir, labels_dir=split_dir)
    for seq in df['log_id'].unique():
        print(f'storing to {base_dir}Visualization_Whole_DETS/{seq}')

        ddf = df[df['log_id'] == seq]
        gdf = gf[gf['log_id'] == seq]
        os.makedirs(f'{base_dir}Visualization_Whole_DETS/{seq}', exist_ok=True)
        
        lims_mins = list()
        lims_maxs = list()
        for i, timestamp in enumerate(sorted(ddf['timestamp_ns'].unique())):
            city_SE3_ego = loader.get_city_SE3_ego(seq, int(timestamp))
            time_gdf = gdf[gdf['timestamp_ns'] == timestamp]
            gdf_city = city_SE3_ego.transform_point_cloud(
                        time_gdf[['tx_m', 'ty_m', 'tz_m']].values)
            mins = np.min(gdf_city, axis=0)
            maxs = np.max(gdf_city, axis=0)

            lims_mins.append(mins)
            lims_maxs.append(maxs)
        lims_mins = np.vstack(lims_mins)
        lims_maxs = np.vstack(lims_maxs)

        mins = np.min(lims_mins, axis=0)
        maxs = np.max(lims_maxs, axis=0)

        lims = np.vstack([mins, maxs])

        x_lim = (lims[0, 0] - 10,
                 lims[1, 0] + 10)
        y_lim = (lims[0, 1] - 10,
                 lims[1, 1] + 10)

        for i, timestamp in enumerate(sorted(ddf['timestamp_ns'].unique())):
            city_SE3_ego = loader.get_city_SE3_ego(seq, int(timestamp))
            
            time_ddf = ddf[ddf['timestamp_ns'] == timestamp]
            time_gdf = gdf[gdf['timestamp_ns'] == timestamp]
            ddf_city = city_SE3_ego.transform_point_cloud(
                        time_ddf[['tx_m', 'ty_m', 'tz_m']].values)
            gdf_city = city_SE3_ego.transform_point_cloud(
                        time_gdf[['tx_m', 'ty_m', 'tz_m']].values)

            fig, ax = plt.subplots()
            j = 0
            for i, row in time_ddf.iterrows():
                plt.scatter(ddf_city[j, 0], ddf_city[j, 1], color='black', marker='o', s=30)
                x_0 = ddf_city[j, 0]-0.5*row['length_m']
                y_0 = ddf_city[j, 1]-0.5*row['width_m']
                rect = patches.Rectangle(
                    (x_0, y_0),
                    row['length_m'],
                    row['width_m'],
                    linewidth=1,
                    edgecolor='black',
                    facecolor='none')
                ax.add_patch(rect)
                j += 1
            
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

                plt.scatter(gdf_city[j, 0], gdf_city[j, 1],
                            color=color, marker='*', s=5)
                x_0 = gdf_city[j, 0]-0.5*row['length_m']
                y_0 = gdf_city[j, 1]-0.5*row['width_m']
                rect = patches.Rectangle(
                    (x_0, y_0),
                    row['length_m'],
                    row['width_m'],
                    linewidth=1,
                    edgecolor=color,
                    facecolor='none')
                ax.add_patch(rect)
                j += 1

            plt.axis('off')
            plt.xlim(x_lim)
            plt.ylim(y_lim)
            plt.savefig(
                f'{base_dir}Visualization_Whole_DETS/{seq}/frame_{timestamp}_{name}.jpg')
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
        min_points=0,
        max_points=1000000,
        base_dir='../../../'):

    gt_folder = os.path.join(gt_folder, split)
    loader = AV2SensorDataLoader(data_dir=Path(
        gt_folder), labels_dir=Path(gt_folder))
    dataset_dir = Path(gt_folder)
    eval_only_roi_instances = False if 'waymo' in gt_folder else True
    # Defaults to competition parameters.
    competition_cfg = DetectionCfg(
        dataset_dir=dataset_dir, eval_only_roi_instances=eval_only_roi_instances)
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
    if just_eval:
        print("Loaded detections...")

    if dts is None:
        return None, np.array([0, 2, 1, 3.142, 0])

    gts['category'] = [_class_dict[c] for c in gts['category']]
    dts['category'] = [_class_dict[c] for c in dts['category']]
    gts['category'] = ['REGULAR_VEHICLE'] * gts.shape[0]

    # remove gt objects without lidar points inside
    print('All GT objects: ', gts.shape[0])
    print('GT objects with 0 points: ', gts[gts['num_interior_pts'] == 0].shape[0])
    print('GT objects with less than 5 points and more than 0: ', gts[np.logical_and(gts['num_interior_pts'] < 5, gts['num_interior_pts'] >= 0)].shape[0])
    print('GT objects with less than 10 points and more than 5: ', gts[np.logical_and(gts['num_interior_pts'] < 10, gts['num_interior_pts'] >= 5)].shape[0])
    print('GT objects with less than 15 points and more than 10: ', gts[np.logical_and(gts['num_interior_pts'] < 15, gts['num_interior_pts'] >= 10)].shape[0])
    print('GT objects with less than 20 points and more than 15: ', gts[np.logical_and(gts['num_interior_pts'] < 20, gts['num_interior_pts'] >= 15)].shape[0])
    print('GT objects with less than 25 points and more than 20: ', gts[np.logical_and(gts['num_interior_pts'] < 25, gts['num_interior_pts'] >= 20)].shape[0])
    print('GT objects with more than 25 points: ', gts[gts['num_interior_pts'] >= 25].shape[0])

    gts = gts[gts['num_interior_pts'] > min_points]

    if (visualize or debug) and dts.shape[0] and gts.shape[0]:
        visualize_whole(dts, gts, name, base_dir)

    '''dts = gts
    dts['score'] = [1] * dts.shape[0]
    print(gts.shape)'''

    if just_eval:
        print("Loaded ground truth...")

    if just_eval:
        print("Evaluate now...")

    # Evaluate instances.
    dts, gts, metrics, num_points_tps, num_points_fns = evaluate(dts, gts, cfg=competition_cfg)

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
        metric = metrics.loc[classes_to_eval].values

    return metrics, metric


if __name__ == '__main__':
    name = 'gt_all_egocomp_margin0.6_width25_oraclenode_oracleedge_4096_8000_mean_dist_over_time_min_mean_max_diffpostrajtime_min_mean_max_vel_nodescore_correlation_mygraph'
    name = 'gt_all_egocomp_margin0.6_width25_oraclenode_oracleedge_4096_8000_pos_min_mean_max_diffpostrajtime_min_mean_max_vel_nodescore_correlation_torchgraph'
    name = 'gt_all_egocomp_margin0.6_width25_oraclenode_oracleedge_4096_8000_pos_min_mean_max_diffpostrajtime_min_mean_max_vel_nodescore_correlation_torchgraph'
    name = 'gt_all_egocomp_margin0.6_width25_oraclenode_oracleedge_4096_20000_pos_min_mean_max_diffpostrajtime_min_mean_max_vel_nodescore_correlation_torchgraph'
    tracker_dir = f'out/{name}/val'
    gt_folder = 'data/waymo_converted'
    seq_list = os.listdir(tracker_dir)
    seq_list = ['16473613811052081539']
    min_points = 0
    max_points = 100000
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
        debug=True,
        visualize=False,
        name=name,
        min_points=min_points,
        max_points=max_points,
        base_dir='')
    
    print(detection_metric)
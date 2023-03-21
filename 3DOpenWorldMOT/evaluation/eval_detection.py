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
            if seq != '16473613811052081539':
                continue
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

                if t == 1543280286123574:
                    print(seq_df[seq_df['timestamp_ns'] == t])
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

                # remove points that are far away (>80,)
                if remove_far:
                    all_centroids = np.asarray(
                        [label.dst_SE3_object.translation for label in labels])
                    dists_to_center = np.sqrt(np.sum(all_centroids ** 2, 1))
                    ind = np.where(dists_to_center <= 80)[0]
                    labels = [labels[i] for i in ind]
                if remove_non_move:
                    if remove_non_move_strategy == 'per_seq':
                        labels = [
                            label for label in labels if label.track_id in movers]
                    elif remove_non_move_strategy == 'per_frame':

                        if i < len(timestamps) - 1:
                            # labels at t+1
                            _labels_t2 = loader.get_labels_at_lidar_timestamp(
                                log_id=seq, lidar_timestamp_ns=int(timestamps[i+1]))
                            city_SE3_t2 = loader.get_city_SE3_ego(
                                seq, int(timestamps[i+1]))
                            labels_t2 = _labels_t2.transform(city_SE3_t2)
                            ids_t2 = [label.track_id for label in labels_t2]
                        else:
                            labels_t2 = list()
                            ids_t2 = list()

                        if i > 0:
                            labels_t0 = loader.get_labels_at_lidar_timestamp(
                                log_id=seq, lidar_timestamp_ns=int(timestamps[i-1]))
                            city_SE3_t0 = loader.get_city_SE3_ego(
                                seq, int(timestamps[i-1]))
                            labels_t0 = labels_t0.transform(city_SE3_t0)
                            ids_t0 = [label.track_id for label in labels_t0]
                        else:
                            labels_t0 = list()
                            ids_t0 = list()

                        city_SE3_t1 = loader.get_city_SE3_ego(
                            seq, int(timestamps[i]))
                        labels_city = labels.transform(city_SE3_t1)

                        other_trans = list()
                        bool_labels = list()
                        vels = list()
                        for m, label in enumerate(labels_city):
                            if len(labels_t2) and label.track_id in ids_t2:
                                other_trans = labels_t2[ids_t2.index(
                                    label.track_id)].dst_SE3_object.translation
                                dist = np.linalg.norm(
                                    label.dst_SE3_object.translation[:-1] - other_trans[:-1])
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
                                if t == 1543280286123574:
                                    print(label.dst_SE3_object.translation, other_trans, _labels_t2[ids_t2.index(
                                    label.track_id)].dst_SE3_object.translation, dist, diff_time, vel)

                            elif len(labels_t0) and label.track_id in ids_t0:
                                other_trans = labels_t0[ids_t0.index(
                                    label.track_id)].dst_SE3_object.translation
                                dist = np.linalg.norm(
                                    label.dst_SE3_object.translation[:-1] - other_trans[:-1])
                                if 'argo' in path:
                                    diff_time = (
                                        t-timestamps[i-1]) / np.power(10, 9)
                                else:
                                    diff_time = (
                                        t-timestamps[i-1]) / np.power(10, 6)
                                vel = dist/diff_time
                                vels.append(vel)
                                bool_labels.append(
                                    vel > remove_non_move_thresh)
                                if t == 1543280286123574:
                                    print(label.dst_SE3_object.translation, other_trans, dist, diff_time, vel)
                            else:
                                vels.append(None)
                                bool_labels.append(False)
                        # filter labels
                        labels = [l for i, l in enumerate(
                            labels) if bool_labels[i]]
                        if t == 1543280286123574:
                            print(labels)
                    else:
                        assert remove_non_move_strategy in [
                            'per_frame', 'per_seq'], 'remove strategy for static objects not defined'

                # get track id of remaining objects
                if t == 1543280286123574:
                    track_ids = [l.track_id for l in labels]
                    print(track_ids)
                    quit()

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


def visualize_whole(df, gf, name):
    for seq in df['log_id'].unique():
        print(f'storing to ../../../Visualization_Whole_DETS/{seq}')

        ddf = df[df['log_id'] == seq]
        gdf = gf[gf['log_id'] == seq]
        os.makedirs(f'../../../Visualization_Whole_DETS/{seq}', exist_ok=True)
        x_lim = (np.min(gdf['tx_m'].values) - 10,
                 np.max(gdf['tx_m'].values) + 10)
        y_lim = (np.min(gdf['ty_m'].values) - 10,
                 np.max(gdf['ty_m'].values) + 10)

        for i, timestamp in enumerate(sorted(ddf['timestamp_ns'].unique())):
            time_ddf = ddf[ddf['timestamp_ns'] == timestamp]
            time_gdf = gdf[gdf['timestamp_ns'] == timestamp]

            fig, ax = plt.subplots()
            for i, row in time_ddf.iterrows():
                plt.scatter(row['tx_m'], row['ty_m'], color='red', marker='o')
                x_0 = row['tx_m']-0.5*row['length_m']
                y_0 = row['ty_m']-0.5*row['width_m']
                rect = patches.Rectangle(
                    (x_0, y_0),
                    row['length_m'],
                    row['width_m'],
                    linewidth=1,
                    edgecolor='red',
                    facecolor='none')
                ax.add_patch(rect)

            for i, row in time_gdf.iterrows():
                if row['category'] == 'PEDESTRIAN':
                    marker = '*'
                else:
                    marker = 'o'
                plt.scatter(row['tx_m'], row['ty_m'],
                            color='black', marker=marker, s=10)
                x_0 = row['tx_m']-0.5*row['length_m']
                y_0 = row['ty_m']-0.5*row['width_m']
                rect = patches.Rectangle(
                    (x_0, y_0),
                    row['length_m'],
                    row['width_m'],
                    linewidth=1,
                    edgecolor='black',
                    facecolor='none')
                ax.add_patch(rect)

            # plt.axis('off')
            plt.xlim(x_lim)
            plt.ylim(y_lim)
            plt.savefig(
                f'../../../Visualization_Whole_DETS/{seq}/frame_{timestamp}_{name}.jpg')
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
        just_eval=False):

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
    print(seq_to_eval)

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
    print(gts[gts['num_interior_pts'] == 0].shape)
    gts = gts[gts['num_interior_pts'] > 0]
    print(gts[gts['num_interior_pts'] < 5].shape)
    print(gts[gts['num_interior_pts'] < 10].shape)
    print(gts[gts['num_interior_pts'] < 15].shape)
    print(gts[gts['num_interior_pts'] < 20].shape)

    if (visualize or debug) and dts.shape[0] and gts.shape[0]:
        visualize_whole(dts, gts, name)

    if just_eval:
        print("Loaded ground truth...")

    if just_eval:
        print("Evaluate now...")

    # Evaluate instances.
    dts, gts, metrics = evaluate(dts, gts, cfg=competition_cfg)

    # AP    ATE    ASE    AOE    CDS
    classes_to_eval = 'REGULAR_VEHICLE'
    if classes_to_eval == 'all':
        metric = metrics.loc['AVERAGE_METRICS'].values
    else:
        metric = metrics.loc[classes_to_eval].values

    return metrics, metric

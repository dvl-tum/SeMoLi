from __future__ import print_function
import glob
import os
import argparse
from waymo_open_dataset.protos.metrics_pb2 import Objects, Object
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import dataset_pb2 as open_dataset
import tensorflow as tf
from collections import defaultdict
import pickle


'''
WAYMO CLASSES:
1 = VEHICLE
2 = PEDESTRIAN
3 = SIGN
4 = CYCLIST
'''


def main(split='training', save_dir='out', detection_file=None, only_stats=False, all_car=False):
    if detection_file is not None:
        with open(f'data_utils/new_seq_splits_Waymo_Converted_fixed_val/{detection_file}.txt', 'r') as f:
            seqs = f.read()
            seqs = seqs.split('\n')
    tf_record_dir = f'/workspace/waymo/waymo_format/{split}/'
    meta_data_save = defaultdict(dict)
    objects = metrics_pb2.Objects()
    object_set = set()
    tfrecords = sorted(glob.glob(f'{tf_record_dir}/*.tfrecord'))
    for i, record in enumerate(tfrecords):
        if detection_file is not None and os.path.basename(record).split('-')[1].split('_')[0] not in seqs:
            continue
        objects, meta_data_save, object_set = extract_frame(record, meta_data_save, objects, only_stats, all_car, object_set=object_set)
        print(f"Tfrecors {i}/{len(tfrecords)}...")
        
    if detection_file is not None:
        split = detection_file
    
    with open(f'{save_dir}/meta/meta_data_{split}.pickle', 'wb') as handle:
        pickle.dump(meta_data_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if only_stats:
        print("stored to {save_dir}/meta/meta_data_{split}.pickle...")
        return
	
    if all_car:
        all_car_add = '_car'
    else:
        all_car_add = ''

    with open(f'{save_dir}/gt/gt_{split}{all_car_add}.bin', 'wb') as f:
        f.write(objects.SerializeToString())
        
    print(f"Stored to {save_dir}/meta/meta_data_{split}.pickle and {save_dir}/gt/gt_{split}{all_car_add}.bin...")
    print(f'Overall objects from the following classes in the set {object_set}....')


def extract_frame(frames_path, meta_data_save=dict(), objects=None, only_stats=False, all_car=False, object_set=set()):
    dataset = tf.data.TFRecordDataset(frames_path, compression_type='')
    for fidx, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frame_dict = dict()
        frame_dict['location'] = frame.context.stats.location
        frame_dict['time_of_day'] = frame.context.stats.time_of_day
        frame_dict['weather'] = frame.context.stats.weather
        frame_dict['laser_object_counts'] = frame.context.stats.weather
        meta_data_save[frame.context.name][frame.timestamp_micros] = frame_dict
        
        if only_stats:
            return objects, meta_data_save, object_set
        
        time = frame.timestamp_micros
        labels = frame.laser_labels
        if len(labels) == 0:
            break
        objects, object_set = extract_labels(labels, frame.context.name, time, objects, all_car, object_set=object_set)
    return objects, meta_data_save, object_set


def extract_labels(box_labels, context_name, time, objects, all_car=False, object_set=set()):
    for obj in box_labels:
        print(obj)
        quit()
        object = metrics_pb2.Object(
             	object=obj,
                score=0.5,
                context_name=context_name,
                frame_timestamp_micros=time)
        if all_car and object.object.type == 3:
            continue
        object_set.add(object.object.type)
        if all_car:
            object.object.type = 1
        objects.objects.append(object)
    return objects, object_set


if __name__ == "__main__":
    save_dir = '/workspace/ExchangeWorkspace/waymo_gt_and_meta'
    all_car = True
    os.makedirs(f'{save_dir}/gt', exist_ok=True)
    os.makedirs(f'{save_dir}/meta', exist_ok=True)
    for detection_set in ['0.1_train_detector', '0.5_train_detector', '0.9_train_detector']: # '0.1_val_gnn', '0.1_val_detector', 'all_training', 'all_validation']:
        if 'all' in detection_set:
            detection_file = None
            if 'training' in detection_set:
                split = 'training'
            else:
                split = 'validation'
        else:
            split = 'training'
            detection_file = detection_set
            
        main(split, save_dir, detection_file, all_car=all_car)

'''
object {
  box {
    center_x: 19.480070895238896
    center_y: -50.34639710054398
    center_z: 2.299644261646847
    width: 2.7734957353748575
    length: 6.179721445654854
    height: 2.6500000000000057
    heading: 3.1052620840875065
  }
  metadata {
    speed_x: -5.067136851525712e-28
    speed_y: -5.067136851525712e-28
    accel_x: 5.630151460658475e-28
    accel_y: 5.630151460658475e-28
  }
  type: TYPE_VEHICLE
  id: "3eaSUD7ej2F7gc9Bisx5VQ"
  num_lidar_points_in_box: 146
}
score: 0.5
context_name: "11048712972908676520_545_000_565_000"
frame_timestamp_micros: 1522684691238230
'''


#!/usr/bin/env bash

export TARGET_DIR="$BASE_DIR/data/AV2/val"
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/val/*/sensors/lidar/*" $TARGET_DIR
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/val/*/annotations.feather" $TARGET_DIR
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/val/*/city_SE3_egovehicle.feather" $TARGET_DIR

export TARGET_DIR="$BASE_DIR/data/AV2/train"
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/train/*/sensors/lidar/*" $TARGET_DIR
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/train/*/annotations.feather" $TARGET_DIR
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/train/*/city_SE3_egovehicle.feather" $TARGET_DIR

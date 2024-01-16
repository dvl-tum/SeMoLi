import distinctipy
from pyarrow import feather
import open3d as o3d
from scipy.spatial.transform import Rotation
import numpy as np
import torch
import os
from av2.utils.typing import NDArrayBool, NDArrayByte, NDArrayFloat, NDArrayInt, NDArrayObject
from av2.geometry.se3 import SE3
from av2.structures.cuboid import Cuboid


def box_center_to_corner(translation, size, rotation_matrix):
    # To return
    l, w, h = size[0], size[1], size[2]

    # Create a bounding box outline
    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])

    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(
        rotation_matrix, bounding_box) + eight_points.transpose()

    return corner_box.transpose()


def show_flows(pc1, pc2, bounding_boxes_filtered, bounding_boxes_orig_gt, bounding_boxes_dets, pc_list, pc_colors):
    bbs = list()
    for color, bounding_boxes in zip([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [bounding_boxes_filtered, bounding_boxes_orig_gt, bounding_boxes_dets]):
        if bounding_boxes is None:
            continue

        for _, bb in bounding_boxes.iterrows():
            center = bb[['tx_m', 'ty_m', 'tz_m']].values
            extent = bb[['length_m', 'width_m', 'height_m']].values
            quat_xyzw = bb[['qw', 'qx', 'qy', 'qz']].values[..., [1, 2, 3, 0]]
            rotation = Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float64)

            corner_box = box_center_to_corner(center, extent, rotation)
            lines = [[0, 1], [1, 2], [2, 3], [0, 3],
            [4, 5], [5, 6], [6, 7], [4, 7],
            [0, 4], [1, 5], [2, 6], [3, 7]]

            # Use the same color for all lines
            c = [color for _ in range(len(lines))]

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corner_box)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(c)

            bbs.append(line_set)
    
    if pc1 is not None:
        pc_colors[0] = (0.46711832347767107, 0.4136978969378489, 0.6572061114161131)
        pc_full = o3d.geometry.PointCloud()
        pc_full.points = o3d.utility.Vector3dVector(pc1)
        pc_full.paint_uniform_color(pc_colors[0])
        pc_full.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pc_full.orient_normals_to_align_with_direction()

    if pc2 is not None:
        pc_colors[1] = (0.5272216993298807, 0.9720614882910129, 0.412923935906695)
        pc_filtered = o3d.geometry.PointCloud()
        pc_filtered.points = o3d.utility.Vector3dVector(pc2)
        pc_filtered.paint_uniform_color(pc_colors[1])
        pc_filtered.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pc_filtered.orient_normals_to_align_with_direction()    

    o3d_pc_list = list()
    for i, pc in enumerate(pc_list):
        seg_pc = o3d.geometry.PointCloud()
        seg_pc.points = o3d.utility.Vector3dVector(pc)
        seg_pc.paint_uniform_color(pc_colors[30+i])
        seg_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        seg_pc.orient_normals_to_align_with_direction()
        o3d_pc_list.append(seg_pc)

    if pc1 is not None and pc2 is not None:
        print('HERE??')
        o3d.visualization.draw_geometries([pc_full] + [pc_filtered] + o3d_pc_list + bbs)
    elif pc1 is None:
        o3d.visualization.draw_geometries([pc_filtered] + o3d_pc_list + bbs)
    else:
        o3d.visualization.draw_geometries([pc_full] + o3d_pc_list + bbs)


def get_interior(rotation, translation, lwh, pc):
    ego_SE3_object = SE3(rotation=rotation, translation=translation)
    cuboid = Cuboid(
            dst_SE3_object=ego_SE3_object,
            length_m=lwh[0],
            width_m=lwh[1],
            height_m=lwh[2],
            category='REGULAR_VEHILE',
            timestamp_ns=0,
        )
    _, mask = cuboid.compute_interior_points(pc)

    return pc[mask]


def filter_bbs(bounding_boxes_filtered, time):
    bounding_boxes_filtered = bounding_boxes_filtered[
        bounding_boxes_filtered['timestamp_ns'] == time]
    bounding_boxes_filtered = bounding_boxes_filtered[
        np.logical_and(bounding_boxes_filtered['tx_m'] < 50, bounding_boxes_filtered['ty_m'] < 20)]
    bounding_boxes_filtered = bounding_boxes_filtered[
        np.logical_and(bounding_boxes_filtered['tx_m'] > -50, bounding_boxes_filtered['ty_m'] > -20)]
    return bounding_boxes_filtered

def filter_pc(pc_filtered):
    mask = np.logical_and(
                np.logical_and(
                    np.logical_and(
                        pc_filtered[:, 0] > -50,
                        pc_filtered[:, 1] > -20
                    ),
                    pc_filtered[:, 0] < 50
                ),
                pc_filtered[:, 1] < 20
            )
    return pc_filtered[mask]



bb_path_waymo = 'timstamps_to_vis/matched/dt_matched_DBSCAN.feather'
bb_path_gt = 'timstamps_to_vis/matched/updated_gt_matched_DBSCAN.feather'
bb_path_gt = 'Argoverse2_filtered/train_1.0_per_frame_remove_non_move_remove_far_filtered_version.feather'
bb_path_ours = 'timstamps_to_vis/matched/dt_matched_ours.feather'
pc_path_full = 'timstamps_to_vis/orig'
pc_path_filtered = 'timstamps_to_vis/filtered'

input_colors = [(0.46711832347767107, 0.4136978969378489, 0.6572061114161131), (0.5272216993298807, 0.9720614882910129, 0.412923935906695)]

colors = distinctipy.get_colors(100, input_colors, pastel_factor=0.)

bounding_boxes_waymo = feather.read_feather(bb_path_waymo)
bounding_boxes_ours = feather.read_feather(bb_path_ours)
bounding_boxes_gt = feather.read_feather(bb_path_gt)

filtered = True
full = True
bb_waymo = False
bb_ours = True
bb_gt = True

# 315968457960513000, 315966781060062000

already_checked = ['315966777759602000.feather','315969827059614000.feather','315966777959995000.feather','315966778159712000.feather', '315966787359753000.feather','315968455160358000.feather','315969824259834000.feather','315966781060062000.feather','315966782559678000.feather','315966785860150000.feather','315966786359792000.feather','315968456759500000.feather','315968458359952000.feather','315968460560256000.feather','315969826159553000.feather','315969834960121000.feather','315969876760072000.feather','315969831859768000.feather','315968422159739000.feather',]
to_vis = ['315966781060062000.feather', '315966777759602000.feather','315969827059614000.feather','315966777959995000.feather','315966778159712000.feather', '315966787359753000.feather','315968455160358000.feather','315969824259834000.feather','315966781060062000.feather','315966782559678000.feather','315966785860150000.feather','315966786359792000.feather','315968456759500000.feather','315968458359952000.feather','315968460560256000.feather','315969826159553000.feather','315969834960121000.feather','315969876760072000.feather','315969831859768000.feather','315968422159739000.feather','315968457960513000.feather','315969824360030000.feather','315969826259749000.feather','315969871660057000.feather','315969874060097000.feather','315976473459770000.feather','315966898459473000.feather']

to_vis = ['315968457960513000.feather'] * 20

# 12071817-ba53-35a4-bf6c-a8e8e7ad8969, ca4144fb-10e5-3895-836f-87001f59ac65, e033cc8e-b23d-3fc6-8954-d90c5e98550e, c654b457-11d4-393c-a638-188855c8f2e5, ad319b98-6faa-3648-98bd-43afdbd20020, 595ec33e-a1aa-3aaf-8821-8d1780db354c, 1b8fc962-7036-4d7f-885e-40b631cbdeaf
for j, p in enumerate(os.listdir(pc_path_full)):
    # p = '41b6f7d7-e431-3992-b783-74b9edf42215'
    # if p == '5481321f-d317-3e80-8061-6e9c635c4ca9' or p == '595acd37-183c-489f-bb8a-c299a86b74c0' or p =='41b6f7d7-e431-3992-b783-74b9edf42215' or p == '87ce1d90-ca77-363b-a885-ec0ef6783847' or p == 'b51561d9-08b0-3599-bc78-016f1441bb91' or p == '12071817-ba53-35a4-bf6c-a8e8e7ad8969'\
    #     or p == 'b98a7838-ac1f-339f-93c5-fe7f98ea8657' or p == 'ca4144fb-10e5-3895-836f-87001f59ac65' or p == 'bffb0c9e-5e3a-3251-ab5e-299491b53cbf' or p == 'e033cc8e-b23d-3fc6-8954-d90c5e98550e' \
    #         or p =='c654b457-11d4-393c-a638-188855c8f2e5' or p == 'ad319b98-6faa-3648-98bd-43afdbd20020' or p == '595ec33e-a1aa-3aaf-8821-8d1780db354c' or p == '1b8fc962-7036-4d7f-885e-40b631cbdeaf':
    #     continue
    print(p)
    if p == '.DS_Store':
        continue
    for count, i in enumerate(sorted(os.listdir(os.path.join(pc_path_full, p)))):
        # i = to_vis[count]
        colors = distinctipy.get_colors(100, input_colors, pastel_factor=0.)

        # if i not in to_vis:
        #     continue
        waymo = filter_bbs(bounding_boxes_waymo, int(i[:-8]))
        ours = filter_bbs(bounding_boxes_ours, int(i[:-8]))
        gt_fail = filter_bbs(bounding_boxes_gt, int(i[:-8]))
        gt_fail = gt_fail[gt_fail['category'] != 1]
        gt_fail = gt_fail[gt_fail['category'] != 2]
        print(gt_fail['category'])
        if gt_fail.shape[0] == 0:
            continue
        print(i, waymo.shape, ours.shape, gt_fail.shape, j)

        pc_filtered = torch.load(f'{pc_path_filtered}/{p}/{i[:-8]}.pt', map_location=torch.device('cpu'))
        pc_filtered = pc_filtered['pc_list'].numpy()
        pc_filtered = filter_pc(pc_filtered)
        

        if full:
            pc_full = feather.read_feather(f'{pc_path_full}/{p}/{i}')
            pc_full = pc_full[['x', 'y', 'z']].values
            pc_full = filter_pc(pc_full)
        else:
            pc_full = None

        pc_list = list()
        for idx, row in ours.iterrows():
            alpha = float(row['rot'])
            rotation = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                                [np.sin(alpha), np.cos(alpha), 0],
                                [0, 0, 1]])
            pc = get_interior(rotation, row[['tx_m', 'ty_m', 'tz_m']].values, row[['length_m', 'width_m', 'height_m']].values, pc_filtered)
            pc_list.append(pc)
        
        if not bb_waymo:
            waymo = None
        if not bb_ours:
            ours = None
        if not bb_gt:
            gt_fail = None
        else:
            print(gt_fail.shape)
        if not filtered:
            pc_filtered = None

        show_flows(
            pc_full,
            pc_filtered,
            waymo, 
            ours,
            gt_fail,
            pc_list,
            colors)
        break
        
# 315966783959760000.feather (48, 29) (8, 29) (1, 27) 0
# 315966785860150000.feather (38, 29) (10, 29) (2, 27) 0
# 315966787459950000.feather (42, 29) (10, 29) (1, 27) 0
# 315966787659679000.feather (37, 29) (8, 29) (1, 27) 0
# 315968457960513000.feather (19, 29) (11, 29) (1, 27) 4
# 315968458260419000.feather (24, 29) (8, 29) (2, 27) 4
# 315968458359952000.feather (22, 29) (10, 29) (1, 27) 4
# 315968458760074000.feather (12, 29) (7, 29) (2, 27) 4
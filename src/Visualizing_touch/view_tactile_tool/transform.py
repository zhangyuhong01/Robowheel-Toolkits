# -*- coding: utf-8 -*-
"""触觉可视化工具：坐标系变换与选定帧数据加载。"""
import os
import json
import numpy as np
from read_tactile import DataLoaderV3

from . import config
from . import io_utils
from . import mano_export


def transform_tactile_to_object_frame(tactile_positions, tactile_forces, object_pose):
    """
    将触觉点和力向量从世界坐标系变换到物体坐标系。
    Returns: (object_positions, object_forces)
    """
    object_pose_inv = np.linalg.inv(object_pose)
    object_rotation_inv = object_pose_inv[:3, :3]
    tactile_positions_homo = np.hstack([tactile_positions, np.ones((tactile_positions.shape[0], 1))])
    object_positions_homo = (object_pose_inv @ tactile_positions_homo.T).T
    object_positions = object_positions_homo[:, :3]
    object_forces = (object_rotation_inv @ tactile_forces.T).T
    return object_positions, object_forces


def get_which_hand_from_selected_frames():
    """从 OUTPUT_DIR/selected_frames.json 读取 select_frames 时选择的左右手。Returns: 'left' | 'right' | None"""
    try:
        with open(io_utils.out_path("selected_frames.json"), "r", encoding="utf-8") as f:
            info = json.load(f)
        return info.get("which_hand")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def load_selected_frames_data(hdf5_path, which_hand="left"):
    """加载之前选定的帧数据。which_hand 若未传或为 None，会优先从 selected_frames.json 中读取。"""
    try:
        with open(io_utils.out_path("selected_frames.json"), "r", encoding="utf-8") as f:
            frame_info = json.load(f)
        selected_frames = frame_info["selected_frames"]
        which_hand = frame_info.get("which_hand", which_hand)
        print(f"Loading data for selected frames: {selected_frames}")
    except FileNotFoundError:
        print("No selected frames found. Please run frame selection first.")
        return None
    loader = DataLoaderV3(hdf5_path, which_hand=which_hand)
    frame_data = {}
    for frame_idx in selected_frames:
        print(f"Loading frame {frame_idx}...")
        tactile_dict = loader.get_tactile_data_dict([frame_idx])
        hand_pose = loader.get_wrist_tf_rel_world([frame_idx])
        if hand_pose is not None:
            hand_pose = hand_pose[0]
        else:
            hand_pose = np.eye(4)
            print("No hand pose data, using identity matrix")
        try:
            obj_poses = loader.get_objs_tf_rel_world([frame_idx])
            if obj_poses:
                obj_id = list(obj_poses.keys())[0]
                object_pose = obj_poses[obj_id][0]
                print(f"Loaded object pose for object {obj_id}")
            else:
                object_pose = np.eye(4)
                print("No object pose data, using identity matrix")
        except Exception as e:
            object_pose = np.eye(4)
            print(f"Error loading object pose: {e}, using identity matrix")
        frame_data[frame_idx] = {
            "tactile_data": tactile_dict,
            "hand_pose": hand_pose,
            "object_pose": object_pose,
        }
    loader.close()
    return frame_data


def get_selected_frame_indices():
    """从 OUTPUT_DIR/selected_frames.json 读取选中的帧号列表。Returns: list 或 None"""
    try:
        with open(io_utils.out_path("selected_frames.json"), "r", encoding="utf-8") as f:
            info = json.load(f)
        return info.get("selected_frames")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def get_selected_frames_quality_scores():
    """从 OUTPUT_DIR/selected_frames.json 读取与 selected_frames 同序的评分列表。Returns: list 或 None"""
    try:
        with open(io_utils.out_path("selected_frames.json"), "r", encoding="utf-8") as f:
            info = json.load(f)
        return info.get("quality_scores")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def get_first_available_gaussian_file(frame_idx=None):
    """获取用于可视化的 gaussian 压力文件。Returns: (文件路径, frame_idx) 或 (None, None)。"""
    if frame_idx is not None:
        path = io_utils.out_path(f"gaussian_pressure_frame_{frame_idx}.json")
        if os.path.exists(path):
            return path, frame_idx
        return None, None
    indices = get_selected_frame_indices()
    if not indices:
        return None, None
    for idx in indices:
        path = io_utils.out_path(f"gaussian_pressure_frame_{idx}.json")
        if os.path.exists(path):
            return path, idx
    return None, None


def test_coordinate_transformation(
    hdf5_path=None, which_hand="left", mano_hdf5_path=None, mano_root=None, progress_callback=None
):
    """
    测试坐标系变换：将触觉数据变换到物体坐标系，并导出手部 OBJ 供后续压力计算。
    progress_callback: 可选 (current, total)。
    """
    if hdf5_path is None:
        hdf5_path = config.DEFAULT_HDF5_PATH
    if mano_root is None:
        mano_root = config.MANO_ASSETS_ROOT
    mano_hdf5 = mano_hdf5_path if mano_hdf5_path else config.DEFAULT_MANO_HDF5_PATH
    if not os.path.isfile(mano_hdf5):
        mano_hdf5 = hdf5_path
    print("Testing coordinate transformation...")
    frame_data = load_selected_frames_data(hdf5_path, which_hand=which_hand)
    if frame_data is None:
        return
    frame_list = list(frame_data.items())
    total_frames = len(frame_list)
    for i, (frame_idx, data) in enumerate(frame_list):
        if progress_callback and total_frames:
            progress_callback(i + 1, total_frames)
        print(f"\n=== Processing Frame {frame_idx} ===")
        tactile_dict = data["tactile_data"]
        hand_pose = data["hand_pose"]
        object_pose = data["object_pose"]
        print("  Loading real tactile data using DataLoaderV3...")
        loader = DataLoaderV3(hdf5_path, which_hand=which_hand)
        tactile_result = loader.get_links_press_press_rel_world(frame_indices=[frame_idx])
        if tactile_result is None:
            print("  No tactile data available for this frame")
            loader.close()
            continue
        links_press_tf_rel_world, links_press_force_rel_link_base, _, _ = tactile_result
        all_positions = []
        all_forces = []
        print(f"  Available sensors: {list(links_press_tf_rel_world.keys())}")
        for sensor_name in links_press_tf_rel_world.keys():
            press_tf_world = links_press_tf_rel_world[sensor_name][0]
            press_positions_world = press_tf_world[:, :3, 3]
            press_forces_link = links_press_force_rel_link_base[sensor_name][0]
            force_magnitudes = np.linalg.norm(press_forces_link, axis=1)
            active_indices = force_magnitudes > 1e-6
            if np.any(active_indices):
                active_positions = press_positions_world[active_indices]
                active_forces = press_forces_link[active_indices]
                print(f"  Sensor {sensor_name}: {len(active_positions)} active tactile points")
                all_positions.extend(active_positions)
                all_forces.extend(active_forces)
        loader.close()
        if len(all_positions) > 0:
            all_positions = np.array(all_positions)
            all_forces = np.array(all_forces)
            print(f"  Total tactile points: {len(all_positions)}")
            object_pose_inv = np.linalg.inv(object_pose)
            positions_homo = np.hstack([all_positions, np.ones((all_positions.shape[0], 1))])
            object_positions_homo = (object_pose_inv @ positions_homo.T).T
            object_positions = object_positions_homo[:, :3]
            object_rotation = object_pose_inv[:3, :3]
            object_forces = (object_rotation @ all_forces.T).T
            transformed_data = {
                "frame_idx": frame_idx,
                "object_positions": object_positions.tolist(),
                "object_forces": object_forces.tolist(),
                "original_positions": all_positions.tolist(),
                "original_forces": all_forces.tolist(),
                "data_source": "real_tactile_data",
                "coordinate_notes": {
                    "positions": "transformed from world to object coordinates",
                    "forces": "link coordinate system (needs further analysis)",
                },
            }
            io_utils.ensure_output_dir()
            output_file = io_utils.out_path(f"transformed_frame_{frame_idx}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(transformed_data, f, indent=2)
            print(f"  Saved transformed data to {output_file}")
            hand_obj_path = io_utils.out_path(f"hand_frame_{frame_idx}.obj")
            if os.path.isfile(mano_hdf5):
                if mano_export.export_hand_obj_for_frame(mano_hdf5, frame_idx, which_hand, mano_root, hand_obj_path):
                    transformed_data["hand_obj_path"] = hand_obj_path
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(transformed_data, f, indent=2)
                else:
                    print("  Hand OBJ export skipped (MANO format required)")
            else:
                print(f"  MANO HDF5 not found: {mano_hdf5}, skip hand OBJ export")
        else:
            print(f"  No active tactile points found in frame {frame_idx}")

# -*- coding: utf-8 -*-
"""触觉可视化工具：MANO 手部 mesh 导出。"""
import os
import re
import sys
import numpy as np
import h5py

from . import io_utils
from .mesh import posquat2transform


def _load_hand_and_object_poses_from_mano_hdf5(hdf5_path, frame_idx, which_hand="right"):
    """
    从 MANO 格式 HDF5（/dataset/observation/{hand}hand/...）读取指定帧的手部位姿与物体位姿。
    Returns:
        hand_pose_world: (4, 4), object_pose_world: (4, 4), hand_joints: (48,) 或 None
    """
    with h5py.File(hdf5_path, "r") as root:
        obs = root.get("/dataset/observation")
        if obs is None:
            return None, None, None
        hand_grp = f"/dataset/observation/{which_hand}hand"
        handpose_ds = root.get(f"{hand_grp}/handpose_mano/data") or root.get(f"{hand_grp}/handpose/data")
        if handpose_ds is None:
            return None, None, None
        hand_pose_world = posquat2transform(handpose_ds[frame_idx : frame_idx + 1, :7])[0]
        joints_data = root.get(f"{hand_grp}/joints_mano/data")
        if joints_data is None:
            joints_data = root.get(f"{hand_grp}/joints/data")
        hand_joints = joints_data[frame_idx] if joints_data is not None else None
        obj_keys = sorted([k for k in obs.keys() if re.match(r"^obj\d+$", k)])
        if not obj_keys:
            return hand_pose_world, None, hand_joints
        obj_data = obs[obj_keys[0]]["data"]
        object_pose_world = posquat2transform(obj_data[frame_idx : frame_idx + 1, :7])[0]
    return hand_pose_world, object_pose_world, hand_joints


def _pose_to_object_frame(pose_world_4x4, object_pose_world_4x4):
    """将世界系下的 4x4 位姿变换到物体坐标系。"""
    obj_inv = np.linalg.inv(np.asarray(object_pose_world_4x4))
    return obj_inv @ np.asarray(pose_world_4x4)


def _get_hand_vertices_faces_in_object_frame(
    hand_pose_obj_4x4, hand_joints, mano_assets_root, which_hand
):
    """
    使用 manotorch 得到手部 mesh（腕部局部系），再变换到物体坐标系。
    Returns: (vertices, faces) 或 (None, None)。
    """
    try:
        import torch
        from manotorch.manolayer import ManoLayer
        from manotorch.axislayer import AxisLayerFK
    except ImportError as e:
        print(f"  未安装 manotorch，跳过手部 OBJ 导出: {e}", file=sys.stderr)
        return None, None
    if hand_joints is None or len(hand_joints) < 48:
        return None, None
    if not os.path.isdir(mano_assets_root):
        print(f"  MANO 资源路径不存在: {mano_assets_root}", file=sys.stderr)
        return None, None
    side = "right" if which_hand == "right" else "left"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mano_layer = ManoLayer(
        rot_mode="axisang",
        center_idx=9,
        mano_assets_root=mano_assets_root,
        use_pca=False,
        side=side,
        flat_hand_mean=True,
    )
    axis_fk = AxisLayerFK(side=side, mano_assets_root=mano_assets_root)
    betas = torch.tensor(
        [0.9861,  3.0000, -3.0000, -3.0000, -3.0000, -3.0000,  3.0000,  3.0000, -0.2849,  3.0000],
        dtype=torch.float32,
        device=device,
    )
    joints = torch.tensor(hand_joints, dtype=torch.float32, device=device).reshape(1, 16, 3)
    composed_aa = axis_fk.compose(joints).clone().reshape(1, 48)
    mano_out = mano_layer(composed_aa, betas.unsqueeze(0))
    verts = mano_out.verts[0].detach().cpu().numpy()
    faces = mano_layer.th_faces.detach().cpu().numpy()
    T = np.asarray(hand_pose_obj_4x4, dtype=np.float64)
    ones = np.ones((verts.shape[0], 1))
    verts_h = np.hstack([verts, ones])
    verts_obj = (T @ verts_h.T).T[:, :3]
    return verts_obj, faces


def export_hand_obj_for_frame(mano_hdf5_path, frame_idx, which_hand, mano_root, out_path):
    """
    导出手部 mesh（物体坐标系）为 OBJ，供压力计算使用。
    成功返回 True，失败返回 False（如 HDF5 无 joints 或非 MANO 格式）。
    """
    hand_pose_world, object_pose_world, hand_joints = _load_hand_and_object_poses_from_mano_hdf5(
        mano_hdf5_path, frame_idx, which_hand=which_hand
    )
    if hand_pose_world is None or object_pose_world is None or hand_joints is None:
        return False
    hand_pose_obj = _pose_to_object_frame(hand_pose_world, object_pose_world)
    verts, faces = _get_hand_vertices_faces_in_object_frame(
        hand_pose_obj, hand_joints, mano_root, which_hand
    )
    if verts is None:
        return False
    io_utils.save_mesh_as_obj(verts, faces, out_path)
    print(f"  已导出手部 OBJ: {out_path} ({verts.shape[0]} 顶点, {faces.shape[0]} 面)")
    return True

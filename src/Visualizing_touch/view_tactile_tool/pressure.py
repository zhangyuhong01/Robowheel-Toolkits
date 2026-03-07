# -*- coding: utf-8 -*-
"""触觉可视化工具：法向压力计算。"""
import os
import json
import numpy as np
from scipy.spatial import KDTree

from . import io_utils
from . import config
from . import mesh
from . import transform


def compute_face_normals(vertices, faces):
    """计算三角面的法向量。Returns: (M, 3) 单位法向量。"""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = np.cross(edge1, edge2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normals = normals / norms
    return normals


def compute_normal_pressure(force_vectors, face_normals):
    """计算力向量在面法向量上的投影（法向压力）。Returns: (N,) 标量压力。"""
    normal_pressures = np.sum(force_vectors * face_normals, axis=1)
    normal_pressures = np.abs(normal_pressures)
    return normal_pressures


def project_and_compute_pressure(object_positions, object_forces, obj_file_path):
    """
    将触觉点投影到物体表面并计算法向压力。
    Returns: dict with projected_points, projected_face_indices, projected_distances,
             face_normals, corresponding_normals, normal_pressures, vertices, faces
    """
    print(f"Loading mesh from {obj_file_path}...")
    vertices, faces = io_utils.load_obj(obj_file_path)
    print(f"  Mesh info: {len(vertices)} vertices, {len(faces)} faces")
    print("Computing face normals...")
    face_normals = compute_face_normals(vertices, faces)
    print("Building KDTree for fast nearest neighbor search...")
    kdtree = KDTree(vertices)
    print("Projecting tactile points to object surface...")
    projected_points = []
    projected_face_indices = []
    projected_distances = []
    for i, point in enumerate(object_positions):
        if i % 5 == 0:
            print(f"  Progress: {i}/{len(object_positions)}")
        projected_point, face_idx, distance = mesh.project_point_to_object(
            point, vertices, faces, kdtree
        )
        projected_points.append(projected_point)
        projected_face_indices.append(face_idx)
        projected_distances.append(distance)
    projected_points = np.array(projected_points)
    projected_face_indices = np.array(projected_face_indices, dtype=int)
    projected_distances = np.array(projected_distances)
    print(f"Projection complete. Average distance: {np.mean(projected_distances):.4f}")
    corresponding_normals = face_normals[projected_face_indices]
    print("Computing normal pressures...")
    normal_pressures = compute_normal_pressure(object_forces, corresponding_normals)
    print(f"Pressure statistics: min={np.min(normal_pressures):.4f} max={np.max(normal_pressures):.4f} mean={np.mean(normal_pressures):.4f} std={np.std(normal_pressures):.4f}")
    return {
        "projected_points": projected_points,
        "projected_face_indices": projected_face_indices,
        "projected_distances": projected_distances,
        "face_normals": face_normals,
        "corresponding_normals": corresponding_normals,
        "normal_pressures": normal_pressures,
        "vertices": vertices,
        "faces": faces,
    }


def test_pressure_calculation(obj_path=None, progress_callback=None):
    """
    对 selected_frames.json 中的每一帧，读取 transformed_frame_{帧号}.json，
    计算压力并保存 pressure_frame_{帧号}.json。
    progress_callback: 可选 (current, total)。
    """
    if obj_path is None:
        obj_path = config.DEFAULT_OBJ_PATH
    print("Testing pressure calculation...")
    frame_indices = transform.get_selected_frame_indices()
    if not frame_indices:
        print("No selected frames found. Please run select_frames first.")
        return None
    if not os.path.exists(obj_path):
        print(f"OBJ file not found: {obj_path}")
        return None
    io_utils.ensure_output_dir()
    print(f"Selected frames: {frame_indices}")
    results_all = {}
    total_frames = len(frame_indices)
    for i, frame_idx in enumerate(frame_indices):
        if progress_callback and total_frames:
            progress_callback(i + 1, total_frames)
        transformed_file = io_utils.out_path(f"transformed_frame_{frame_idx}.json")
        if not os.path.exists(transformed_file):
            print(f"Skip frame {frame_idx}: {transformed_file} not found (run test_transform first)")
            continue
        try:
            with open(transformed_file, "r", encoding="utf-8") as f:
                transformed_data = json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skip frame {frame_idx}: failed to load {transformed_file} ({e})")
            continue
        object_positions = np.array(transformed_data["object_positions"])
        object_forces = np.array(transformed_data["object_forces"])
        print(f"Processing frame {frame_idx} with {len(object_positions)} tactile points")
        results = project_and_compute_pressure(object_positions, object_forces, obj_path)
        output_data = {
            "frame_idx": frame_idx,
            "projected_points": results["projected_points"].tolist(),
            "projected_face_indices": results["projected_face_indices"].tolist(),
            "projected_distances": results["projected_distances"].tolist(),
            "normal_pressures": results["normal_pressures"].tolist(),
            "pressure_statistics": {
                "min": float(np.min(results["normal_pressures"])),
                "max": float(np.max(results["normal_pressures"])),
                "mean": float(np.mean(results["normal_pressures"])),
                "std": float(np.std(results["normal_pressures"])),
            },
            "obj_file": obj_path,
        }
        hand_obj_path = io_utils.out_path(f"hand_frame_{frame_idx}.obj")
        if os.path.exists(hand_obj_path):
            print(f"  Computing pressure on hand mesh: {hand_obj_path}")
            hand_results = project_and_compute_pressure(object_positions, object_forces, hand_obj_path)
            output_data["hand_projected_points"] = hand_results["projected_points"].tolist()
            output_data["hand_projected_face_indices"] = hand_results["projected_face_indices"].tolist()
            output_data["hand_projected_distances"] = hand_results["projected_distances"].tolist()
            output_data["hand_normal_pressures"] = hand_results["normal_pressures"].tolist()
            output_data["hand_pressure_statistics"] = {
                "min": float(np.min(hand_results["normal_pressures"])),
                "max": float(np.max(hand_results["normal_pressures"])),
                "mean": float(np.mean(hand_results["normal_pressures"])),
                "std": float(np.std(hand_results["normal_pressures"])),
            }
            output_data["hand_obj_file"] = hand_obj_path
        output_file = io_utils.out_path(f"pressure_frame_{frame_idx}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved {output_file}")
        results_all[frame_idx] = results
    if not results_all:
        print("No frame was processed. Ensure transformed_frame_*.json exist (run test_transform).")
        return None
    print(f"Pressure calculation done for {len(results_all)} frame(s): {list(results_all.keys())}")
    return results_all

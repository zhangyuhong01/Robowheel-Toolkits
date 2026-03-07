# -*- coding: utf-8 -*-
"""触觉可视化工具：高斯压力扩散。"""
import os
import json
import numpy as np

from . import io_utils
from . import config
from . import transform


def compute_face_centers(vertices, faces):
    """计算三角面的中心点。Returns: (M, 3)。"""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_centers = (v0 + v1 + v2) / 3.0
    return face_centers


def distribute_pressure_gaussian(tactile_points, tactile_pressures, face_centers, sigma=0.01):
    """
    使用高斯分布将触觉点压力扩散到周围的三角面。
    Returns: (M,) 每个三角面的扩散压力值。
    """
    print(f"Distributing pressure using Gaussian blur (sigma={sigma:.4f})...")
    n_faces = len(face_centers)
    n_tactile = len(tactile_points)
    face_pressures = np.zeros(n_faces)
    print(f"  Processing {n_tactile} tactile points -> {n_faces} faces")
    for i, (tactile_point, pressure) in enumerate(zip(tactile_points, tactile_pressures)):
        if i % 5 == 0:
            print(f"  Progress: {i}/{n_tactile}")
        distances = np.linalg.norm(face_centers - tactile_point, axis=1)
        gaussian_weights = np.exp(-distances**2 / (2 * sigma**2))
        face_pressures += pressure * gaussian_weights
    print("Gaussian distribution complete")
    print(f"  Face pressure statistics: min={np.min(face_pressures):.6f} max={np.max(face_pressures):.6f} mean={np.mean(face_pressures):.6f} non_zero={np.sum(face_pressures > 1e-6)}/{n_faces}")
    return face_pressures


def test_gaussian_pressure_distribution(obj_path=None, progress_callback=None):
    """
    对 selected_frames.json 中的每一帧，读取 pressure_frame_{帧号}.json，
    计算高斯扩散并保存 gaussian_pressure_frame_{帧号}.json。
    progress_callback: 可选 (current, total)。
    """
    if obj_path is None:
        obj_path = config.DEFAULT_OBJ_PATH
    print("Testing Gaussian pressure distribution...")
    frame_indices = transform.get_selected_frame_indices()
    if not frame_indices:
        print("No selected frames found. Please run select_frames first.")
        return None
    if not os.path.exists(obj_path):
        print(f"OBJ file not found: {obj_path}")
        return None
    io_utils.ensure_output_dir()
    print(f"Selected frames: {frame_indices}")
    sigma_values = [0.005, 0.01, 0.02, 0.05]
    processed = 0
    results = {}
    total_frames = len(frame_indices)
    for i, frame_idx in enumerate(frame_indices):
        if progress_callback and total_frames:
            progress_callback(i + 1, total_frames)
        pressure_file = io_utils.out_path(f"pressure_frame_{frame_idx}.json")
        if not os.path.exists(pressure_file):
            print(f"Skip frame {frame_idx}: {pressure_file} not found (run test_pressure first)")
            continue
        try:
            with open(pressure_file, "r", encoding="utf-8") as f:
                pressure_data = json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skip frame {frame_idx}: failed to load {pressure_file} ({e})")
            continue
        projected_points = np.array(pressure_data["projected_points"])
        normal_pressures = np.array(pressure_data["normal_pressures"])
        print(f"\nProcessing frame {frame_idx} with {len(projected_points)} tactile points")
        print(f"Loading mesh from {obj_path}...")
        vertices, faces = io_utils.load_obj(obj_path)
        print(f"  Mesh info: {len(vertices)} vertices, {len(faces)} faces")
        face_centers = compute_face_centers(vertices, faces)
        results = {}
        for sigma in sigma_values:
            print(f"  Object sigma = {sigma}")
            face_pressures = distribute_pressure_gaussian(
                projected_points, normal_pressures, face_centers, sigma=sigma
            )
            results[f"sigma_{sigma}"] = {
                "sigma": sigma,
                "face_pressures": face_pressures.tolist(),
                "statistics": {
                    "min": float(np.min(face_pressures)),
                    "max": float(np.max(face_pressures)),
                    "mean": float(np.mean(face_pressures)),
                    "std": float(np.std(face_pressures)),
                    "non_zero_faces": int(np.sum(face_pressures > 1e-6)),
                    "total_faces": len(face_pressures),
                },
            }
        output_data = {
            "frame_idx": frame_idx,
            "tactile_points": projected_points.tolist(),
            "tactile_pressures": normal_pressures.tolist(),
            "face_centers": face_centers.tolist(),
            "gaussian_results": results,
            "mesh_info": {
                "vertices_count": len(vertices),
                "faces_count": len(faces),
                "obj_file": obj_path,
            },
        }
        if "hand_normal_pressures" in pressure_data and "hand_projected_points" in pressure_data:
            hand_obj_path = io_utils.resolve_out_path(
                pressure_data.get("hand_obj_file", f"hand_frame_{frame_idx}.obj")
            )
            if os.path.exists(hand_obj_path):
                print(f"  Hand Gaussian from {hand_obj_path}")
                hand_projected = np.array(pressure_data["hand_projected_points"])
                hand_pressures = np.array(pressure_data["hand_normal_pressures"])
                hand_vertices, hand_faces = io_utils.load_obj(hand_obj_path)
                hand_face_centers = compute_face_centers(hand_vertices, hand_faces)
                hand_results = {}
                for sigma in sigma_values:
                    hand_face_pressures = distribute_pressure_gaussian(
                        hand_projected, hand_pressures, hand_face_centers, sigma=sigma
                    )
                    hand_results[f"sigma_{sigma}"] = {
                        "sigma": sigma,
                        "face_pressures": hand_face_pressures.tolist(),
                        "statistics": {
                            "min": float(np.min(hand_face_pressures)),
                            "max": float(np.max(hand_face_pressures)),
                            "mean": float(np.mean(hand_face_pressures)),
                            "std": float(np.std(hand_face_pressures)),
                            "non_zero_faces": int(np.sum(hand_face_pressures > 1e-6)),
                            "total_faces": len(hand_face_pressures),
                        },
                    }
                output_data["hand_tactile_points"] = pressure_data["hand_projected_points"]
                output_data["hand_tactile_pressures"] = pressure_data["hand_normal_pressures"]
                output_data["hand_face_centers"] = hand_face_centers.tolist()
                output_data["hand_gaussian_results"] = hand_results
                output_data["hand_mesh_info"] = {
                    "vertices_count": len(hand_vertices),
                    "faces_count": len(hand_faces),
                    "obj_file": hand_obj_path,
                }
        output_file = io_utils.out_path(f"gaussian_pressure_frame_{frame_idx}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Saved {output_file}")
        processed += 1
        print(f"  Sigma analysis for frame {frame_idx}:")
        for sigma in sigma_values:
            stats = results[f"sigma_{sigma}"]["statistics"]
            coverage = stats["non_zero_faces"] / stats["total_faces"] * 100
            print(f"    sigma={sigma:5.3f}: {stats['non_zero_faces']:5d} faces ({coverage:5.1f}%), max={stats['max']:.6f}")
    if processed == 0:
        print("No frame was processed. Ensure pressure_frame_*.json exist (run test_pressure).")
        return None
    print(f"\nGaussian distribution done for {processed} frame(s).")
    return results

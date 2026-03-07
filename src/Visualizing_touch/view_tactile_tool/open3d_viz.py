# -*- coding: utf-8 -*-
"""触觉可视化工具：Open3D 压力云图可视化。"""
import os
import json
import numpy as np

from . import io_utils
from . import config
from . import transform


def _open3d_mesh_vertex_pressures(vertices, faces, face_pressures):
    """从面压力插值到顶点压力，返回 vertex_pressures。"""
    vertex_pressures = np.zeros(len(vertices))
    vertex_counts = np.zeros(len(vertices))
    for i, face in enumerate(faces):
        for vertex_idx in face:
            vertex_pressures[vertex_idx] += face_pressures[i]
            vertex_counts[vertex_idx] += 1
    vertex_counts[vertex_counts == 0] = 1
    return vertex_pressures / vertex_counts


def visualize_pressure_map_open3d(
    vertices, faces, face_pressures, tactile_points=None,
    title="Pressure Distribution", sigma_value=None, save_image=True, frame_idx=None
):
    """使用 Open3D 在物体表面渲染压力分布热力图。Returns: Open3D mesh 或 None。"""
    try:
        import open3d as o3d
        import matplotlib.pyplot as plt
    except ImportError:
        print("Missing Open3D. Please install: pip install open3d")
        return None
    print("Creating Open3D surface pressure visualization...")
    print(f"  Mesh: {len(vertices)} vertices, {len(faces)} faces")
    print(f"  Pressure range: {np.min(face_pressures):.6f} - {np.max(face_pressures):.6f}")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    vertex_pressures = _open3d_mesh_vertex_pressures(vertices, faces, face_pressures)
    min_pressure = vertex_pressures.min()
    max_pressure = vertex_pressures.max()
    if max_pressure > min_pressure:
        normalized_pressures = (vertex_pressures - min_pressure) / (max_pressure - min_pressure)
    else:
        normalized_pressures = np.zeros_like(vertex_pressures)
    cmap = plt.cm.hot
    colors = cmap(normalized_pressures)[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1200, height=900)
    vis.add_geometry(mesh)
    tactile_spheres = []
    if tactile_points is not None:
        print(f"  Adding {len(tactile_points)} tactile points")
        for point in tactile_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
            sphere.translate(point)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])
            tactile_spheres.append(sphere)
            vis.add_geometry(sphere)
    render_option = vis.get_render_option()
    render_option.mesh_show_wireframe = False
    render_option.mesh_show_back_face = True
    render_option.light_on = True
    render_option.background_color = np.array([1.0, 1.0, 1.0])
    view_control = vis.get_view_control()
    view_control.set_front([0.5, 0.5, 0.7])
    view_control.set_lookat([0.0, 0.0, 0.0])
    view_control.set_up([0.0, 0.0, 1.0])
    view_control.set_zoom(0.8)
    for _ in range(10):
        vis.poll_events()
        vis.update_renderer()
    if save_image and sigma_value is not None:
        io_utils.ensure_output_dir()
        base = f"pressure_open3d_{sigma_value}"
        filename = io_utils.out_path(f"{base}_frame{frame_idx}.png" if frame_idx is not None else f"{base}.png")
        vis.capture_screen_image(filename)
        print(f"  Saved Open3D visualization to {filename}")
    vis.run()
    vis.destroy_window()
    print("  Open3D visualization complete!")
    return mesh


def visualize_pressure_map_open3d_dual(
    obj_vertices, obj_faces, obj_face_pressures,
    hand_vertices, hand_faces, hand_face_pressures,
    tactile_points=None,
    title="Pressure (Object + Hand)",
    sigma_value=None, save_image=True, frame_idx=None
):
    """使用 Open3D 同时可视化物体与手部的压力分布（同一场景、统一颜色条）。"""
    try:
        import open3d as o3d
        import matplotlib.pyplot as plt
    except ImportError:
        print("Missing Open3D. Please install: pip install open3d")
        return None
    vp_obj = _open3d_mesh_vertex_pressures(obj_vertices, obj_faces, obj_face_pressures)
    vp_hand = _open3d_mesh_vertex_pressures(hand_vertices, hand_faces, hand_face_pressures)
    vmin = min(vp_obj.min(), vp_hand.min())
    vmax = max(vp_obj.max(), vp_hand.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    print("Creating Open3D dual (object + hand) pressure visualization...")
    print(f"  Object: {len(obj_vertices)} verts, {len(obj_faces)} faces")
    print(f"  Hand: {len(hand_vertices)} verts, {len(hand_faces)} faces")
    print(f"  Pressure range (unified): {vmin:.6f} - {vmax:.6f}")
    cmap = plt.cm.hot

    def mesh_from_verts_faces_pressures(vertices, faces, vertex_pressures):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        norm = (vertex_pressures - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0, 1)
        colors = cmap(norm)[:, :3]
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        mesh.compute_vertex_normals()
        return mesh

    mesh_obj = mesh_from_verts_faces_pressures(obj_vertices, obj_faces, vp_obj)
    mesh_hand = mesh_from_verts_faces_pressures(hand_vertices, hand_faces, vp_hand)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1200, height=900)
    vis.add_geometry(mesh_obj)
    vis.add_geometry(mesh_hand)
    if tactile_points is not None:
        print(f"  Adding {len(tactile_points)} tactile points")
        for point in tactile_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
            sphere.translate(point)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])
            vis.add_geometry(sphere)
    render_option = vis.get_render_option()
    render_option.mesh_show_wireframe = False
    render_option.mesh_show_back_face = True
    render_option.light_on = True
    render_option.background_color = np.array([1.0, 1.0, 1.0])
    view_control = vis.get_view_control()
    view_control.set_front([0.5, 0.5, 0.7])
    view_control.set_lookat([0.0, 0.0, 0.0])
    view_control.set_up([0.0, 0.0, 1.0])
    view_control.set_zoom(0.8)
    for _ in range(10):
        vis.poll_events()
        vis.update_renderer()
    if save_image and sigma_value is not None:
        io_utils.ensure_output_dir()
        base = f"pressure_open3d_dual_{sigma_value}"
        filename = io_utils.out_path(f"{base}_frame{frame_idx}.png" if frame_idx is not None else f"{base}.png")
        vis.capture_screen_image(filename)
        print(f"  Saved Open3D dual visualization to {filename}")
    vis.run()
    vis.destroy_window()
    print("  Open3D dual (object + hand) visualization complete!")
    return mesh_obj


def test_open3d_visualization(obj_path=None, frame_idx=None):
    """
    测试 Open3D 压力云图可视化。多帧时依次可视化，窗口标题显示当前帧数和评分。
    frame_idx: 单帧时传 int；None 时对 selected_frames 中所有有 gaussian 文件的帧依次可视化。
    """
    if obj_path is None:
        obj_path = config.DEFAULT_OBJ_PATH
    print("Testing Open3D pressure visualization...")
    if frame_idx is None:
        frame_list = transform.get_selected_frame_indices() or []
    else:
        frame_list = [int(frame_idx)]
    quality_list = transform.get_selected_frames_quality_scores()
    quality_by_frame = {}
    if quality_list and frame_list:
        for i, fid in enumerate(frame_list):
            if i < len(quality_list) and quality_list[i] is not None:
                quality_by_frame[fid] = quality_list[i]
    if not frame_list:
        print("No selected frames. Run select_frames first or pass frame_idx.")
        return
    frames_to_show = []
    for fid in frame_list:
        gf, _ = transform.get_first_available_gaussian_file(fid)
        if gf:
            frames_to_show.append(fid)
    if not frames_to_show:
        print("No Gaussian pressure data found. Run test_gaussian first.")
        return
    total_frames = len(frames_to_show)
    print(f"Open3D will show {total_frames} frame(s) in order. Close each window to see the next.")
    mode = config.VISUALIZATION_MODE
    sigma_values = [0.01]
    for idx_one, frame_idx_cur in enumerate(frames_to_show):
        current_n = idx_one + 1
        score_str = f"{quality_by_frame.get(frame_idx_cur, 0):.2f}" if frame_idx_cur in quality_by_frame else "N/A"
        title_prefix = f"Frame {frame_idx_cur} | Score: {score_str} | ({current_n}/{total_frames})"
        gaussian_file = io_utils.out_path(f"gaussian_pressure_frame_{frame_idx_cur}.json")
        try:
            with open(gaussian_file, "r", encoding="utf-8") as f:
                gaussian_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        tactile_points = np.array(gaussian_data["tactile_points"])
        frame_idx_loaded = gaussian_data["frame_idx"]
        has_hand = "hand_gaussian_results" in gaussian_data
        print(f"[{current_n}/{total_frames}] Frame {frame_idx_loaded} with {len(tactile_points)} tactile points (object+hand: {has_hand})")
        if not os.path.exists(obj_path):
            print(f"OBJ file not found: {obj_path}")
            return
        vertices, faces = io_utils.load_obj(obj_path)
        hand_vertices, hand_faces = None, None
        if has_hand and "hand_mesh_info" in gaussian_data:
            hand_obj_path = io_utils.resolve_out_path(
                gaussian_data["hand_mesh_info"].get("obj_file", f"hand_frame_{frame_idx_loaded}.obj")
            )
            if os.path.exists(hand_obj_path):
                hand_vertices, hand_faces = io_utils.load_obj(hand_obj_path)
        for sigma in sigma_values:
            sigma_key = f"sigma_{sigma}"
            if sigma_key not in gaussian_data.get("gaussian_results", {}):
                continue
            face_pressures = np.array(gaussian_data["gaussian_results"][sigma_key]["face_pressures"])
            if mode == "object":
                visualize_pressure_map_open3d(
                    vertices, faces, face_pressures, tactile_points,
                    title=f"{title_prefix} (Object, sigma={sigma})",
                    sigma_value=sigma, save_image=True,
                    frame_idx=frame_idx_loaded
                )
            elif mode == "hand":
                if has_hand and hand_vertices is not None and sigma_key in gaussian_data.get("hand_gaussian_results", {}):
                    hand_face_pressures = np.array(gaussian_data["hand_gaussian_results"][sigma_key]["face_pressures"])
                    hand_tactile = np.array(gaussian_data.get("hand_tactile_points", tactile_points))
                    visualize_pressure_map_open3d(
                        hand_vertices, hand_faces, hand_face_pressures, hand_tactile,
                        title=f"{title_prefix} (Hand, sigma={sigma})",
                        sigma_value=sigma, save_image=True,
                        frame_idx=frame_idx_loaded
                    )
            else:
                if has_hand and hand_vertices is not None and sigma_key in gaussian_data.get("hand_gaussian_results", {}):
                    hand_face_pressures = np.array(gaussian_data["hand_gaussian_results"][sigma_key]["face_pressures"])
                    visualize_pressure_map_open3d_dual(
                        obj_vertices=vertices, obj_faces=faces, obj_face_pressures=face_pressures,
                        hand_vertices=hand_vertices, hand_faces=hand_faces, hand_face_pressures=hand_face_pressures,
                        tactile_points=tactile_points,
                        title=f"{title_prefix} (Object+Hand, sigma={sigma})",
                        sigma_value=sigma, save_image=True,
                        frame_idx=frame_idx_loaded
                    )
                else:
                    visualize_pressure_map_open3d(
                        vertices, faces, face_pressures, tactile_points,
                        title=f"{title_prefix} (sigma={sigma})",
                        sigma_value=sigma, save_image=True,
                        frame_idx=frame_idx_loaded
                    )
    print("All Open3D visualizations complete!")

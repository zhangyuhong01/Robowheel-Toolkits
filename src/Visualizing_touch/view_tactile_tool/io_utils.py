# -*- coding: utf-8 -*-
"""触觉可视化工具：OBJ 读写与输出目录。"""
import os
import numpy as np

from .config import OUTPUT_DIR


def ensure_output_dir():
    """确保 OUTPUT_DIR 存在，保存前调用。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def out_path(*names):
    """输出目录下的路径。例: out_path('a.png') -> output/a.png"""
    return os.path.join(OUTPUT_DIR, os.path.join(*names) if len(names) > 1 else names[0])


def resolve_out_path(path):
    """将相对路径解析为 OUTPUT_DIR 下的绝对路径；已是绝对路径则原样返回。"""
    if not path:
        return path
    return os.path.join(OUTPUT_DIR, path) if not os.path.isabs(path) else path


def load_obj(file_path):
    """读取 obj，提取顶点和三角形面片。"""
    vertices = []
    faces = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append(list(map(float, line.split()[1:4])))
            elif line.startswith("f "):
                face = []
                for v in line.split()[1:]:
                    vertex_index = v.split("/")[0]
                    face.append(int(vertex_index) - 1)
                if len(face) >= 3:
                    faces.append(face[:3])
    return np.array(vertices), np.array(faces)


def save_mesh_as_obj(vertices, faces, filepath):
    """
    将顶点和面片写入 OBJ 文件（与 load_obj 可互读）。
    vertices: (N, 3)，faces: (F, 3) 为 0-based 顶点索引；OBJ 中 f 使用 1-based。
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for i in range(vertices.shape[0]):
            f.write(f"v {vertices[i, 0]:.6f} {vertices[i, 1]:.6f} {vertices[i, 2]:.6f}\n")
        for i in range(faces.shape[0]):
            a, b, c = faces[i, 0] + 1, faces[i, 1] + 1, faces[i, 2] + 1
            f.write(f"f {a} {b} {c}\n")


def clear_intermediate_files(frame_indices):
    """
    删除 pipeline 运行产生的中间文件（不删 selected_frames.json / .txt）。
    对每帧删除：transformed_frame_{f}.json, pressure_frame_{f}.json,
    gaussian_pressure_frame_{f}.json, hand_frame_{f}.obj
    """
    if not frame_indices:
        return
    removed = []
    for f in frame_indices:
        for name in (
            f"transformed_frame_{f}.json",
            f"pressure_frame_{f}.json",
            f"gaussian_pressure_frame_{f}.json",
            f"hand_frame_{f}.obj",
        ):
            p = out_path(name)
            if os.path.isfile(p):
                try:
                    os.remove(p)
                    removed.append(name)
                except OSError as e:
                    print(f"  [WARN] Failed to remove {p}: {e}")
    if removed:
        print(f"  Cleared {len(removed)} intermediate file(s) in {OUTPUT_DIR}")

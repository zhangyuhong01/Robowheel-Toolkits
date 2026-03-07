# -*- coding: utf-8 -*-
"""触觉可视化工具：mesh 几何与点到表面投影。"""
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R_scipy

from . import io_utils


def posquat2transform(posquat):
    """[N, 7] 位置+四元数 [x,y,z, w,x,y,z] -> [N, 4, 4] 齐次变换矩阵。"""
    posquat = np.asarray(posquat)
    if posquat.ndim == 1:
        posquat = posquat.reshape(1, -1)
    assert posquat.shape[1] >= 7
    tf = np.zeros((posquat.shape[0], 4, 4), dtype=np.float64)
    tf[:, :3, 3] = posquat[:, :3]
    quat = posquat[:, 3:7]  # [w, x, y, z]
    quat_xyzw = np.column_stack([quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]])
    tf[:, :3, :3] = R_scipy.from_quat(quat_xyzw).as_matrix()
    tf[:, 3, 3] = 1.0
    return tf


def closest_point_on_triangle(P, A, B, C):
    """计算 P 到三角形 ABC 的最近点。"""
    AB = B - A
    AC = C - A
    AP = P - A

    d1 = np.dot(AB, AP)
    d2 = np.dot(AC, AP)
    if d1 <= 0 and d2 <= 0:
        return A

    BP = P - B
    d3 = np.dot(AB, BP)
    d4 = np.dot(AC, BP)
    if d3 >= 0 and d4 <= d3:
        return B

    CP = P - C
    d5 = np.dot(AB, CP)
    d6 = np.dot(AC, CP)
    if d6 >= 0 and d5 <= d6:
        return C

    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return A + v * AB

    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return A + w * AC

    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return B + w * (C - B)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return A + AB * v + AC * w


def project_point_to_object(point, vertices, faces, kdtree):
    """计算点到物体表面的最近投影点。"""
    _, nearest_idx = kdtree.query(point)
    nearest_vertex = vertices[nearest_idx]

    adjacent_faces = []
    adjacent_face_indices = []

    for i, face in enumerate(faces):
        if nearest_idx in face:
            adjacent_faces.append(face)
            adjacent_face_indices.append(i)

    min_dist = float("inf")
    projection = None
    best_face_idx = None

    for face, face_idx in zip(adjacent_faces, adjacent_face_indices):
        A, B, C = vertices[face]
        proj = closest_point_on_triangle(point, A, B, C)
        dist = np.linalg.norm(point - proj)
        if dist < min_dist:
            min_dist = dist
            projection = proj
            best_face_idx = face_idx

    if projection is None:
        projection = nearest_vertex
        best_face_idx = 0
        min_dist = np.linalg.norm(point - nearest_vertex)

    return projection, best_face_idx, min_dist


def project_point_cloud(points, obj_path):
    """投影整个点云到物体表面。"""
    vertices, faces = io_utils.load_obj(obj_path)
    kdtree = KDTree(vertices)
    projections = []
    for i, point in enumerate(points):
        print(f"dealing with point {i+1}/{len(points)}")
        proj = project_point_to_object(point, vertices, faces, kdtree)
        projections.append(proj[0])
    return np.array(projections)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ir_geometry.py — 由 depth_exr + COLMAP 位姿重建"逐像素世界坐标与有效性掩码"

用途：给温度场/法线提供几何载体；并把地面、天空从资产像素中区分出来。
（IR 渲染里地面与天空是**常色块**、不含温度，必须屏蔽，否则会污染温度统计与烘焙。）

约定（与 mesh-splatting 项目一致）：
  * depth_exr = 相机轴 Z，单位米，float32；背景哨兵 >= 1e4
  * COLMAP: p_cam = R(w2c) @ p_world + tvec  =>  p_world = R(w2c)^T @ (p_cam - tvec)
"""
import os
import glob

import numpy as np

try:
    import OpenEXR
except ImportError:
    OpenEXR = None

import sys
sys.path.insert(0, os.environ.get("MESHSPLAT_ROOT", "/home/yyc/Documents/Ones/mesh-splatting"))
try:
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
except Exception:
    read_extrinsics_binary = read_intrinsics_binary = qvec2rotmat = None

DEPTH_SENTINEL = 1e4


def intrinsics_poses(dataset):
    if read_extrinsics_binary is None:
        raise RuntimeError("无法导入 scene.colmap_loader：请设置环境变量 MESHSPLAT_ROOT=<mesh-splatting 仓库根>")
    ext = read_extrinsics_binary(os.path.join(dataset, "sparse/0/images.bin"))
    cam = list(read_intrinsics_binary(os.path.join(dataset, "sparse/0/cameras.bin")).values())[0]
    m, p = cam.model, list(cam.params)
    if m == "PINHOLE":
        K = (p[0], p[1], p[2], p[3])
    elif m in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        K = (p[0], p[0], p[1], p[2])
    elif m in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        K = (p[0], p[1], p[2], p[3])
    else:
        raise ValueError(f"unsupported camera model: {m}")
    return K, {os.path.splitext(v.name)[0]: v for v in ext.values()}


def read_depth(path):
    with OpenEXR.File(path) as f:
        ch = f.channels()
        arr = np.asarray(list(ch.values())[0].pixels)
    Z = arr[..., 0] if arr.ndim == 3 else arr
    return Z.astype(np.float32)


def unproject_world(Z, K, ent):
    """相机轴深度 -> 世界坐标 XYZ (H,W,3)"""
    H, W = Z.shape
    fx, fy, cx, cy = K
    u = np.arange(W)[None, :].repeat(H, 0).astype(np.float32)
    v = np.arange(H)[:, None].repeat(W, 1).astype(np.float32)
    xc = (u - cx) / fx * Z
    yc = (v - cy) / fy * Z
    pc = np.stack([xc, yc, Z], -1)                      # (H,W,3) 相机系
    R = qvec2rotmat(ent.qvec)                           # world -> cam
    t = np.asarray(ent.tvec, np.float32)
    XYZ = (pc.reshape(-1, 3) - t) @ R                   # p_world = R^T (p_cam - t)
    return XYZ.reshape(H, W, 3)


def valid_depth(Z):
    return (Z > 1e-3) & (Z < DEPTH_SENTINEL)


def ground_mask(XYZ, ground_bbox, tol=0.35):
    """地面物体像素：落在其 bbox 的 XY 范围内、且高度接近其上表面。

    IR 渲染里地面被替换为常色块（不含温度），烘焙/统计时必须剔除。
    """
    if not ground_bbox:
        return np.zeros(XYZ.shape[:2], bool)
    mn = np.asarray(ground_bbox["bbox_min"], np.float32)
    mx = np.asarray(ground_bbox["bbox_max"], np.float32)
    x, y, z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    pad = max(tol, 0.0)
    inside_xy = (x > mn[0] - pad) & (x < mx[0] + pad) & (y > mn[1] - pad) & (y < mx[1] + pad)
    near_top = np.abs(z - mx[2]) <= tol
    return inside_xy & near_top


def list_frames(depth_dir, depth_ext=".exr"):
    return sorted(glob.glob(os.path.join(depth_dir, f"*{depth_ext}")))

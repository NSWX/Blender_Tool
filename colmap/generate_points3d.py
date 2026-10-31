"""
generate_points3d.py — 由深度图反投影生成 points3D.bin + points3D_normals.npy
===============================================================================
职责边界（与 export_sparse_from_blender.py 分工）：
    · export_sparse_from_blender.py（Blender 内）生成 cameras.bin + images.bin
    · 本脚本（mesh_splatting 环境运行，需要 cv2/OpenEXR/scipy）：
        depth_exr 反投影 → 世界系点云 → 统计滤波 + 体素自适应 →
        points3D.bin + points3D_normals.npy（写入 sparse/0）

用法（在 micromamba mesh_splatting 环境）：
    python generate_points3d.py --data_dir <数据集> --sparse <sparse目录 或 与数据同级>
        [--points 120000] [--per_view 2000] [--voxel <m>] [--no_stat] [--far 1e9]

    --sparse 若为数据集根目录（含 sparse/0），自动定位稀疏目录。

坐标系（方案 A = 3DGS/NeRF Blender 标准）：
    点云保持 Blender 世界坐标；位姿（images.bin 的 w2c）→ 反投影：
        depth EXR 沿相机 -Z，在 COLMAP 系下对应 +Z 前向深度 d：
        X_c=(u-cx)/fx·d, Y_c=(v-cy)/fy·d, Z_c=d
        P_world = w2c⁻¹ @ [X_c, Y_c, Z_c, 1]

输入：
    <data_dir>/images/<name>.png           RGB（与 images.bin 名字一致）
    <data_dir>/depth_exr/<name>.exr        Blender Z pass（R 通道，far≈1e10）
    <data_dir>/normals/<name>.png          世界系法线（可选）
输出：
    <sparse>/0/points3D.bin                体素自适应后的种子点云
    <sparse>/0/points3D_normals.npy        与 points3D 同序的世界系法线
"""

import os
import sys
import struct
import argparse

import numpy as np
from OpenEXR import InputFile
import Imath
import cv2

# ==================== 0. 默认配置 ====================
DEFAULT_POINTS   = 120_000   # 目标点数（体素自适应）
DEFAULT_PER_VIEW = 2000      # 每视图采样像素上限
DEPTH_FAR        = 1e9       # 深度远平面阈值（>= 视为背景）


# ==================== 1. 读取 sparse ====================
def read_intrinsics_bin(path):
    """cameras.bin -> {cam_id: (model, W, H, params)}"""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id, model = struct.unpack("<ii", f.read(8))
            W, H = struct.unpack("<QQ", f.read(16))
            n_params = 4 if model == 1 else 3  # PINHOLE=1(4参数), SIMPLE_PINHOLE=2(3参数)
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            out[cam_id] = (model, W, H, params)
    return out


def read_images_bin(path):
    """images.bin -> [(name, w2c 4x4)]（qvec+tvec 转回矩阵）"""
    def qvec2rotmat(q):
        qw, qx, qy, qz = q
        R = np.array([
            [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2]])
        return R

    entries = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4)  # image_id (int32)
            q = struct.unpack("<dddd", f.read(32))
            t = struct.unpack("<ddd", f.read(24))
            f.read(4)  # camera_id (int32)
            name = b""
            while True:
                b = f.read(1)
                if b == b"\0" or not b:
                    break
                name += b
            f.read(8)  # num points2D
            M = np.eye(4)
            M[:3, :3] = qvec2rotmat(np.array(q))
            M[:3, 3] = np.array(t)
            entries.append((name.decode("utf-8"), M))
    return entries


# ==================== 2. 深度反投影 ====================
def read_exr_r_depths(path):
    exr = InputFile(path)
    dw = exr.header()['dataWindow']
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    ct = exr.header()['channels']['R']
    if ct.type != Imath.PixelType(Imath.PixelType.FLOAT):
        raise ValueError(f"EXR R 通道非 FLOAT: {ct.type} ({path})")
    raw = exr.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
    return np.frombuffer(raw, dtype=np.float32).reshape(H, W).copy()


def unproject_frame(depth, fx, fy, cx, cy, far=DEPTH_FAR):
    valid = (depth > 0) & (depth < far)
    if not valid.any():
        return np.zeros((0, 3), np.float32), np.array([], np.int32), np.array([], np.int32)
    vs, us = np.nonzero(valid)
    d = depth[vs, us].astype(np.float64)
    pts = np.stack([(us - cx) / fx * d, (vs - cy) / fy * d, d], axis=1)
    return pts, us, vs


# ==================== 3. 点云后处理 ====================
def voxel_downsample(pts, colors, normals, voxel_size):
    if len(pts) == 0:
        return pts, colors, normals
    keys = np.floor(pts / voxel_size).astype(np.int64)
    spans = keys.max(axis=0) - keys.min(axis=0) + 1
    strides = np.array([1, spans[0], spans[0] * spans[1]], dtype=np.int64)
    codes = ((keys - keys.min(axis=0)) * strides).sum(axis=1)
    _, first = np.unique(codes, return_index=True)
    first.sort()
    return pts[first], colors[first], (normals[first] if normals is not None else None)


def statistical_outlier_removal(pts, colors, normals, nb_neighbors=20, std_ratio=2.0):
    from scipy.spatial import cKDTree
    if len(pts) < 2 * nb_neighbors:
        return pts, colors, normals
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=min(nb_neighbors + 1, len(pts)))
    mean_d = d[:, 1:].mean(axis=1) if d.ndim == 2 else d.mean()
    thr = mean_d.mean() + std_ratio * mean_d.std()
    keep = mean_d < thr
    return pts[keep], colors[keep], (normals[keep] if normals is not None else None)


def adaptive_downsample_to_target(pts, colors, normals, target,
                                  voxel_lo=0.005, voxel_hi=0.5, tol=0.05, max_iter=30):
    if len(pts) <= target * (1 + tol):
        print(f"[INFO] 点数 {len(pts)} <= target*1.05，无需降采样")
        return pts, colors, normals
    lo, hi = voxel_lo, voxel_hi
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        p, c, n = voxel_downsample(pts, colors, normals, mid)
        cnt = len(p)
        if cnt < target * (1 - tol):
            hi = mid
        elif cnt > target * (1 + tol):
            lo = mid
        else:
            print(f"[INFO] 体素自适应: voxel={mid:.4f}m -> {cnt} 点（目标 {target}）")
            return p, c, n
        if hi - lo < 1e-5:
            break
    candidates = []
    for vs in (lo, hi, (lo + hi) / 2):
        p, c, n = voxel_downsample(pts, colors, normals, vs)
        candidates.append((abs(len(p) - target), p, c, n, vs))
    candidates.sort(key=lambda x: x[0])
    _, p, c, n, vs = candidates[0]
    print(f"[INFO] 体素自适应(收敛): voxel={vs:.4f}m -> {len(p)} 点（目标 {target}）")
    return p, c, n


def finalize_points(XYZ, RGB, NRM, target, voxel=None, do_stat=True):
    if do_stat and len(XYZ) > 1000:
        n0 = len(XYZ)
        XYZ, RGB, NRM = statistical_outlier_removal(XYZ, RGB, NRM)
        print(f"[INFO] 统计滤波: {n0} -> {len(XYZ)} 点")
    if voxel and voxel > 0:
        XYZ, RGB, NRM = voxel_downsample(XYZ, RGB, NRM, voxel)
        print(f"[INFO] 固定体素 {voxel}m -> {len(XYZ)} 点")
    elif target and target > 0 and len(XYZ) > target:
        XYZ, RGB, NRM = adaptive_downsample_to_target(XYZ, RGB, NRM, target)
    return XYZ, RGB, NRM


def write_points3d_bin(path, XYZ, RGB):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(XYZ)))
        for pid in range(len(XYZ)):
            x, y, z = XYZ[pid]
            r, g, b = int(RGB[pid][0]), int(RGB[pid][1]), int(RGB[pid][2])
            f.write(struct.pack("<QdddBBBdQ", pid + 1, x, y, z, r, g, b, 0.0, 0))


# ==================== 4. 主流程 ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="数据集根（images/depth_exr/normals）")
    # [FIX 2026-10-01] 法线来源可选：默认 normals/（旧行为，零回归）。
    # 用深度反解的几何法线（make_normals.py 产物 normals_geo/）时传 --normals_dir normals_geo，
    # 使 BBP 参考法线与训练/评估读到的 GT 法线来自同一套口径。
    ap.add_argument("--normals_dir", default="normals", type=str,
                    help="法线图目录：相对 data_dir 的子目录名或绝对路径（默认 normals）")
    ap.add_argument("--sparse", default=None,
                    help="sparse 根目录（含 sparse/0）；None = data_dir")
    ap.add_argument("--points", type=int, default=DEFAULT_POINTS)
    ap.add_argument("--per_view", type=int, default=DEFAULT_PER_VIEW)
    ap.add_argument("--voxel", type=float, default=None, help="固定体素尺寸(m)")
    ap.add_argument("--no_stat", action="store_true")
    ap.add_argument("--far", type=float, default=DEPTH_FAR)
    args = ap.parse_args()

    data_dir = args.data_dir
    sparse_root = args.sparse or data_dir
    sparse0 = os.path.join(sparse_root, "sparse", "0")
    for need in ("cameras.bin", "images.bin"):
        if not os.path.exists(os.path.join(sparse0, need)):
            print(f"[ERROR] 缺少 {sparse0}/{need} —— 先运行 export_sparse_from_blender.py")
            return 1
    print(f"[INFO] sparse 目录: {sparse0}")

    # ---- 读内参（单相机 PINHOLE）----
    cams = read_intrinsics_bin(os.path.join(sparse0, "cameras.bin"))
    cam_id, (model, W, H, params) = next(iter(cams.items()))
    if model != 1:
        print(f"[ERROR] 仅支持 PINHOLE(model=1)，当前 model={model}")
        return 1
    fx, fy, cx, cy = params
    print(f"[INFO] 内参: fx={fx:.3f} fy={fy:.3f} cx={cx:.2f} cy={cy:.2f} @ {W}x{H}")

    # ---- 位姿 ----
    entries = read_images_bin(os.path.join(sparse0, "images.bin"))
    print(f"[INFO] 位姿: {len(entries)} 帧")
    # 图像名映射：images.bin 名字 -> 索引
    name2idx = {name: i for i, (name, _) in enumerate(entries)}

    img_dir = os.path.join(data_dir, "images")
    dep_dir = os.path.join(data_dir, "depth_exr")
    nor_dir = args.normals_dir if os.path.isabs(args.normals_dir) else os.path.join(data_dir, args.normals_dir)
    print(f"[INFO] 法线目录: {nor_dir}")

    # 相机系 -> 世界系（w2c 求逆）
    all_pts, all_rgb, all_nrm = [], [], []
    for i, (name, w2c) in enumerate(entries):
        base = os.path.splitext(name)[0]
        dep_path = os.path.join(dep_dir, base + ".exr")
        img_path = os.path.join(img_dir, name)
        nor_path = os.path.join(nor_dir, name)
        if not os.path.exists(dep_path):
            print(f"[SKIP] {name}: 无 {dep_path}")
            continue
        depth = read_exr_r_depths(dep_path)
        pts_cam, us, vs = unproject_frame(depth, fx, fy, cx, cy, far=args.far)
        if len(pts_cam) == 0:
            continue
        c2w = np.linalg.inv(w2c)
        homo = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=1)
        pts_world = (c2w @ homo.T).T[:, :3]

        rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if rgb is not None:
            r_ = rgb[vs, us, 2]; g_ = rgb[vs, us, 1]; b_ = rgb[vs, us, 0]
        else:
            r_, g_, b_ = np.full(len(us), 128), np.full(len(us), 128), np.full(len(us), 128)

        nrm = None
        if os.path.exists(nor_path):
            nimg = cv2.imread(nor_path, cv2.IMREAD_COLOR)
            if nimg is not None:
                n = nimg[vs, us].astype(np.float64) / 255.0 * 2.0 - 1.0
                # [FIX 2026-10-01] 无效法线判据写反过一版（`>= 0.9` 会误杀全部真法线：
                # 真法线 |n|≈1.0，而背景 (255,255,255)/(0,0,0) → (±1,±1,±1) → |n|=√3≈1.732）。
                # 正确判据 = **上界**：只把 √3 那一类（背景/无几何）判为无效。
                # 零向量 = 下游消费者（BBP 护栏 / RVA-BAKE）的“无效”标记，点云几何保持完整。
                raw = np.linalg.norm(n, axis=1)
                valid = (raw > 0.5) & (raw <= 1.1)
                n = n / (raw[:, None] + 1e-8)
                n[~valid] = 0.0
                nrm = n

        if len(pts_world) > args.per_view:
            idx = np.random.RandomState((i + 1) * 1000).choice(len(pts_world), args.per_view, replace=False)
            idx.sort()
            pts_world, r_, g_, b_ = pts_world[idx], r_[idx], g_[idx], b_[idx]
            if nrm is not None:
                nrm = nrm[idx]

        all_pts.append(pts_world)
        all_rgb.append(np.stack([r_, g_, b_], axis=1).astype(np.uint8))
        if nrm is not None:
            all_nrm.append(nrm)
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(entries)} 视图（累计 {sum(len(p) for p in all_pts)} 点）")

    if not all_pts:
        print("[ERROR] 无任何点生成——检查 depth_exr 路径与 far 阈值")
        return 1

    XYZ = np.concatenate(all_pts, axis=0)
    RGB = np.concatenate(all_rgb, axis=0)
    NRM = np.concatenate(all_nrm, axis=0) if all_nrm else None
    print(f"[INFO] 反投影原始点数: {len(XYZ)}")

    XYZ, RGB, NRM = finalize_points(XYZ, RGB, NRM, args.points, args.voxel,
                                    do_stat=not args.no_stat)

    write_points3d_bin(os.path.join(sparse0, "points3D.bin"), XYZ, RGB)
    if NRM is not None:
        np.save(os.path.join(sparse0, "points3D_normals.npy"), NRM.astype(np.float32))
        print(f"[INFO] points3D_normals.npy: {NRM.shape}")
    else:
        print("[WARN] 未生成法线文件（normals/ 缺失）")

    print(f"[DONE] points3D.bin: {len(XYZ)} 点 -> {sparse0}")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # 纯 Python 环境可安全使用
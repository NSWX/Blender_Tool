
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 COLMAP points3D.bin 转换为带 RGB 颜色的 points3D.ply。

支持标准 COLMAP points3D.bin 格式：
    uint64: 点数量
    每个点：
        uint64: POINT3D_ID
        double: X
        double: Y
        double: Z
        uint8: R
        uint8: G
        uint8: B
        double: ERROR
        uint64: TRACK 长度
        TRACK 长度组 (int32 IMAGE_ID, int32 POINT2D_IDX)

输出：
    binary_little_endian PLY
    vertex properties: x, y, z, red, green, blue
"""

import argparse
import os
import struct
import sys

import numpy as np


def read_points3d_bin(path):
    """读取标准 COLMAP points3D.bin，返回 XYZ、RGB、ERROR。"""
    xyz_list = []
    rgb_list = []
    error_list = []

    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("文件过短，无法读取点数量。")

        num_points = struct.unpack("<Q", raw)[0]
        print(f"[INFO] BIN 点数: {num_points:,}")

        for i in range(num_points):
            # ID + XYZ + RGB + ERROR + TRACK 长度
            raw = f.read(51)
            if len(raw) != 51:
                raise ValueError(
                    f"第 {i} 个点数据不完整，文件可能损坏。"
                )

            (
                point_id,
                x, y, z,
                r, g, b,
                error,
                track_length,
            ) = struct.unpack("<QdddBBBdQ", raw)

            xyz_list.append((x, y, z))
            rgb_list.append((r, g, b))
            error_list.append(error)

            # 跳过 TRACK：每个观测由两个 int32 构成
            track_bytes = track_length * 8
            if track_bytes:
                f.seek(track_bytes, os.SEEK_CUR)

        if f.read(1):
            print("[WARN] 文件末尾存在额外数据，已忽略。")

    xyz = np.asarray(xyz_list, dtype=np.float64).reshape(-1, 3)
    rgb = np.asarray(rgb_list, dtype=np.uint8).reshape(-1, 3)
    errors = np.asarray(error_list, dtype=np.float64)

    if not np.isfinite(xyz).all():
        raise ValueError("点云坐标包含 NaN 或 Inf。")

    print(f"[INFO] XYZ 范围:\n{xyz.min(axis=0)}\n{xyz.max(axis=0)}")
    print(f"[INFO] RGB 最小值: {rgb.min(axis=0)}")
    print(f"[INFO] RGB 最大值: {rgb.max(axis=0)}")
    print(f"[INFO] 不同 RGB 颜色数: {len(np.unique(rgb, axis=0)):,}")

    return xyz, rgb, errors


def write_colored_ply(path, xyz, rgb):
    """写出带 RGB 属性的二进制 PLY。"""
    if xyz.shape != rgb.shape or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("XYZ 和 RGB 必须具有相同的 N×3 形状。")

    vertex_dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ])

    vertices = np.empty(len(xyz), dtype=vertex_dtype)
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    vertices["red"] = rgb[:, 0]
    vertices["green"] = rgb[:, 1]
    vertices["blue"] = rgb[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)

    print(f"[DONE] PLY 已保存: {path}")
    print(f"[DONE] 顶点数: {len(vertices):,}")
    print(f"[DONE] 文件大小: {os.path.getsize(path) / 1024**2:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert COLMAP points3D.bin to colored PLY."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入 points3D.bin 路径",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 PLY 路径；默认与输入文件同目录",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = args.output or os.path.join(
        os.path.dirname(input_path), "points3D1.ply"
    )
    output_path = os.path.abspath(output_path)

    if not os.path.isfile(input_path):
        print(f"[ERROR] 输入文件不存在: {input_path}")
        return 1

    if os.path.abspath(input_path) == output_path:
        print("[ERROR] 输入和输出路径不能相同。")
        return 1

    try:
        xyz, rgb, _ = read_points3d_bin(input_path)
        write_colored_ply(output_path, xyz, rgb)
    except (OSError, ValueError, struct.error) as e:
        print(f"[ERROR] 转换失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())


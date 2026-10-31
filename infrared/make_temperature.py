#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_temperature.py — 纯代码生成温度图（°C），不再需要 Blender 渲染红外

温度场与 blender-infrared-render 技能**同一套语义**（世界坐标解析函数）：
    T(p) = 分区梯度(z)                      # zones[0] 为基础，后续 zones 按 x_gt/z_gt 门控切换
         + Σ 噪声层((NoiseFac - 0.5) * amplitude)     # cloud_noise / fine_noise
         + Σ 套管热点(extra * clamp(1 - d²/r², 0, 1))  # 二次衰减（与技能 MapRange 一致）
输入：dataset 的 depth_exr/ + sparse/0/（相机位姿）+ 一次导出的 heat_sources.json（热源/地面）。

与 Blender 渲染的唯一差别：噪声层是**近似实现**（Blender 的 Noise Texture 是它自己的
分形 Perlin，无法逐位复现）。因此：
  * 要做 RVA-BAKE 的 GT 定量指标 → 用 --noise off（确定性场，可复现、可解析）
  * 要做"看起来像"的红外图 → 用 --noise approx（幅度/尺度语义一致，纹理 realization 不同）

用法:
  python make_temperature.py --dataset /path/DS --heat-sources /path/heat_sources.json \\
      --out /path/DS/scalars_code [--noise off|approx] [--exr] [--palette-png] [--limit N]

输出:
  <out>/<stem>.png        16-bit 归一化 (T-t_min)/(t_max-t_min)（喂 export_digital_twin）
  <out>/<stem>_mask.png   255=资产像素（有温度），0=地面/天空/无深度
  <out>/<stem>.exr        float32 °C（--exr）
  <out>/<stem>_ir.png     伪彩红外图（--palette-png，与技能输出同链路）
下游:
  python export_digital_twin.py ... --scalar_dir <out> \\
      --scalar_scale <t_max-t_min> --scalar_offset <t_min>
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ir_palette import apply_palette, PALETTES            # noqa: E402
import ir_geometry as G                                   # noqa: E402


# ------------------------------------------------------------------ 温度场
def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def _hash3(ix, iy, iz, seed):
    """确定性整数哈希 -> [0,1)（value noise 用）"""
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263
         + iz.astype(np.int64) * 2147483647 + int(seed) * 1013904223)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float32) / float(0x1000000)


def value_noise(p, seed=0):
    """三线性 value noise，输入 p (N,3) 世界坐标*scale"""
    i = np.floor(p).astype(np.int64)
    f = _smoothstep(p - i)
    c = {}
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                c[(dx, dy, dz)] = _hash3(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed)
    x0 = c[(0, 0, 0)] * (1 - f[:, 0]) + c[(1, 0, 0)] * f[:, 0]
    x1 = c[(0, 1, 0)] * (1 - f[:, 0]) + c[(1, 1, 0)] * f[:, 0]
    x2 = c[(0, 0, 1)] * (1 - f[:, 0]) + c[(1, 0, 1)] * f[:, 0]
    x3 = c[(0, 1, 1)] * (1 - f[:, 0]) + c[(1, 1, 1)] * f[:, 0]
    y0 = x0 * (1 - f[:, 1]) + x1 * f[:, 1]
    y1 = x2 * (1 - f[:, 1]) + x3 * f[:, 1]
    return y0 * (1 - f[:, 2]) + y1 * f[:, 2]


def fbm(p, scale, detail, seed):
    """分形叠加：octaves = max(1, round(detail))，与 Blender NoiseTexture 的 detail 语义近似"""
    oct_ = max(1, int(round(detail)))
    amp, tot, norm = 1.0, np.zeros(len(p), np.float32), 0.0
    q = p * scale
    for o in range(oct_):
        tot += amp * value_noise(q, seed + o)
        norm += amp
        amp *= 0.5
        q = q * 2.0
    return tot / max(norm, 1e-9)


def temperature_field(XYZ, hs, noise_mode="off", seed=0):
    """XYZ (H,W,3) 世界坐标 -> T (H,W) °C"""
    zones = hs.get("zones") or []
    x, y, z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    if not zones:
        T = np.full(z.shape, 25.0, np.float32)
    else:
        b = zones[0]
        r = np.clip((z - float(b["z_from"])) / max(float(b["z_to"]) - float(b["z_from"]), 1e-9), 0.0, 1.0)
        T = float(b["t_from"]) + (float(b["t_to"]) - float(b["t_from"])) * r
        for zn in zones[1:]:
            r2 = np.clip((z - float(zn["z_from"])) / max(float(zn["z_to"]) - float(zn["z_from"]), 1e-9), 0.0, 1.0)
            g = float(zn["t_from"]) + (float(zn["t_to"]) - float(zn["t_from"])) * r2
            thr = float(zn.get("x_gt", zn.get("z_gt", 0.0)))
            axis = x if "x_gt" in zn else z
            T = np.where(axis > thr, g, T)

    if noise_mode == "approx":
        flat = XYZ.reshape(-1, 3)
        for key in ("cloud_noise", "fine_noise"):
            nc = hs.get(key)
            if not nc:
                continue
            T = T + (fbm(flat, float(nc.get("scale", 5.0)), float(nc.get("detail", 2.0)), seed)
                     - 0.5).reshape(T.shape) * float(nc.get("amplitude", 1.0))

    for (hx, hy, hz, r, extra) in hs.get("hotspots", []):
        d2 = (x - hx) ** 2 + (y - hy) ** 2 + (z - hz) ** 2
        T = T + float(extra) * np.clip(1.0 - d2 / max(r * r, 1e-9), 0.0, 1.0)
    return T.astype(np.float32)


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser(description="代码侧生成温度图（替代 Blender IR 渲染）")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--heat-sources", required=True)
    ap.add_argument("--depth-sub", default="depth_exr")
    ap.add_argument("--out", default="scalars_code")
    ap.add_argument("--noise", default="off", choices=["off", "approx"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ground-tol", type=float, default=0.35)
    ap.add_argument("--exr", action="store_true")
    ap.add_argument("--palette-png", action="store_true", help="同时输出伪彩 IR 图（可与技能渲染对比）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ds = os.path.abspath(args.dataset)
    with open(args.heat_sources, encoding="utf-8") as f:
        hs = json.load(f)
    t_min, t_max = float(hs.get("t_min", 20.0)), float(hs.get("t_max", 50.0))
    palette = hs.get("palette", "IRON")
    out = os.path.join(ds, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(out, exist_ok=True)

    K, stems = G.intrinsics_poses(ds)
    frames = G.list_frames(os.path.join(ds, args.depth_sub))
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        print(f"[FATAL] {ds}/{args.depth_sub} 无 exr"); return 1
    print(f"[make_temperature] {len(frames)} 帧  噪声={args.noise}  palette={palette} "
          f"量程 {t_min}–{t_max} °C\n  热源 {len(hs.get('hotspots', []))} 个, "
          f"地面={'有' if hs.get('ground') else '无'}  -> {out}")

    t0 = time.time()
    done = 0
    asset_frac, tmed = [], []
    for exr in frames:
        stem = os.path.splitext(os.path.basename(exr))[0]
        if os.path.exists(os.path.join(out, f"{stem}.png")) and not args.overwrite:
            continue
        ent = stems.get(stem)
        if ent is None:
            continue
        Z = G.read_depth(exr)
        vd = G.valid_depth(Z)
        XYZ = G.unproject_world(np.where(vd, Z, 1.0).astype(np.float32), K, ent)
        T = temperature_field(XYZ, hs, noise_mode=args.noise, seed=args.seed)
        gnd = G.ground_mask(XYZ, hs.get("ground"), tol=args.ground_tol)
        asset = vd & (~gnd)                       # 资产像素：有温度、非地面、非天空

        norm = np.clip((T - t_min) / max(t_max - t_min, 1e-9), 0.0, 1.0)
        import cv2
        cv2.imwrite(os.path.join(out, f"{stem}.png"), np.round(norm * 65535.0).astype(np.uint16))
        cv2.imwrite(os.path.join(out, f"{stem}_mask.png"), (asset * 255).astype(np.uint8))
        if args.exr:
            import OpenEXR
            with OpenEXR.File({}, {"T": np.where(asset, T, np.nan).astype(np.float32)}) as fo:
                fo.write(os.path.join(out, f"{stem}.exr"))
        if args.palette_png:
            ir = apply_palette(np.where(vd, T, t_min), palette=palette, t_min=t_min, t_max=t_max)
            ir[~vd] = 0
            cv2.imwrite(os.path.join(out, f"{stem}_ir.png"), ir[:, :, ::-1])
        if asset.any():
            asset_frac.append(float(asset.mean())); tmed.append(float(np.median(T[asset])))
        done += 1
        if done % 50 == 0:
            print(f"  {done} 帧 … {time.time()-t0:.0f}s", flush=True)

    print(f"[make_temperature] 完成 {done} 帧（{time.time()-t0:.0f}s）")
    if asset_frac:
        print(f"  资产像素占比 中位 {np.median(asset_frac)*100:.1f}%（其余=地面/天空/无深度，已写入 *_mask.png）")
        print(f"  资产像素温度 中位 {np.median(tmed):.2f} °C")
    print(f"  下游: --scalar_dir {out} --scalar_scale {t_max-t_min:.1f} --scalar_offset {t_min:.1f}"
          f"   （若用 EXR 则直接是 °C，scale/offset 传 1/0）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

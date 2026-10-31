#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
palette_invert.py — 由已渲染的伪彩红外图反演温度图（°C）

适用：数据集已经用 blender-infrared-render 技能渲染过 infrared/*.png，不想重渲。
原理：ir_palette.py 的精确前向模型（线性空间插值 + Standard 视图变换）反查最近点。

重要：伪彩图里**不是所有像素都有温度**——地面物体与天空在技能里被赋予常色块
（M_IR_GroundFlat / __tmp_ir_world），它们会反演出一个假温度。本工具用"到色板折线的
残差"把它们识别出来，输出 mask_target.png（255=温度有效），并在统计里把它们剔除。

用法:
  python palette_invert.py --src /path/DS/infrared --config /path/DS/ir_config_SC11.json \\
      --out /path/DS/scalars_ir [--exr] [--limit N]
  # 下游 RVA-BAKE:
  python export_digital_twin.py -m <run> -s /path/DS --scalar_dir /path/DS/scalars_ir \\
      --scalar_scale 30.0 --scalar_offset 20.0    # 值域 = [t_min, t_max]
"""
import os
import sys
import glob
import time
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ir_palette import invert_rgb, PALETTES   # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None


def load_config(path):
    cfg = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    return {
        "palette": cfg.get("palette", "IRON"),
        "t_min": float(cfg.get("t_min", 20.0)),
        "t_max": float(cfg.get("t_max", 50.0)),
    }


def main():
    ap = argparse.ArgumentParser(description="伪彩红外图 -> 温度图 (°C)")
    ap.add_argument("--src", required=True, help="伪彩 PNG 目录（如 infrared/）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="ir_config*.json（读 palette/t_min/t_max）")
    ap.add_argument("--palette", default=None, choices=list(PALETTES.keys()))
    ap.add_argument("--t-min", type=float, default=None)
    ap.add_argument("--t-max", type=float, default=None)
    ap.add_argument("--residual-tol", type=float, default=0.04,
                    help="到色板折线的最大色差（0-1）才算温度有效像素")
    ap.add_argument("--dataset", default=None,
                    help="数据集根目录（有 sparse/0 与 depth_exr 时启用**几何掩码**：地面/天空剔除）")
    ap.add_argument("--heat-sources", default=None, help="heat_sources.json（提供地面 bbox）")
    ap.add_argument("--ground-tol", type=float, default=0.35)
    ap.add_argument("--exr", action="store_true", help="额外输出 float32 °C EXR")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    c = load_config(args.config)
    palette = args.palette or c["palette"]
    t_min = args.t_min if args.t_min is not None else c["t_min"]
    t_max = args.t_max if args.t_max is not None else c["t_max"]
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.src, "*.png")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"[FATAL] {args.src} 下没有 PNG"); return 1
    print(f"[palette_invert] {len(files)} 帧  palette={palette} 量程 {t_min}–{t_max} °C  -> {args.out}")

    # 可选：几何掩码（地面/天空剔除）
    geo = None
    if args.dataset and args.heat_sources and os.path.exists(args.heat_sources):
        import ir_geometry as G
        with open(args.heat_sources, encoding="utf-8") as f:
            hs = json.load(f)
        geo = (G, G.intrinsics_poses(os.path.abspath(args.dataset))[0],
               G.intrinsics_poses(os.path.abspath(args.dataset))[1], hs.get("ground"))
        print(f"  几何掩码: 启用（地面物体 {geo[3]['name'] if geo[3] else 'None'}，"
              f"depth_exr 与伪彩帧需同名同序）")
    else:
        print("  几何掩码: 未启用（仅用色板残差判定；地面/天空可能被误判为温度）")

    t0 = time.time()
    done = 0
    cov, tmed, tmax_seen = [], [], []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        dst = os.path.join(args.out, f"{stem}.png")
        if os.path.exists(dst) and not args.overwrite:
            continue
        bgr = cv2.imread(f, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = bgr[:, :, ::-1]
        T, R = invert_rgb(rgb, palette=palette, t_min=t_min, t_max=t_max)
        ok = R <= args.residual_tol
        if geo is not None:
            Gmod, K, stems, gbbox = geo
            ent = stems.get(stem)
            exr = os.path.join(os.path.abspath(args.dataset), "depth_exr", f"{stem}.exr")
            if ent is not None and os.path.exists(exr):
                Z = Gmod.read_depth(exr)
                vd = Gmod.valid_depth(Z)
                XYZ = Gmod.unproject_world(np.where(vd, Z, 1.0).astype(np.float32), K, ent)
                gnd = Gmod.ground_mask(XYZ, gbbox, tol=args.ground_tol)
                ok = ok & vd & (~gnd)
        # 16-bit 归一化 PNG：值 = (T - t_min)/(t_max - t_min) -> 下游用 --scalar_scale/--scalar_offset 还原
        norm = np.clip((T - t_min) / max(t_max - t_min, 1e-9), 0.0, 1.0)
        cv2.imwrite(dst, np.round(norm * 65535.0).astype(np.uint16))
        m = os.path.join(args.out, "mask_target.png")
        cv2.imwrite(os.path.join(args.out, f"{stem}_mask.png"), (ok * 255).astype(np.uint8))
        if args.exr:
            import OpenEXR
            with OpenEXR.File({}, {"T": np.where(ok, T, np.nan).astype(np.float32)}) as fo:
                fo.write(os.path.join(args.out, f"{stem}.exr"))
        if ok.any():
            cov.append(float(ok.mean())); tmed.append(float(np.median(T[ok]))); tmax_seen.append(float(T[ok].max()))
        done += 1
        if done % 50 == 0:
            print(f"  {done} 帧 … {time.time()-t0:.0f}s", flush=True)
    print(f"[palette_invert] 完成 {done} 帧（用时 {time.time()-t0:.0f}s）")
    if cov:
        print(f"  温度有效像素占比 中位 {np.median(cov)*100:.1f}%  "
              f"（其余为地面/天空常色块，已用 *_mask.png 标出）")
        print(f"  有效像素温度 中位 {np.median(tmed):.2f} °C，逐帧最高中位 {np.median(tmax_seen):.2f} °C")
    print(f"  下游: --scalar_dir {args.out} --scalar_scale {t_max-t_min:.1f} --scalar_offset {t_min:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

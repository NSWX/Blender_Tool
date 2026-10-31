#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ir_palette.py — 红外伪彩调色板的**精确前向/反向**模型（与 blender-infrared-render 技能同一套约定）

关键物理/色彩链路（blender_ir.py + 技能准则 4）:
  1. 停机点颜色以 sRGB 0-255 给出，写入 ColorRamp 前做 s2l()（sRGB→线性）；
  2. ColorRamp 在线性空间做线性插值（color_mode='RGB'）；
  3. Emission(strength=1) 输出该线性色，view_transform='Standard' 再编码回 sRGB。
  => 前向 = l2s( lerp_linear(s2l(c0), s2l(c1), t) )
     **不是**在 sRGB 空间直接插值——后者会带来 0.1~0.5°C 的系统偏差。

导出接口:
  forward_curve(...)      -> 折线 (M,3) sRGB + 对应归一化温度 u
  invert_rgb(rgb_u8, ...) -> (T 数组, 残差数组 0-1)；残差大 => 该像素不在色板上（地面/天空等常色块）
  apply_palette(T, ...)   -> 由温度生成伪彩 RGB（用于代码侧画 IR 图）
"""
import numpy as np

PALETTES = {
    "IRON": [(0.00, (0, 0, 0)), (0.12, (13, 3, 26)), (0.25, (51, 5, 31)),
             (0.38, (115, 13, 15)), (0.52, (191, 38, 10)), (0.66, (242, 102, 15)),
             (0.78, (255, 166, 38)), (0.88, (255, 224, 115)), (1.00, (255, 255, 255))],
    "LAVA": [(0.00, (0, 0, 0)), (0.25, (60, 0, 0)), (0.50, (160, 20, 0)),
             (0.70, (235, 90, 0)), (0.85, (255, 180, 60)), (1.00, (255, 255, 220))],
    "WHITEHOT": [(0.00, (0, 0, 0)), (0.30, (70, 70, 70)), (0.60, (160, 160, 160)),
                 (0.85, (230, 230, 230)), (1.00, (255, 255, 255))],
    "RAINBOW": [(0.00, (20, 0, 60)), (0.20, (60, 0, 140)), (0.40, (0, 120, 120)),
                (0.60, (120, 200, 0)), (0.80, (255, 160, 0)), (1.00, (255, 40, 40))],
}


def s2l(v):
    c = max(0.0, min(1.0, float(v) / 255.0))
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def l2s(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def forward_curve(palette="IRON", n_per_seg=24):
    """返回 (pts (M,3) sRGB 0-1, u (M,), seg_id (M,))：精确管线下的色板折线"""
    stops = PALETTES.get(palette, PALETTES["IRON"]) if isinstance(palette, str) else palette
    us = np.array([p for p, _ in stops], float)
    lin = np.array([[s2l(c[0]), s2l(c[1]), s2l(c[2])] for _, c in stops], float)
    pts, uu, seg = [], [], []
    for i in range(len(stops) - 1):
        for k in range(n_per_seg):
            t = k / n_per_seg
            u = us[i] + t * (us[i + 1] - us[i])
            c = lin[i] * (1 - t) + lin[i + 1] * t
            pts.append(l2s(c)); uu.append(u); seg.append(i)
    pts.append(l2s(lin[-1])); uu.append(us[-1]); seg.append(len(stops) - 2)
    return np.array(pts, np.float32), np.array(uu, np.float32), np.array(seg, np.int32)


def _chords(stops):
    us = np.array([p for p, _ in stops], float)
    lin = np.array([[s2l(c[0]), s2l(c[1]), s2l(c[2])] for _, c in stops], float)
    srgb = np.stack([l2s(lin)], 0)[0]
    return us, srgb


def invert_rgb(rgb_u8, palette="IRON", t_min=20.0, t_max=50.0, chunk=400_000):
    """rgb_u8 (H,W,3) uint8 -> (T (H,W) float32 °C, residual (H,W) float32 0-1)

    residual = 像素到色板折线的最近距离（0-1 归一化色差）；> ~0.04 说明该像素不在色板上
    （典型：地面/天空被渲染成常色块，或叠加了 OSD 标注）。
    """
    stops = PALETTES.get(palette, PALETTES["IRON"]) if isinstance(palette, str) else palette
    us_stop, srgb_stop = _chords(stops)
    pts, uu, seg = forward_curve(stops)
    H, W = rgb_u8.shape[:2]
    flat = rgb_u8.reshape(-1, 3).astype(np.float32) / 255.0
    T = np.zeros(len(flat), np.float32)
    R = np.zeros(len(flat), np.float32)

    # 阶段 1：对 8 条弦做投影，选出候选分段（含左右邻段）
    for s in range(0, len(flat), chunk):
        x = flat[s:s + chunk]
        n_px = len(x)
        best_d = np.full(n_px, 1e9, np.float32)
        best_seg = np.zeros(n_px, np.int32)
        for i in range(len(us_stop) - 1):
            a, b = srgb_stop[i], srgb_stop[i + 1]
            d = b - a
            l2 = float((d * d).sum())
            t = np.clip(((x - a) @ d) / max(l2, 1e-9), 0.0, 1.0)
            dist = np.linalg.norm(x - (a + t[:, None] * d), axis=1)
            upd = dist < best_d
            best_d = np.where(upd, dist, best_d)
            best_seg = np.where(upd, i, best_seg)
        # 阶段 2：在候选段（及其邻段）的折线点上取最近点（逐段累加最小值，避免覆盖）
        bd = np.full(n_px, 1e9, np.float32)
        bu = np.zeros(n_px, np.float32)
        for i in range(len(us_stop) - 1):
            sel = np.where(np.abs(best_seg - i) <= 1)[0]
            if len(sel) == 0:
                continue
            p = pts[seg == i]                       # (m,3)
            idxs = uu[seg == i]                     # (m,)
            xx = x[sel]                             # (k,3)
            d2 = ((xx[:, None, :] - p[None, :, :]) ** 2).sum(-1)     # (k,m)
            j = d2.argmin(1)
            dist = np.sqrt(d2[np.arange(len(sel)), j])
            better = dist < bd[sel]
            bd[sel[better]] = dist[better]
            bu[sel[better]] = idxs[j][better]
        T[s:s + chunk] = t_min + (t_max - t_min) * bu
        R[s:s + chunk] = bd
    return T.reshape(H, W), R.reshape(H, W)


def apply_palette(T, palette="IRON", t_min=20.0, t_max=50.0):
    """温度(°C) -> 伪彩 RGB uint8（与 Blender 渲染同链路，可直接与技能输出对比）"""
    stops = PALETTES.get(palette, PALETTES["IRON"]) if isinstance(palette, str) else palette
    us = np.array([p for p, _ in stops], float)
    lin = np.array([[s2l(c[0]), s2l(c[1]), s2l(c[2])] for _, c in stops], float)
    u = np.clip((np.asarray(T, np.float32) - t_min) / max(t_max - t_min, 1e-9), 0.0, 1.0)
    out = np.zeros((*u.shape, 3), np.float32)
    for i in range(len(us) - 1):
        m = (u >= us[i]) & (u <= us[i + 1] if i == len(us) - 2 else u < us[i + 1])
        if not m.any():
            continue
        t = (u[m] - us[i]) / max(us[i + 1] - us[i], 1e-9)
        out[m] = lin[i] * (1 - t)[:, None] + lin[i + 1] * t[:, None]
    return np.clip(np.round(l2s(out) * 255.0), 0, 255).astype(np.uint8)

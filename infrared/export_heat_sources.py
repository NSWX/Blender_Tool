#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_heat_sources.py — 从 .blend 导出温度场所需的"热源"参数（Blender 侧，只读，不渲染）

温度场在 blender-infrared-render 技能里由材质节点定义，但它是**世界坐标的解析函数**：
    T(p) = 分区梯度(z) + 噪声(p) + Σ 套管热点(||p - c_i||)
因此只要把「套管热点中心/半径/温升」和「地面物体几何」一次性导出成 JSON，
温度图就能完全由代码从 depth_exr + 位姿算出，无需每次渲染。

导出内容:
  hotspots     : [(x, y, z, radius, extra_T), ...]  与技能 detect_bushings() 完全同算法
  ground       : 地面物体 bbox（代码侧据此把地面像素从资产温度里剔除）
  zones/noise  : 从 ir_config 复制，保证代码侧与渲染侧同参

用法（三种）:
  1) 命令行:  blender -b scene.blend --python export_heat_sources.py -- \\
                  --config ir_config.json --out heat_sources.json
  2) Blender 文本编辑器: 设好 CONFIG/OUT_PATH 全局后运行
  3) MCP execute_blender_code:
        cfg = json.load(open('<ir_config.json>')); 
        g = {'__name__':'ir_export','CONFIG':cfg,'OUT_PATH':'<out.json>'}
        exec(compile(open('<本文件>').read(),'export_heat_sources.py','exec'), g)
"""
import json
import os
import sys

try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None


def detect_bushings(cfg):
    """与 blender_ir.py 的 detect_bushings() 同算法：按 bbox 高细比筛出套管，按顶端高度排名。"""
    col = bpy.data.collections.get(cfg.get("target_collection", ""))
    if not col:
        return []
    b_cfg = cfg.get("bushing", {})
    min_h = b_cfg.get("min_height", 3.5)
    min_r = b_cfg.get("min_ratio", 3.8)
    z_min = b_cfg.get("z_min", 18.0)

    cands = []
    for obj in col.all_objects:
        if obj.type != "MESH":
            continue
        bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        mn = Vector((min(c[i] for c in bb) for i in range(3)))
        mx = Vector((max(c[i] for c in bb) for i in range(3)))
        sz = mx - mn
        base = max(sz.x, sz.y)
        if sz.z > min_h and base > 0 and (sz.z / base) > min_r and mx.z > z_min:
            cands.append(((mn + mx) / 2.0, mx.z))

    uniq = {}
    for c, top in cands:
        key = (round(c.x, 1), round(c.y, 1))
        if key not in uniq or top > uniq[key][1]:
            uniq[key] = (c, top)

    tops = sorted(((c.x, c.y, z) for (c, z) in uniq.values()), key=lambda t: -t[2])
    rules = b_cfg.get("hotspots", [
        {"rank": 0, "extra": 40.0, "radius": 1.4},
        {"rank": 1, "extra": 20.0, "radius": 1.4},
        {"rank": "other", "extra": 8.0, "radius": 1.0},
    ])
    out = []
    for i, (x, y, z) in enumerate(tops):
        rule = next((r for r in rules if r.get("rank") == i),
                    next((r for r in rules if r.get("rank") == "other"),
                         {"extra": 8.0, "radius": 1.0}))
        out.append([float(x), float(y), float(z - 0.5), float(rule["radius"]), float(rule["extra"])])
    # 未参与排名规则的套管（rank:'other' 之外）也一并记录，便于代码侧复核
    return out


def ground_bbox(cfg):
    g = bpy.data.objects.get(cfg.get("ground_object", "")) if cfg.get("ground_object") else None
    if g is None or g.type != "MESH":
        return None
    bb = [g.matrix_world @ Vector(c) for c in g.bound_box]
    mn = [min(c[i] for c in bb) for i in range(3)]
    mx = [max(c[i] for c in bb) for i in range(3)]
    return {"name": g.name, "bbox_min": [float(v) for v in mn], "bbox_max": [float(v) for v in mx]}


def camera_info(cfg):
    name = cfg.get("camera_object")
    cam = bpy.data.objects.get(name) if name else bpy.context.scene.camera
    if cam is None:
        return None
    return {"name": cam.name, "type": cam.data.type,
            "lens": float(cam.data.lens), "sensor_width": float(cam.data.sensor_width)}


def build_payload(cfg):
    hotspots = detect_bushings(cfg) if cfg.get("detect_bushings", True) else []
    return {
        "target_collection": cfg.get("target_collection"),
        "ground_object": cfg.get("ground_object"),
        "camera_object": cfg.get("camera_object"),
        "camera": camera_info(cfg),
        "hotspots": hotspots,
        "ground": ground_bbox(cfg),
        "zones": cfg.get("zones", []),
        "cloud_noise": cfg.get("cloud_noise"),
        "fine_noise": cfg.get("fine_noise"),
        "palette": cfg.get("palette", "IRON"),
        "t_min": cfg.get("t_min", 20.0),
        "t_max": cfg.get("t_max", 50.0),
    }


def _run(cfg, out_path):
    payload = build_payload(cfg)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"HEAT_SOURCES_EXPORTED: {out_path}")
    print(f"  hotspots={len(payload['hotspots'])} ground={bool(payload['ground'])} "
          f"zones={len(payload['zones'])}")
    for i, h in enumerate(payload["hotspots"]):
        print(f"  Sp{i}: xyz=({h[0]:.3f},{h[1]:.3f},{h[2]:.3f}) r={h[3]:.2f} +{h[4]:.1f}°C")


if __name__ == "__main__":
    # --- 命令行模式: blender -b scene.blend --python export_heat_sources.py -- --config A --out B
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    with open(a.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.config)), "heat_sources.json")
    _run(cfg, out)
elif "CONFIG" in globals():
    # --- MCP / 文本编辑器模式: 外部注入 CONFIG 与（可选）OUT_PATH
    _out = globals().get("OUT_PATH") or os.path.join(
        globals().get("CONFIG", {}).get("output_dir", "."), "heat_sources.json")
    _run(CONFIG, _out)

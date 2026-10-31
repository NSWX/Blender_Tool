# 代码侧生成 GT 法线（不再用 Blender 渲染 normal pass）

**结论：Blender 只需要渲染 `images/`（可见光）与 `depth_exr/`（32-bit float、米）+ 导出 `sparse/0/` 位姿。`normals/` 一律由本目录的脚本从深度+位姿反解。**

> **两处副本必须同步**：本目录是**权威手改点**，技能 `blender-normal-maps` 的 `scripts/make_normals.py` 是同一份副本（让技能自包含、可随技能目录迁移）。改完本目录后执行：
> `cp make_normals.py ~/.hermes/skills/creative/blender-normal-maps/scripts/`

## 为什么不再渲染法线（三条实证过的失败模式）

| # | 失败模式 | 证据 |
|---|---|---|
| 1 | **合成器 File Output 槽位默认 `save_as_render=True`**，法线数据被场景色彩管理（`view_transform` / `look` / `exposure`）色调映射，**不可逆** | SC11 场景实测 `Khronos PBR Neutral + Very High Contrast + exposure=-0.15`；法线 R 通道 n=0 被写成交 89/255（线性应为 127.5），同一几何法线下颜色 std 达 20–97/255。32-bit float EXR 天然免疫——所以 `depth_exr` 一直是好的、只有法线坏掉 |
| 2 | **Cycles 的 Normal pass 在 Blender 世界系**，而训练侧 `rend_normal` 在 COLMAP/位姿世界系 | Kabsch 审计：SC11 差 **21.4°**、Transforme01 差 **61.4°** 的固定旋转；模型法线哪怕完美也会被量出 ~60° 误差（论文里 EDGE 的 52.84° 主要就是这个） |
| 3 | **编码/符号约定跨导出易丢失**，而读取器固定按 `(c/255)*2-1` 解码 | `scene/dataset_readers.py` |

反解法的额外好处：法线是**几何真值**（不掺材质/bump/色彩管理），且 `depth_exr` 已在数据集里，**零额外渲染成本**。

## 用法

```bash
cd /home/yyc/Documents/Tools/Blender/normals
eval "$(micromamba shell hook --shell bash)" && micromamba activate mesh_splatting
export MESHSPLAT_ROOT=/home/yyc/Documents/Ones/mesh-splatting     # mesh-splatting 仓库根

# 生成（默认写 <dataset>/normals_geo/）
env -u PYTHONPATH python make_normals.py --dataset /path/to/DS --out normals_geo

# 自检：与已有法线目录做三级坐标系审计（全局旋转 / 逐视图旋转 / 反射）
env -u PYTHONPATH python make_normals.py --dataset /path/to/DS --verify normals
```

训练与评估时**无需改代码**，用已存在的开关覆盖读取器的 `normals/` 约定：

```bash
python train.py           -s /path/to/DS ... --mono_normal_dir /path/to/DS/normals_geo
python eval_normal_error.py -m <run> --split test --n_views 40 --scaling 2 \
                            --mono_normal_dir /path/to/DS/normals_geo
```

> 注意：`mono_normal_dir` 默认值是 `""` 而不是 `None`，而 `get_combined_args` 里 **CLI 非 None 值优先于 cfg_args**。
> 所以评估一个训练时带过 `--mono_normal_dir` 的 run 时，必须显式传值，否则会被清空回 `normals/`。

## 输出契约（改之前先读 `scene/dataset_readers.py`）

- 8-bit RGB PNG，文件名 = COLMAP image name 的 stem（`0001.png`），与 `images/` 同名同序、同分辨率
- 值编码 `c = round((n + 1) / 2 × 255)`，`n` 为**相机外参所在世界系**的单位法线
- 深度哨兵像素（`Z ≥ 1e4`）写黑 `(0,0,0)` → 解码为 `(-1,-1,-1)`（模长 √3），靠 `depth_mask` 屏蔽
- 符号：`n = -cross(du, dv)`（默认 `--sign neg`，是本项目既有 GT 的约定；用 `--verify` 的"同向/反向"确认）
- 断崖像素（3×3 深度跨度 > `0.03·|Z| + 0.05`）用邻域最平坦像素的法线替换（避免有限差分跨面污染）

## 验证记录（2026-09，SC11 / Transforme01 / BDZ02_v2 / BDZ01_v2）

- 生成耗时 ≈ 0.4 s/帧（SC11 200 帧 87 MB、T01 150 帧、BDZ02_v2 800 帧、BDZ01_v2 500 帧）
- 解码往返：有效区 `|n|` 中位 **1.0000**（p1 0.9963 / p99 1.0041）
- 与项目读取器实跑一致：`normal_map` 值域 `[-1,1]`，前景 `|n|≈1` 占比与有效深度占比吻合、背景黑占比吻合
- 与"曾经验证正确"的历史导出（旧 SC11 normals）比对到 **0.02°**
- 与 Blender 渲染 pass 的关系：Transforme01 差 **0.70°**（仅差 61.4° 坐标系旋转，数据完好）、SC11 差 **3.4°**（旋转 21.4° + 色调映射残差）；BDZ01_v2 / BDZ02_v2 的原法线**三层审计全部失败**（18.0° / 25.2°），即原数据已被破坏
- 修正 GT 后 BDZ02_v2 的 EDGE 评估：baseline median 51.94° → **23.60°**，相对增益不变（mean −8.4% / cos −14.5% / p95 最大改善）

## 依赖

`numpy`、`opencv-python`、`OpenEXR`、`Pillow`，以及 mesh-splatting 仓库的 `scene.colmap_loader`（通过 `MESHSPLAT_ROOT` 指向）。

## 常见坑

| 坑 | 说明 |
|---|---|
| 16-bit PNG 被 PIL 读成 8-bit | PIL 读 16-bit RGBA 只给高字节。本工具默认写 8-bit，与读取器口径一致 |
| `np.gradient(Z, 0)` | 第二个位置参数是 **spacing** 不是 axis，会除零；必须写 `np.gradient(Z, axis=0)` |
| `.reshape(Z.shape)` | 相机→世界变换后是 `(N,3)`，必须 `reshape((*Z.shape, 3))`，否则 size 不匹配 |
| 评估脚本静默回退 train | `eval_normal_error.py` 在 test 为空时回退 train 却仍打印 `split=test`；先 `grep eval= <run>/cfg_args` 确认该 run 真有 test split |
| 相机模型 | 必须分支处理 PINHOLE(4) / SIMPLE_PINHOLE(3, fx=fy=f) / OPENCV(4)；本工具已内置 |
| 位姿帧漂移 | 若数据集在训练后被重新导出（如 DAI 重生 `sparse/0`），GT 与模型世界系会错配；用 `cameras.json` 与 `sparse/0` 的**相机中心**做一致性检查（旋转矩阵比对约定易错，别用旋转残差下结论） |

## 文件

| 文件 | 作用 |
|---|---|
| `make_normals.py` | 主工具：生成 / 三级坐标系审计 |
| `README.md` | 本文件 |

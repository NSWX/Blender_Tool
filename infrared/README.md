# 代码侧生成温度图（可选替代 Blender 红外渲染）+ 伪彩图反演

**结论：温度图完全可以由代码从 `depth_exr` + 位姿 + 一次性导出的热源 JSON 生成，不需要每次跑 Blender 渲染。**
伪彩红外图（给人看的）也能用代码画出来；但**下游定量用途请用温度图 + 资产掩码**，不要用伪彩图。

> **两处副本必须同步**：本目录是**权威手改点**，技能 `blender-infrared-render` 的 `scripts/` 是同一份副本（让技能自包含、可随技能目录迁移）。改完本目录后执行：
> `cp ir_palette.py ir_geometry.py make_temperature.py palette_invert.py export_heat_sources.py ~/.hermes/skills/creative/blender-infrared-render/scripts/`

## ⚠️ 第一件必须知道的事：伪彩图里大部分像素没有温度

`blender-infrared-render` 技能把 **地面物体** 换成常色自发光材质（`M_IR_GroundFlat`）、**世界背景**也是常色（`__tmp_ir_world`）——
这两块在渲染图里是**固定颜色块，不含任何温度信息**，但它们**恰好落在 IRON 色板暗端附近**，用色板反演会得到一个"看起来合理"的假温度。

SC11 实测（frame 0001）：

| 区域 | 精确颜色 | 像素占比 |
|---|---|---|
| 天空/背景 | RGB(16,7,26) / (15,6,25) / (16,7,25) | **≈ 51%** |
| 地面 | RGB(33,22,48) / (33,22,49) | **≈ 15.5%** |
| **资产（真正有温度）** | 沿色板的连续色 | **≈ 20%**（其余为无深度/边缘） |

所以：**任何温度统计/烘焙都必须先套 `*_mask.png`**。加上掩码后，SC11 的资产温度中位是 **25.23 °C**（不套掩码会得到 23.92 °C 的偏值）。

## 两条路线

### 路线 A（推荐，未来数据集）：代码侧直接生成温度

```bash
cd /home/yyc/Documents/Tools/Blender/infrared
eval "$(micromamba shell hook --shell bash)" && micromamba activate mesh_splatting
export MESHSPLAT_ROOT=/home/yyc/Documents/Ones/mesh-splatting

# ① 一次性从 .blend 导出热源（套管热点 + 地面 bbox + 分区/噪声参数）——只读，不改场景
blender -b SC11.blend --python export_heat_sources.py -- \
    --config /path/DS/ir_config_SC11.json --out /path/DS/heat_sources.json

# ② 每帧温度图（写 <dataset>/scalars_code/）
env -u PYTHONPATH python make_temperature.py --dataset /path/DS \
    --heat-sources /path/DS/heat_sources.json --out scalars_code \
    --noise off --exr --palette-png
```

温度场与技能**同一套语义**（世界坐标解析函数）：

```
T(p) = 分区梯度(z)                                  # zones[0] 为基础，后续 zones 按 x_gt/z_gt 门控切换
     + Σ (NoiseTexture.Fac - 0.5) × amplitude        # cloud_noise / fine_noise
     + Σ 热点 extra × clamp(1 - d²/r², 0, 1)          # 套管热点，二次衰减（与技能 MapRange 完全一致）
```

> **噪声说明**：Blender 的 Noise Texture 是它自己的分形 Perlin，无法逐位复现。因此
> `--noise off`（默认）= 确定性解析场（**推荐给 RVA-BAKE 定量指标**，可复现、可解析）；
> `--noise approx` = 幅度/尺度语义一致的 numpy value-noise，纹理 realization 不同（给"看起来像"的图用）。

### 路线 B：已渲染过的数据集，直接反演伪彩图

```bash
env -u PYTHONPATH python palette_invert.py --src /path/DS/infrared \
    --config /path/DS/ir_config_SC11.json \
    --dataset /path/DS --heat-sources /path/DS/heat_sources.json \
    --out /path/DS/scalars_ir --exr
```
`ir_palette.py` 用的是**精确前向模型**：停机点颜色先做 sRGB→线性（`s2l`），ColorRamp 在线性空间插值，再由
`view_transform='Standard'` 编码回 sRGB。**不要在 sRGB 空间直接插值色板**——那会带来 0.1–0.5 °C 的系统偏差。

## 验证记录（SC11，frame 0001–0003，资产像素交集）

| 指标 | 结果 |
|---|---|
| 资产掩码占比 | 代码 **20.9%** vs 反演 **20.8%**（两种独立方法一致度 **99.5%**） |
| \|ΔT\| 中位 | **0.146 °C**（≈ 8-bit 色板量化极限） |
| \|ΔT\| p90 / max | 0.371 °C / 6.5–7.4 °C（仅出现在边缘抗锯齿像素） |
| 温度中位 | 代码 25.21 °C vs 反演 25.22 °C |
| 套管热点 | 15 个（rank0 +28 °C、rank1 +16 °C、其余 +9 °C），最高温 50 °C（撞 `t_max` 上限） |

结论：代码侧**完全复刻**技能的温场；SC11 的渲染实质是解析场（噪声未生效）。

## 输出与下游用法

| 文件 | 内容 |
|---|---|
| `<stem>.png` | 16-bit 归一化 `(T - t_min) / (t_max - t_min)`，直接喂 `export_digital_twin.py --scalar_dir` |
| `<stem>_mask.png` | 255 = 资产像素（有温度）；0 = 地面/天空/无深度 |
| `<stem>.exr` | float32 °C（无效像素为 NaN）—— 需要精细定量时用它 |
| `<stem>_ir.png` | 伪彩红外图（`--palette-png`，与技能渲染同链路，可用于对比/出图） |

```bash
# 下游：数字孪生标量烘焙（--scalar_mask_dir 必传：地面/天空假温度不得进入融合）
python export_digital_twin.py -m <run> -s /path/DS --data_device cpu \
    --scalar_dir /path/DS/scalars_ir --scalar_scale 30.0 --scalar_offset 20.0 \
    --scalar_mask_dir /path/DS/scalars_ir            # 自动识别 <stem>_mask.png
# 若用 EXR（已是 °C）：--scalar_scale 1.0 --scalar_offset 0.0
```

> `--scalar_mask_dir`（2026-09-30 加入，默认空 = 不启用）：掩码为 0 的像素（地面常色块/天空/无深度）
> 会从**可见性与权重**中剔除 → 既不进余弦累加，也不进 BEP 精度场与覆盖度统计。
> 日志会打印被剔除的 `(顶点,视角)` 观测数，便于核对量级。

## 前提与依赖

- 数据集必须有 `depth_exr/`（相机轴 Z、米、32-bit float）与 `sparse/0/` 位姿；帧名需与 `images/` / `infrared/` 同名同序
- `heat_sources.json` 由 `export_heat_sources.py` 一次性导出（内容：15 个热点 / 地面 bbox / zones / 噪声参数 / palette / t_min / t_max）
- Python 依赖：`numpy`、`opencv-python`、`OpenEXR`、`Pillow`

### 不想开 Blender 也能导出热源？

`export_heat_sources.py` 的 `detect_bushings()` 就是技能里那套 bbox 高细比筛选（`min_height` / `min_ratio` / `z_min`，按顶端高度排名套 hotspot 规则）。
如果套管位置你本来就清楚，直接手写 `heat_sources.json` 即可（格式见内附的 `heat_sources_SC11.json` 示例）。

### MCP 方式（Blender 已开着场景时，无需命令行）

```python
import json
cfg = json.load(open('/path/DS/ir_config_SC11.json'))
g = {'__name__': 'ir_export', 'CONFIG': cfg, 'OUT_PATH': '/path/DS/heat_sources.json'}
exec(compile(open('/home/yyc/Documents/Tools/Blender/infrared/export_heat_sources.py').read(),
             'export_heat_sources.py', 'exec'), g)
```

## 常见坑

| 坑 | 说明 |
|---|---|
| 地面/天空假温度 | 见上文，必须用掩码；只靠"到色板折线的残差"判不出来（天空色离 IRON 暗端只有 0.02 的色差） |
| `OpenEXR` 新版写入 API | 不是 `OpenEXR.Channel(OpenEXR.FLOAT)`；正确写法 `OpenEXR.File({}, {"T": arr_2d}).write(path)`（标量通道用 2D 数组） |
| 直接 sRGB 插值色板 | 会引入 0.1–0.5 °C 系统偏差；必须走 `ir_palette.py` 的线性空间插值模型 |
| 值域与标定 | 16-bit PNG 存的是归一化值，下游必须传 `--scalar_scale (t_max-t_min) --scalar_offset t_min`；**切勿**把 `infrared/*.png` 直接当 `--scalar_dir`（会把 R 通道当温度） |
| 地面 bbox 容差 | `--ground-tol` 默认 0.35 m；地面有坡度/厚度时需调大，否则地面残留会被算成资产 |
| 帧对齐 | `depth_exr/0001.exr` 必须对应 `infrared/0001.png` 同一相机帧；换过相机动画/帧区间会错配 |
| `t_max` 饱和 | 热点超过 `t_max` 会被色板截断（SC11 已有 50.0 °C 饱和像素），需要更高量程就改 `t_max` 重渲/重算 |

## 文件

| 文件 | 作用 |
|---|---|
| `ir_palette.py` | 色板精确前向/反向模型（IRON/LAVA/WHITEHOT/RAINBOW） |
| `ir_geometry.py` | depth_exr + 位姿 → 逐像素世界坐标、地面/天空掩码 |
| `make_temperature.py` | **路线 A**：代码侧生成温度图（+ 掩码 + 可选伪彩图） |
| `palette_invert.py` | **路线 B**：伪彩红外图反演温度图（+ 几何掩码） |
| `export_heat_sources.py` | Blender 侧一次性导出热源 JSON（只读，不改场景） |
| `heat_sources_SC11.json` | SC11 的导出示例（15 热点 / 地面 bbox / 分区参数） |

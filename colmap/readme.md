# Blender 数据集 → MeshSplatting sparse 导出工具（两段式）

从 Blender 渲染数据集（depth_exr 反投影）生成 MeshSplatting 可直接训练的
COLMAP sparse（`cameras.bin + images.bin + points3D.bin + points3D_normals.npy`），
**无需 COLMAP SfM 重建**，弱纹理/重复场景下也不会匹配失败。

## 文件分工

| 脚本 | 运行环境 | 职责 | 依赖 |
|---|---|---|---|
| `export_sparse_from_blender.py` | **Blender 内**（`blender -b` 或 GUI Scripting） | 内参自动读当前相机 → `cameras.bin`；位姿 → `images.bin` | 仅 bpy + numpy |
| `generate_points3d.py` | **mesh_splatting 环境**（micromamba） | depth_exr 反投影 → 统计滤波 + 体素自适应 → `points3D.bin` + `points3D_normals.npy` | cv2 / OpenEXR / scipy |

分段的理由：Blender 内嵌 Python **没有** cv2 / OpenEXR / scipy，
而 RGB 采样、EXR 读取、统计滤波都需要它们；此处仅将"相机内参/位姿"这类
纯 bpy 能做的留在 Blender 内自动完成（换相机/分辨率**无需改代码**）。

## 快速开始

### 方式 A：Blender GUI Scripting 面板（粘贴运行）

1. 打开 `.blend` → 切换工作区到 **Scripting** → **New** 新建文本
2. 把 `export_sparse_from_blender.py` **整个内容**粘进去
3. **只改文件最顶部「★★★ 必改配置」三行**（见下），其余不动
4. 点 **运行（▶）** 按钮

```python
# ★★★ 必改配置 ★★★
OUTPUT_DIR   = "/mnt/8TB-HDD/.../BDZ01_v2"   # ← 输出位置：生成 <这里>/sparse/0/
DATA_DIR     = "/mnt/8TB-HDD/.../BDZ01"      # ← 位姿所在数据集根（找 processed_poses.txt）
CAMERA_NAME  = None                          # ← None=当前激活相机；或 "Camera2"
```

> 输出位置 = `OUTPUT_DIR` 本身：脚本会创建 `<OUTPUT_DIR>/sparse/0/`
> 并写入 `cameras.bin` + `images.bin`。
> ⚠️ 注意不要写 `.../sparse/0` —— 脚本会自动追加 `sparse/0/`。

### 方式 B：命令行（Blender -b 批处理）

```bash
cd /home/yyc/Documents/Tools/Blender/colmap
blender -b <你的场景.blend> --python export_sparse_from_blender.py -- \
    --out /path/to/output \
    --data_dir /path/to/dataset     # 数据集根（找 processed_poses.txt + images/ 取名）
    # 或 --poses /path/to/poses.txt （直接指定位姿文件）
```

### 方式 C：生成点云（mesh_splatting 环境，第二步）

```bash
micromamba activate mesh_splatting  # 或 eval "$(micromamba shell hook --shell bash)" 后 activate
env -u PYTHONPATH python generate_points3d.py \
    --data_dir /path/to/dataset \
    --sparse /path/to/output        # 与方式 A/B 的 OUTPUT_DIR 相同（含 sparse/0）
    --points 120000                 # 目标点数（体素自适应）
```

完成后数据集即可直接训练：

```bash
cd /home/yyc/Documents/Ones/mesh-splatting
python train.py -s /path/to/dataset -m output/<exp> \
    --enable_mono_seeds --mono_normal_dir /path/to/dataset/normals \
    --enable_adaptive_scaling --max_points 1000000 --densify_until_iter 6000 --data_device cpu
```

> ⚠️ 训练前置：把 sparse 放到数据集目录（`<dataset>/sparse/0/*`），
> 或让 `train.py -s` 指向含 sparse 的根目录。图像名（0001.png…）须与 images.bin 一致。

## 数据集目录要求

```
数据集根/
├── images/                RGB 图（PNG/JPG，命名 0001.png 起）
├── depth_exr/             深度 EXR（Blender Z pass，R 通道，far≈1e10）
├── normals/               世界系法线 PNG（可选，有则输出 points3D_normals.npy）
└── processed_poses.txt    每行 16 个数 = 列主序 4x4 w2c（500 行=500 帧）
```

若 processed_poses.txt 缺失，Blender 脚本支持 `--mode frames` 直接从
Blender 动画帧取位姿（`--image_root <images/>`，图像名须与帧号对应）。

## 关键参数

`export_sparse_from_blender.py`：
- `--out` 输出根目录（sparse/0 写在其下）
- `--cam` 相机对象名（默认场景激活相机，type=='CAMERA'）
- `--mode poses|frames` 位姿来源（默认 poses）
- `--poses` 位姿 txt 直接路径（优先于 --data_dir 内查找）
- `--data_dir` 数据集根：找 processed_poses.txt / camera_transforms.txt，并读 images/ 文件名

`generate_points3d.py`：
- `--points 120000` 体素自适应目标点数（默认 12 万；改 10 万/15 万均可）
- `--per_view 2000` 每视图采样像素上限（默认 2000；500 图 → 100 万原始点）
- `--voxel <m>` 固定体素尺寸（不设则由 --points 自动二分搜索）
- `--no_stat` 关闭统计离群点剔除（默认开：kNN=20, std=2.0）
- `--far 1e9` 深度远平面阈值

## 坐标系（方案 A = 3DGS/NeRF Blender 标准）

- Blender 相机系 -Z 前向 / Y up；COLMAP 系 +Z 前向 / Y down
- 点云保持 Blender 世界坐标**不翻转**；位姿列主序转置即 COLMAP w2c
- 深度 EXR 沿相机 -Z，在 COLMAP 系下对应 +Z 前向深度 d：
  `P_cam = (u-cx)/fx·d, (v-cy)/fy·d, d`；`P_world = w2c⁻¹·[P_cam,1]`
- 内参：HORIZONTAL fit → `fx = lens/sensor_width × 分辨率宽`，fy=fx，cx=W/2，cy=H/2

## 已验证（2026-08-25，BDZ01_v2 + BDZ02.blend）

- 内参自动读取：24mm/36mm/1280×960 → fx=853.333 ✓
- 位姿：BDZ01 processed_poses.txt（列主序 w2c_T）与 BDZ01_v2 原目录位姿逐帧一致 ✓
- 反投影：500 图 ×2000px = 100 万点 → 统计滤波 962,663 → 体素 0.206m → **117,809 点**
- 项目 readColmapSceneInfo 读回：500 相机、117,809 点、fx=853.33、
  前方 96.5%、单相机入图 81.4%、法线单位范数 1.0 ✓
- bbox 73×70×20m 正常（相机圈 22×14m + 视野深度 ≤78m，点云覆盖整个可见场景）

## 常见坑（踩过并已修）

1. **images.bin 的 image_id 是 int32**（4 字节，非 8）——按 8 字节读会指针错位、名字变空。
2. **processed_poses.txt 是列主序**（reshape(4,4) 后 t 在最后一行），须转置才成 w2c；
   `camera_transforms.txt` 才是行主序 c2w（t 在最后一列）。脚本默认 `POSES_INTERP='w2c_T'`。
3. **Blender 内无 cv2/OpenEXR/scipy**——EXR 可用 `bpy.data.images` 读（实测与 OpenEXR
   库逐像素一致）；RGB/法线采样与统计滤波放 generate_points3d.py（mesh_splatting 环境）。
4. **PNG 是 16-bit**（not 8-bit）——法线/图像读入后值域 0..1 按 65535 规约，采样前须确认。
5. **项目 CameraInfo.R 是 c2w 旋转**，投影公式是 `x_cam = R.T@X + T`（勿用 R@X）。
6. **BDZ01_v2 目录里没有 processed_poses.txt**——它实际在 `BDZ01/processed_poses.txt`
   （两者轨迹逐帧一致，见上），直接指路径即可。
# Blender GT 位姿直接导出为 COLMAP 二进制稀疏模型指南

本工具用于在 Ubuntu 系统下，直接将 Blender 中人工绘制/绑定的相机轨迹和相机内参，高精度转换并导出为 COLMAP 的标准二进制稀疏模型文件（`cameras.bin`, `images.bin`, `points3D.bin`）。

通过直接导出 Ground Truth (GT) 轨迹，**完美跳过并替代了 COLMAP 特征提取和 SfM（运动恢复结构）反推步骤**，彻底避免了弱纹理或重复场景下 COLMAP 匹配报错、尺度飘移及运动轨迹有误的问题。

---

## 📂 推荐项目目录结构

为了配合 3DGS (3D Gaussian Splatting)、NeRF 等下游算法的默认读取逻辑，建议将项目整理为如下结构：

```text
你的项目根目录/
├── images/               # 你已经从 Blender 中渲染并保存好的所有图片序列
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
└── sparse/
    └── 0/                # Blender 脚本输出的目标路径
        ├── cameras.bin   # 相机内参
        ├── images.bin    # 相机外参（完美轨迹）
        └── points3D.bin  # 空的点云文件（用于结构占位）

```



## 🚀 使用方法

### 1. 准备工作

确保你的 Blender 工程中：

- 相机的动画、路径和帧率与你渲染出图时的状态完全一致。
- 当前场景的 **Active Camera（激活相机）** 是你需要导出位姿的那个相机。

### 2. 运行脚本

1. 在 Blender 顶部工作区切换到 **Scripting (脚本)** 面板。
2. 点击 **New (新建)** 按钮，将转换脚本完整粘贴进去。
3. 根据你的 Ubuntu 系统实际情况，修改脚本顶部的配置项：
   - `output_dir`: 改为你存放 `.bin` 文件的绝对路径（例如 `"/home/user/dataset/sparse/0"`）。
   - `image_ext`: 改为图像的实际格式（Linux 严格区分大小写，如 `".png"` 或 `".jpg"`）。
   - `image_name_format`: 如果你的图片是 `0001.png` 则保持 `"{:04d}"`；如果是 `1.png` 则改为 `"{}"`。
4. 点击右上角的 **运行（播放键 ▷）** 按钮。

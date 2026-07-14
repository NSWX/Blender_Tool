
## 项目功能

这个项目是一个**Blender到Replica数据集的转换和渲染工具**，主要包含两个核心组件：

### 1. **Blender2Replica.py** - 姿态转换工具
- **功能**：将Blender导出的相机姿态转换为Replica SDK所需的格式
- **输入**：`camera_transforms.txt`（Blender导出的4×4变换矩阵）
- **输出**：
  - `traj.txt`：相机到世界坐标系的变换（c2w）
  - `processed_poses.txt`：世界到相机坐标系的变换（w2c），格式为 `(c2w⁻¹)ᵀ`

### 2. **render.cpp** - 渲染引擎
- **功能**：基于Replica SDK修改的渲染器，用于生成RGB-D图像
- **输入**：`processed_poses.txt`、3D网格文件（.ply）和纹理文件夹
- **输出**：RGB图像（frame000000.jpg）和深度图像（depth000000.png）

## 运行步骤

### 第一步：安装依赖
```bash
sudo apt-get install build-essential libgl1-mesa-dev
sudo apt-get install libglew-dev libsdl2-dev libsdl2-image-dev libglm-dev libfreetype6-dev
sudo apt-get install libglfw3-dev libglfw3
sudo apt-get install mesa-utils
sudo apt-get install libjpeg-dev libpng12-dev libtiff5-dev libopenexr-dev  
sudo apt-get install libjpeg-dev
sudo apt-get install doxygen
```

### 第二步：构建Replica项目
按照[Replica数据集官方文档](https://github.com/facebookresearch/Replica-Dataset)构建Replica SDK

### 第三步：转换Blender姿态
```bash
python Blender2Replica.py /path/to/your/blender/export/directory/
```
这会在指定目录生成`processed_poses.txt`和`traj.txt`文件

### 第四步：渲染图像
```bash
cd ~/ImageSave/office0_loop
~/Replica-Dataset/build/ReplicaSDK/ReplicaRenderer ~/RawMaterial/office_0/mesh.ply ~/RawMaterial/office_0/textures/
```

## 技术细节

- **图像尺寸**：1200×680像素
- **深度范围**：0.0001到100.0米
- **深度缩放**：65535.0f × 0.1f
- **相机参数**：与iMAP论文中的参数保持一致
- **输出格式**：RGB图像为JPG格式，深度图像为16位PNG格式

这个工具链特别适用于计算机视觉研究，特别是需要从Blender场景生成真实感RGB-D数据集的场景。
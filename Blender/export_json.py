import bpy
import json
import os
from mathutils import Matrix

# 确保获取到激活的相机
camera = bpy.context.scene.camera

if camera is not None and camera.type == 'CAMERA':
    # 设置路径配置
    dataset_dir = "/mnt/8TB-HDD/Documents1/Datasets/Electric_Assets/SC11"
    images_dir = os.path.join(dataset_dir, "images")
    export_file = os.path.join(dataset_dir, "camera_transforms.json")
    sequence_name = "sequence_0001"
    
    # 1. 自动扫描 images 目录下所有的 .png 图片文件
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.png')])
    
    if not image_files:
        raise ValueError(f"No .png image files found in {images_dir}")

    # 获取动画的起始和当前帧
    start_frame = bpy.context.scene.frame_start
    current_frame = bpy.context.scene.frame_current

    sequence_data = []

    # Blender 到 OpenCV 坐标系的转换矩阵（翻转 Y 轴和 Z 轴）
    # Blender: X右, Y上, Z后 -> OpenCV: X右, Y下, Z前
    to_opencv = Matrix((
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ))

    # 2. 遍历图像文件，按顺序匹配 Blender 动画帧
    for i, img_filename in enumerate(image_files):
        # 对应 Blender 的帧：第一张图对应 start_frame，第二张对应 start_frame + 1，以此类推
        frame = start_frame + i
        
        # 如果超出了 Blender 动画的总帧数，则给出提示并跳过
        if frame > bpy.context.scene.frame_end:
            print(f"Warning: Image count exceeds Blender animation frame range at {img_filename}")
            break

        # 设置 Blender 帧并更新场景
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        bpy.context.evaluated_depsgraph_get().update()

        # 获取相机世界矩阵并乘以转换矩阵（转为 OpenCV 坐标系）
        matrix_world = camera.matrix_world.copy()
        matrix = matrix_world @ to_opencv

        # 提取 3x3 旋转矩阵 R
        R = [
            [matrix[0][0], matrix[0][1], matrix[0][2]],
            [matrix[1][0], matrix[1][1], matrix[1][2]],
            [matrix[2][0], matrix[2][1], matrix[2][2]]
        ]
        
        # 提取 3x1 平移向量 T (矩阵第四列的前三行)
        T = [matrix[0][3], matrix[1][3], matrix[2][3]]

        # 拼接你期望的相对 filepath（如 sequence_0001/images/0001.png）
        filepath = f"{sequence_name}/images/{img_filename}"

        # 添加到列表
        sequence_data.append({
            "filepath": filepath,
            "R": R,
            "T": T
        })

    # 恢复场景原本的当前帧
    bpy.context.scene.frame_set(current_frame)

    # 组装最终的 JSON 结构
    output_data = {
        sequence_name: sequence_data
    }

    # 确保输出目录存在
    os.makedirs(os.path.dirname(export_file), exist_ok=True)

    # 导出为 JSON 文件
    with open(export_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(output_data, jsonfile, indent=2)

    print(f"Successfully matched {len(sequence_data)} PNG images and exported to: {export_file}")
else:
    print("No camera selected or active object is not a camera.")

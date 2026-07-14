import bpy
import mathutils
import os
import math
import struct

# ==================== 1. 自定义配置项 (Ubuntu 路径格式) ====================
# 请修改为你在 Ubuntu 上的目标保存绝对路径 (末尾不需要加斜杠)
output_dir = "/home/your_username/dataset/sparse/0" 

# 请修改为你渲染导出的图片格式（必须与你已有的图片后缀完全一致，如 .png, .jpg）
image_ext = ".png" 

# 图像文件名的编号格式：
# "{:04d}" 对应 0001.png, 0002.png 等（4位补零）
# "{}"     对应 1.png, 2.png 等（无补零）
image_name_format = "{:04d}" 
# =========================================================================

def get_camera_intrinsics(cam_ob, scene):
    """ 高精度计算相机的 PINHOLE 内参 (fx, fy, cx, cy) """
    cam_data = cam_ob.data
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    
    sensor_width = cam_data.sensor_width
    sensor_height = cam_data.sensor_height
    focal_length_mm = cam_data.lens
    
    # 适配 Blender 的不同 Sensor Fit 模式
    if cam_data.sensor_fit == 'HORIZONTAL' or (cam_data.sensor_fit == 'AUTO' and width >= height):
        fx = (focal_length_mm / sensor_width) * width
        fy = fx
    else:
        fy = (focal_length_mm / sensor_height) * height
        fx = fy
        
    cx = width / 2.0 + cam_data.shift_x * width
    shift_scale = width if width > height else height
    cy = height / 2.0 - cam_data.shift_y * shift_scale
    
    return width, height, fx, fy, cx, cy

def export_colmap_binary():
    scene = bpy.context.scene
    cam_ob = scene.camera
    
    if not cam_ob:
        print("【错误】：场景中未找到激活的相机 (Active Camera)！")
        return
        
    # 规范化 Linux 路径并建立文件夹
    out_path = os.path.abspath(os.path.expanduser(output_dir))
    try:
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
    except Exception as e:
        print(f"【错误】：无法创建目标目录，请检查 Ubuntu 文件系统权限！\n 详情: {e}")
        return
        
    width, height, fx, fy, cx, cy = get_camera_intrinsics(cam_ob, scene)
    
    # ------------------ 一、写入 cameras.bin ------------------
    cameras_path = os.path.join(out_path, "cameras.bin")
    with open(cameras_path, "wb") as f:
        f.write(struct.pack("<Q", 1))                         # 只有 1 个相机 (uint64)
        f.write(struct.pack("<iiQQ", 1, 1, width, height))   # camera_id=1, model_id=1 (PINHOLE), w, h
        f.write(struct.pack("<dddd", fx, fy, cx, cy))         # 内参参数 (4个 double)
        
    # ------------------ 二、写入 images.bin ------------------
    # 从 Blender 坐标系 (右, 上, 后) 到 COLMAP 坐标系 (右, 下, 前) 的转换矩阵
    R_flip = mathutils.Matrix.Rotation(math.pi, 4, 'X')
    
    start_frame = scene.frame_start
    end_frame = scene.frame_end
    num_images = end_frame - start_frame + 1
    
    images_path = os.path.join(out_path, "images.bin")
    with open(images_path, "wb") as f:
        # 写入图片总数量 (uint64)
        f.write(struct.pack("<Q", num_images))
        
        image_id = 1
        for frame in range(start_frame, end_frame + 1):
            scene.frame_set(frame)
            
            # 计算 World-to-Camera (w2c)
            c2w = cam_ob.matrix_world.copy()
            c2w_colmap = c2w @ R_flip
            w2c_colmap = c2w_colmap.inverted()
            
            # 提取四元数 R 和平移向量 t
            q = w2c_colmap.to_quaternion()
            t = w2c_colmap.to_translation()
            
            # 生成对应的渲染文件名（Linux 严格区分大小写，需要带 C 风格 null 结束符）
            image_name = f"{image_name_format.format(frame)}{image_ext}"
            image_name_bytes = (image_name + "\0").encode('utf-8')
            
            # 写入每张图像的头部外参数据:
            # image_id (int32), qw, qx, qy, qz, tx, ty, tz (均为 double), camera_id (int32, 固定为1)
            f.write(struct.pack("<idddddddi", image_id, q.w, q.x, q.y, q.z, t.x, t.y, t.z, 1))
            f.write(image_name_bytes)
            
            # 2D 特征点数量写入 0 (uint64)，代表没有经过 SfM 提取点
            f.write(struct.pack("<Q", 0))
            
            image_id += 1

    # ------------------ 三、写入 points3D.bin ------------------
    # 写入一个空的 3D 点云二进制数据头，防止 3DGS/NeRF 的数据加载模块报错
    points3D_path = os.path.join(out_path, "points3D.bin")
    with open(points3D_path, "wb") as f:
        f.write(struct.pack("<Q", 0)) # 3D点数量 = 0 (uint64)
        
    print(f"\n🎉 Linux 环境下 COLMAP 稀疏模型导出成功！")
    print(f"📂 输出目录: {out_path}")

# 执行函数
export_colmap_binary()

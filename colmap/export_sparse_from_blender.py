# ============================================================================
# ★★★ 必改配置（Blender Scripting 面板粘贴运行：只需改这里 ★★★）
# ============================================================================
# 输出位置：cameras.bin + images.bin 将写到  <OUTPUT_DIR>/sparse/0/
# 例：OUTPUT_DIR = "/mnt/8TB-HDD/Documents1/Datasets/Electric_Assets/BDZ01_v2"
#     → 生成 BDZ01_v2/sparse/0/cameras.bin + images.bin
OUTPUT_DIR   = "/mnt/8TB-HDD/Documents1/Datasets/Electric_Assets/Transforme02"

# 位姿来源（二选一，填一个即可）：
#   1) 数据集根目录（内含 processed_poses.txt 或 camera_transforms.txt）：
DATA_DIR     =     OUTPUT_DIR
#   2) 或位姿 txt 直接路径（优先；None 则由 DATA_DIR 查找）：
POSES_FILE   = None

# 相机：None = 场景激活相机；或指定对象名如 "Camera2"
CAMERA_NAME  = None

# ============================================================================
# 可选配置（一般无需改动）
# ============================================================================
MODE         = "frames"        # "poses" 读位姿 txt | "frames" Blender 逐帧取位姿
IMAGE_ROOT   = OUTPUT_DIR + "/images"           # frames 模式：images/ 目录
IMAGE_EXT    = ".png"         # frames 模式：图像扩展名
FRAME_START  = 1              # frames 模式：起始帧（图像 0001.png -> 帧 1）
FRAME_END    = -1             # frames 模式：-1 = scene.frame_end
POSES_INTERP = "w2c_T"        # poses 文件解释：'w2c'|'w2c_T'(列主序,默认)|'c2w'

# ============================================================================
# 以下代码无需改动
# ============================================================================

"""
export_sparse_from_blender.py — 在 Blender 内导出 cameras.bin + images.bin
============================================================================
职责边界（与 generate_points3d.py 分工）：
    · 本脚本（Blender 内运行，仅依赖 bpy + numpy）：
        内参自动从当前激活相机（type == 'CAMERA'）读取 → cameras.bin
        位姿（processed_poses.txt 或 Blender 逐帧）→ images.bin
    · generate_points3d.py（mesh_splatting 环境运行，需要 cv2/OpenEXR/scipy）：
        depth_exr 反投影 → points3D.bin + points3D_normals.npy

用法：
    1) Blender GUI Scripting 面板：粘贴整个脚本 → 改顶部 OUTPUT_DIR/DATA_DIR
       → 点运行（▶）。此路径无需命令行参数。
    2) 命令行（等价）：
       blender -b <场景.blend> --python export_sparse_from_blender.py -- \
           --out /path/to/sparse --data_dir /path/to/dataset
    3) 无 processed_poses.txt 时：MODE="frames" 从 Blender 逐帧取位姿
       （图像名须与帧号对应，如 0001.png -> frame 1；IMAGE_ROOT 指定 images/）。

坐标系约定（方案 A = 3DGS/NeRF Blender 标准）：
    Blender 相机系: -Z 前向, Y up（OpenGL 约定）
    COLMAP 相机系:  +Z 前向, Y down（图像 v 轴向下）
    · 点云保持 Blender 世界坐标，不做翻转（由 generate_points3d.py 保证）
    · processed_poses.txt 为列主序矩阵（reshape(4,4) 后平移在最后一行），
      读入转置即 COLMAP w2c（t 在最后一列）；脚本自动处理。
    · Blender 逐帧位姿：c2w_colmap = c2w_blender @ C，C = diag(1,-1,-1)

输出：
    <out>/sparse/0/cameras.bin   PINHOLE，内参自动计算
    <out>/sparse/0/images.bin    w2c 位姿（qvec + tvec）
"""

import bpy
import mathutils
import os
import sys
import math
import struct
import argparse

import numpy as np

# ==================== 1. 内参（自动读当前相机） ====================
def get_camera_intrinsics(cam_ob, scene):
    """由 Blender 相机对象 + 渲染分辨率计算 PINHOLE 内参（mm -> px）。"""
    cam_data = cam_ob.data
    width  = scene.render.resolution_x
    height = scene.render.resolution_y
    sw, sh = cam_data.sensor_width, cam_data.sensor_height
    fl = cam_data.lens
    if cam_data.sensor_fit == 'HORIZONTAL' or (cam_data.sensor_fit == 'AUTO' and width >= height):
        fx = (fl / sw) * width
        fy = fx
    else:
        fy = (fl / sh) * height
        fx = fy
    cx = width / 2.0 + cam_data.shift_x * width
    cy = height / 2.0 - cam_data.shift_y * (width if width > height else height)
    return width, height, fx, fy, cx, cy


def find_camera(scene, name=None):
    if name:
        ob = scene.objects.get(name)
        if ob and ob.type == 'CAMERA':
            return ob
    if scene.camera and scene.camera.type == 'CAMERA':
        return scene.camera
    for ob in scene.objects:
        if ob.type == 'CAMERA':
            return ob
    return None


# ==================== 2. 位姿 ====================
def load_poses_txt(path):
    """解析位姿 txt：每行 16 个数 -> Nx4x4 float64。也兼容每行 12 个数。"""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    mats = []
    for line in lines:
        v = [float(x) for x in line.split()]
        if len(v) == 16:
            mats.append(np.array(v, np.float64).reshape(4, 4))
        elif len(v) == 12:
            m = np.eye(4)
            m[:3, :4] = np.array(v, np.float64).reshape(3, 4)
            mats.append(m)
    return np.stack(mats) if mats else None


def interp_w2c(M):
    """按 POSES_INTERP 约定把原始 4x4 矩阵变成 COLMAP w2c（t 在最后一列）。"""
    if POSES_INTERP == 'w2c':
        return M
    if POSES_INTERP == 'w2c_T':
        return M.T
    if POSES_INTERP == 'c2w':
        return np.linalg.inv(M)
    raise ValueError(f"未知 POSES_INTERP: {POSES_INTERP}")


def mat_to_qvec_t(M):
    """w2c 旋转矩阵 -> (qvec, tvec)，COLMAP qvec 顺序 (w,x,y,z)。"""
    Rm = M[:3, :3]
    t = M[:3, 3]
    tr = Rm[0, 0] + Rm[1, 1] + Rm[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = 0.25 * s, (Rm[2, 1] - Rm[1, 2]) / s, (Rm[0, 2] - Rm[2, 0]) / s, (Rm[1, 0] - Rm[0, 1]) / s
    else:
        i = int(np.argmax([Rm[0, 0], Rm[1, 1], Rm[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2
            qw, qx, qy, qz = (Rm[2, 1] - Rm[1, 2]) / s, 0.25 * s, (Rm[0, 1] + Rm[1, 0]) / s, (Rm[0, 2] + Rm[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2
            qw, qx, qy, qz = (Rm[0, 2] - Rm[2, 0]) / s, (Rm[0, 1] + Rm[1, 0]) / s, 0.25 * s, (Rm[1, 2] + Rm[2, 1]) / s
        else:
            s = np.sqrt(1.0 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2
            qw, qx, qy, qz = (Rm[1, 0] - Rm[0, 1]) / s, (Rm[0, 2] + Rm[2, 0]) / s, (Rm[1, 2] + Rm[2, 1]) / s, 0.25 * s
    return np.array([qw, qx, qy, qz]), t


# ==================== 3. 写盘 ====================
def write_cameras_bin(path, width, height, fx, fy, cx, cy):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, width, height))
        f.write(struct.pack("<dddd", fx, fy, cx, cy))


def write_images_bin(path, entries):
    """entries: [(image_name, w2c 4x4)] -> COLMAP images.bin（qvec/tvec）。"""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(entries)))
        for i, (name, w2c) in enumerate(entries):
            q, t = mat_to_qvec_t(w2c)
            f.write(struct.pack("<idddddddi", i + 1, q[0], q[1], q[2], q[3], t[0], t[1], t[2], 1))
            f.write((name + "\0").encode("utf-8"))
            f.write(struct.pack("<Q", 0))


# ==================== 4. 主流程 ====================
def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUTPUT_DIR)
    ap.add_argument("--cam", default=CAMERA_NAME, help="相机对象名（None=激活相机）")
    ap.add_argument("--mode", choices=["poses", "frames"], default=MODE,
                    help="poses=读位姿 txt；frames=Blender 逐帧")
    ap.add_argument("--data_dir", default=DATA_DIR, help="poses 模式：数据集根（找 processed_poses.txt）")
    ap.add_argument("--poses", default=POSES_FILE, help="poses 模式：位姿 txt 直接路径")
    ap.add_argument("--image_root", default=IMAGE_ROOT, help="frames 模式：images/ 目录")
    ap.add_argument("--image_ext", default=IMAGE_EXT)
    ap.add_argument("--frame_start", type=int, default=FRAME_START)
    ap.add_argument("--frame_end", type=int, default=FRAME_END)
    args = ap.parse_args(argv)

    scene = bpy.context.scene
    cam_ob = find_camera(scene, args.cam)
    if not cam_ob:
        print(f"[ERROR] 找不到相机（name={args.cam!r}）。请确认场景存在 type=='CAMERA' 对象。")
        return 1

    # ---- 内参：自动从当前相机读取 ----
    width, height, fx, fy, cx, cy = get_camera_intrinsics(cam_ob, scene)
    print(f"[INFO] 相机: {cam_ob.name}  lens={cam_ob.data.lens}mm "
          f"sensor={cam_ob.data.sensor_width}x{cam_ob.data.sensor_height} fit={cam_ob.data.sensor_fit}")
    print(f"[INFO] 内参: fx={fx:.4f} fy={fy:.4f} cx={cx:.2f} cy={cy:.2f} @ {width}x{height}")

    out_sparse = os.path.join(args.out, "sparse", "0")
    os.makedirs(out_sparse, exist_ok=True)
    write_cameras_bin(os.path.join(out_sparse, "cameras.bin"), width, height, fx, fy, cx, cy)
    print(f"[INFO] cameras.bin 已写入（PINHOLE {width}x{height} fx={fx:.2f}）")

    # ---- 位姿 ----
    C_FLIP = mathutils.Matrix.Rotation(math.pi, 4, 'X')  # diag(1,-1,-1)

    if args.mode == "poses":
        poses_path = args.poses
        if not poses_path and args.data_dir:
            for cand in ("processed_poses.txt", "camera_transforms.txt"):
                p = os.path.join(args.data_dir, cand)
                if os.path.exists(p):
                    poses_path = p
                    break
        if not poses_path or not os.path.exists(poses_path):
            print("[ERROR] poses 模式需要 --poses <txt> 或 --data_dir 含 processed_poses.txt")
            return 1
        poses = load_poses_txt(poses_path)
        if poses is None:
            print(f"[ERROR] 无法解析位姿文件: {poses_path}")
            return 1
        # 图像名：优先用 images/ 目录文件名（保序），否则 0001..N
        names = None
        if args.data_dir:
            img_dir = os.path.join(args.data_dir, "images")
            if os.path.isdir(img_dir):
                cand = sorted(n for n in os.listdir(img_dir)
                              if n.lower().endswith((".png", ".jpg", ".jpeg")))
                if len(cand) >= len(poses):
                    names = cand[:len(poses)]
                elif len(cand) == len(poses):
                    names = cand
        if names is None:
            names = [f"{i + 1:04d}{args.image_ext}" for i in range(len(poses))]
        entries = [(names[i], interp_w2c(poses[i])) for i in range(len(poses))]
        print(f"[INFO] 位姿来源: {poses_path}（{len(entries)} 帧，POSES_INTERP={POSES_INTERP}）")
    else:  # frames 模式：Blender 逐帧
        if not args.image_root:
            print("[ERROR] frames 模式需要 --image_root <images/>")
            return 1
        image_names = sorted(n for n in os.listdir(args.image_root)
                             if n.lower().endswith(args.image_ext.lower()))
        end = scene.frame_end if args.frame_end <= 0 else args.frame_end
        n_frames = min(len(image_names), end - args.frame_start + 1)
        entries = []
        for k in range(n_frames):
            fr = args.frame_start + k
            scene.frame_set(fr)
            deps = bpy.context.evaluated_depsgraph_get()
            c2w = cam_ob.matrix_world.copy()
            c2w_colmap = c2w @ C_FLIP
            w2c = c2w_colmap.inverted()
            w2c_np = np.array([[w2c[l][m] for m in range(4)] for l in range(4)], np.float64)
            entries.append((image_names[k], w2c_np))
        print(f"[INFO] 位姿来源: Blender 逐帧（{len(entries)} 帧，frame {args.frame_start}.."
              f"{args.frame_start + n_frames - 1}，图像 {image_names[0]}..{image_names[-1]}）")

    write_images_bin(os.path.join(out_sparse, "images.bin"), entries)
    print(f"[DONE] images.bin 已写入（{len(entries)} 帧）-> {out_sparse}")
    print(f"[提示] 下一步运行 generate_points3d.py 生成 points3D.bin")
    return 0


if __name__ == "__main__":
    # 注意：GUI Scripting 面板里不能用 raise SystemExit —— 会让整个 Blender 退出！
    # 这里用普通调用，错误通过返回值 + 打印报告。
    main()

# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/data/imc_preprocessing.py

import os
import struct
import argparse
import collections
import numpy as np

Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

# Mapping of COLMAP camera model id -> (name, num_params)
CAMERA_MODEL_INFO = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("SIMPLE_RADIAL_FISHEYE", 4),
    4: ("RADIAL", 5),
    5: ("RADIAL_FISHEYE", 5),
    6: ("OPENCV", 8),
    7: ("OPENCV_FISHEYE", 8),
    8: ("FULL_OPENCV", 12),
    9: ("FOV", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

def qvec2rotmat(qvec):
    """Convert quaternion (qw, qx, qy, qz) -> 3x3 rotation matrix (Hamilton convention)."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),       2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy),     2 * (qy * qz + qw * qx),     1 - 2 * (qx * qx + qy * qy)]
    ], dtype=float)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack exactly num_bytes using struct format."""
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError(f"Expected {num_bytes} bytes but got {len(data)} bytes")
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path_to_model_file):
    """Read COLMAP cameras.bin (binary) and return dict camera_id -> Camera(...)"""
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id = read_next_bytes(fid, 4, "i")[0]
            model_id = read_next_bytes(fid, 4, "i")[0]
            
            width = read_next_bytes(fid, 8, "Q")[0]
            height = read_next_bytes(fid, 8, "Q")[0]
            
            model_id = int(model_id)
            model_name, num_params = CAMERA_MODEL_INFO.get(model_id, (f"MODEL_{model_id}", 4))
            
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = Camera(id=camera_id, model=model_name, width=width, height=height, params=np.array(params))
    return cameras

def read_images_binary(path_to_model_file):
    """Read COLMAP images.bin (binary) and return dict image_id -> Image(...)"""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(fid, 4, "i")[0]
            qvec = np.array(read_next_bytes(fid, 8 * 4, "d" * 4))
            tvec = np.array(read_next_bytes(fid, 8 * 3, "d" * 3))
            camera_id = read_next_bytes(fid, 4, "i")[0]

            name_bytes = bytearray()
            while True:
                b = fid.read(1)
                if len(b) == 0:
                    raise EOFError("Unexpected EOF while reading image name")
                if b == b'\x00':
                    break
                name_bytes.extend(b)
            image_name = name_bytes.decode("utf-8")

            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(num_points2D * 24, os.SEEK_CUR)

            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=None, point3D_ids=None)
    return images

def read_model(path):
    cameras = read_cameras_binary(os.path.join(path, "cameras.bin"))
    images = read_images_binary(os.path.join(path, "images.bin"))
    return cameras, images

def get_intrinsic_matrix(cam):
    """Helper to cleanly extract the 3x3 intrinsic matrix based on camera model."""
    if cam.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE"]:
        fx = fy = cam.params[0]
        cx = cam.params[1]
        cy = cam.params[2]
    elif cam.model in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"]:
        fx, fy, cx, cy = cam.params[:4]
    else:
        return None
        
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=float)

# ---------------- Main processing script ----------------

def extract_and_write(base_dir):
    print(f"🔍 Starting processing for scenes in: {base_dir}")

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    for scene_name in sorted(os.listdir(base_dir)):
        scene_path = os.path.join(base_dir, scene_name)
        
        if not os.path.isdir(scene_path):
            continue

        dense_dir = os.path.join(scene_path, "dense")
        sparse_dir = os.path.join(dense_dir, "sparse")

        if not os.path.exists(dense_dir):
            print(f"⚠️  Skipping '{scene_name}': No 'dense' folder found.")
            continue
        
        if not os.path.exists(sparse_dir):
            print(f"⚠️  Skipping '{scene_name}': No 'sparse' folder found inside 'dense/'.")
            continue

        poses_output_dir = os.path.join(dense_dir, "poses")
        intrinsics_output_dir = os.path.join(dense_dir, "intrinsics")
        os.makedirs(poses_output_dir, exist_ok=True)
        os.makedirs(intrinsics_output_dir, exist_ok=True)

        print(f"➡️  Processing scene: {scene_name}")

        try:
            cameras, images = read_model(sparse_dir)
        except Exception as e:
            print(f"⚠️  Failed reading binaries in {sparse_dir}: {e}")
            continue

        if len(cameras) == 0 or len(images) == 0:
            print(f"⚠️  No cameras/images found in {sparse_dir}, skipping.")
            continue

        # Retain the fallback global intrinsics write
        cam_id = 1 if 1 in cameras else sorted(cameras.keys())[0]
        global_cam = cameras[cam_id]
        global_K = get_intrinsic_matrix(global_cam)
        
        if global_K is not None:
            intrinsics_txt_path = os.path.join(dense_dir, "camera_intrinsics.txt")
            np.savetxt(intrinsics_txt_path, global_K, fmt="%.6f")
            print(f"   • Wrote global intrinsics to {intrinsics_txt_path} (model: {global_cam.model})")
        else:
            print(f"⚠️  Skipping Global Intrinsics for {scene_name}: Unsupported camera model {global_cam.model}")

        # Write camera-to-world (c2w) matrices and individual intrinsics for each image
        sorted_images = sorted(images.values(), key=lambda img: img.name)
        missing_intrinsic_count = 0
        
        for img in sorted_images:
            # 1. Calculate and save pose
            R_w2c = qvec2rotmat(img.qvec)
            t_w2c = img.tvec
            R_c2w = R_w2c.T
            t_c2w = -R_c2w @ t_w2c

            c2w = np.eye(4, dtype=float)
            c2w[:3, :3] = R_c2w
            c2w[:3, 3] = t_c2w

            base_filename = os.path.splitext(os.path.basename(img.name))[0]
            pose_file_path = os.path.join(poses_output_dir, f"{base_filename}.txt")
            np.savetxt(pose_file_path, c2w, fmt="%.8e")
            
            # 2. Extract and save per-image intrinsic
            cam = cameras[img.camera_id]
            K = get_intrinsic_matrix(cam)
            if K is not None:
                intrinsics_file_path = os.path.join(intrinsics_output_dir, f"{base_filename}.txt")
                np.savetxt(intrinsics_file_path, K, fmt="%.6f")
            else:
                missing_intrinsic_count += 1
            
        print(f"   • Wrote {len(sorted_images)} poses to {poses_output_dir}/")
        print(f"   • Wrote {len(sorted_images) - missing_intrinsic_count} per-image intrinsics to {intrinsics_output_dir}/")
        if missing_intrinsic_count > 0:
            print(f"   • ⚠️ Skipped {missing_intrinsic_count} intrinsics due to unsupported models.")

    print("\n🎉 All scenes processed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the IMC data folder")
    args = parser.parse_args()
    extract_and_write(args.data_path)
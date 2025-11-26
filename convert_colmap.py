import os
import argparse
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import struct


def rotmat2qvec(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]


def read_ply_binary(ply_path):
    """Read binary PLY file and extract vertices (x, y, z, r, g, b).

    Returns:
        points: Nx3 array of xyz coordinates
        colors: Nx3 array of RGB values (0-255)
    """
    with open(ply_path, 'rb') as f:
        # Read header
        line = f.readline().decode('utf-8').strip()
        if line != 'ply':
            raise ValueError(f"Not a valid PLY file: {ply_path}")

        vertex_count = 0
        while True:
            line = f.readline().decode('utf-8').strip()
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line == 'end_header':
                break

        # Read binary vertex data: 3 floats (x,y,z) + 3 uchars (r,g,b)
        # Format: '<3f3B' = little-endian, 3 floats, 3 unsigned chars
        points = []
        colors = []
        for _ in range(vertex_count):
            data = f.read(15)  # 3*4 bytes (floats) + 3*1 bytes (uchars) = 15 bytes
            x, y, z, r, g, b = struct.unpack('<3f3B', data)
            points.append([x, y, z])
            colors.append([r, g, b])

        return np.array(points), np.array(colors)


def main(exp_dir, image_dir, pcd_file=None):
    """Convert VGGT-Long output to COLMAP format.

    Args:
        exp_dir: VGGT-Long experiment output directory
        image_dir: Original input image directory
        pcd_file: Optional path to PLY point cloud file to append to points3D.txt
    """
    poses_file = os.path.join(exp_dir, 'camera_poses.txt')
    intrinsics_file = os.path.join(exp_dir, 'intrinsic.txt')
    colmap_dir = os.path.join(exp_dir, 'colmap')

    os.makedirs(colmap_dir, exist_ok=True)

    # Read C2W poses (4x4 matrices)
    c2w_poses = []
    with open(poses_file, 'r') as f:
        for line in f:
            if line.strip():
                c2w_poses.append(np.fromstring(line.strip(), sep=' ').reshape(4, 4))

    # Read intrinsics (fx, fy, cx, cy)
    intrinsics = []
    with open(intrinsics_file, 'r') as f:
        for line in f:
            if line.strip():
                intrinsics.append(np.fromstring(line.strip(), sep=' '))

    # Get image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

    if len(c2w_poses) != len(intrinsics) or len(c2w_poses) != len(image_files):
        print(f"Error: Mismatched counts - poses:{len(c2w_poses)}, intrinsics:{len(intrinsics)}, images:{len(image_files)}")
        return

    # Write cameras.txt
    with open(os.path.join(colmap_dir, 'cameras.txt'), 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, (fx, fy, cx, cy) in enumerate(intrinsics):
            with Image.open(os.path.join(image_dir, image_files[i])) as img:
                width, height = img.size
            f.write(f"{i+1} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    # Write images.txt
    with open(os.path.join(colmap_dir, 'images.txt'), 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[]\n")
        for i, c2w in enumerate(c2w_poses):
            # C2W to W2C conversion: R_w2c = R_c2w^T, T = -R_w2c @ C
            R_c2w = c2w[:3, :3]
            C = c2w[:3, 3]
            R_w2c = R_c2w.T
            T = -R_w2c @ C
            q = rotmat2qvec(R_w2c)
            f.write(f"{i+1} {q[0]} {q[1]} {q[2]} {q[3]} {T[0]} {T[1]} {T[2]} {i+1} {image_files[i]}\n")
            f.write("\n")

    # Write points3D.txt
    with open(os.path.join(colmap_dir, 'points3D.txt'), 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")

        # If PCD file is provided, read and append points
        if pcd_file and os.path.exists(pcd_file):
            print(f"Reading point cloud from {pcd_file}...")
            points, colors = read_ply_binary(pcd_file)
            print(f"Loaded {len(points)} points from PCD file")

            # Write each point to points3D.txt
            # Format: POINT3D_ID X Y Z R G B ERROR TRACK[]
            # ERROR is set to 0.0, TRACK[] is empty (no feature tracking from PCD)
            for i, (pt, col) in enumerate(zip(points, colors)):
                point_id = i + 1
                x, y, z = pt
                r, g, b = col
                f.write(f"{point_id} {x} {y} {z} {int(r)} {int(g)} {int(b)} 0.0 0 0 0 0 0 0\n")

            print(f"Written {len(points)} points to points3D.txt")
        else:
            if pcd_file:
                print(f"Warning: PCD file not found at {pcd_file}")

    print(f"COLMAP format written to {colmap_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert VGGT-Long output to COLMAP format')
    parser.add_argument('--exp_dir', required=True, help='VGGT-Long experiment output directory')
    parser.add_argument('--image_dir', required=True, help='Original input image directory')
    parser.add_argument('--pcd_file', default=None, help='Optional PLY point cloud file to append to points3D.txt')
    args = parser.parse_args()
    main(args.exp_dir, args.image_dir, args.pcd_file)
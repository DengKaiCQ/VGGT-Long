import os
import argparse
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import struct
from typing import Optional, Tuple


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


def scale_intrinsics(fx, fy, cx, cy, width, height, pred_w: Optional[int] = None, pred_h: Optional[int] = None, disable: bool = False):
    """Minimal helper to scale intrinsics from predicted resolution to image resolution.

    If pred_w/pred_h are None, infer from cx,cy as round(cx*2), round(cy*2).
    """
    if disable:
        return fx, fy, cx, cy, 1.0, 1.0

    if pred_w is None:
        pred_w = int(round(cx * 2)) if cx > 0 else width
    if pred_h is None:
        pred_h = int(round(cy * 2)) if cy > 0 else height

    if pred_w <= 0 or pred_h <= 0 or pred_w > width or pred_h > height:
        scale_x = 1.0
        scale_y = 1.0
    else:
        scale_x = float(width) / float(pred_w)
        scale_y = float(height) / float(pred_h)

    fx_s = fx * scale_x
    fy_s = fy * scale_y
    cx_s = cx * scale_x
    cy_s = cy * scale_y

    return fx_s, fy_s, cx_s, cy_s, scale_x, scale_y


def main(exp_dir, image_dir, pcd_file=None, no_scale=False, pred_w=None, pred_h=None, verbose=False):
    """Convert VGGT-Long output to COLMAP format.

    Args:
        exp_dir: VGGT-Long experiment output directory
        image_dir: Original input image directory
        pcd_file: Optional path to PLY point cloud file to append to points3D.txt
        no_scale: if True, disable heuristic intrinsics scaling
        pred_w/pred_h: optional manual prediction resolution overrides
        verbose: print diagnostic messages
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

    # Write cameras.txt (with heuristic scaling)
    with open(os.path.join(colmap_dir, 'cameras.txt'), 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, (fx, fy, cx, cy) in enumerate(intrinsics):
            with Image.open(os.path.join(image_dir, image_files[i])) as img:
                width, height = img.size


            fx_s, fy_s, cx_s, cy_s, scale_x, scale_y = scale_intrinsics(
                float(fx), float(fy), float(cx), float(cy), width, height, pred_w=pred_w, pred_h=pred_h, disable=no_scale
            )

            if verbose and (abs(scale_x - 1.0) > 1e-3 or abs(scale_y - 1.0) > 1e-3):
                inferred_pw = pred_w if pred_w is not None else int(round(cx * 2))
                inferred_ph = pred_h if pred_h is not None else int(round(cy * 2))
                print(f"[convert] camera {i+1}: inferred pred {inferred_pw}x{inferred_ph} -> scale {scale_x:.3f},{scale_y:.3f}; fx {fx:.2f}->{fx_s:.2f}, cx {cx:.1f}->{cx_s:.1f}")

            f.write(f"{i+1} PINHOLE {width} {height} {fx_s} {fy_s} {cx_s} {cy_s}\n")

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
    parser.add_argument('--no-scale', action='store_true', help='Disable heuristic intrinsics scaling')
    parser.add_argument('--pred_w', type=int, default=None, help='Override inferred prediction width')
    parser.add_argument('--pred_h', type=int, default=None, help='Override inferred prediction height')
    parser.add_argument('--verbose', action='store_true', help='Print diagnostic messages')
    args = parser.parse_args()
    main(args.exp_dir, args.image_dir, args.pcd_file, no_scale=args.no_scale, pred_w=args.pred_w, pred_h=args.pred_h, verbose=args.verbose)

# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/tools/vis.py

import open3d as o3d
import numpy as np


def draw_camera(c2w, cam_width=0.24/2, cam_height=0.16/2, f=0.10, color=[0, 1, 0], show_axis=True):
    points = [[0, 0, 0], [-cam_width, -cam_height, f], [cam_width, -cam_height, f],
              [cam_width, cam_height, f], [-cam_width, cam_height, f]]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    colors = [color for i in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.transform(c2w)

    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
        axis.scale(min(cam_width, cam_height), np.array([0., 0., 0.]))
        axis.transform(c2w)
        return [line_set, axis]
    else:
        return [line_set]


def interactive_pcd_viewer(pts_raw, colors_raw, conf_raw,
                           pts_raw_0=None, colors_raw_0=None, conf_raw_0=None,
                           c2w=None, c2w_0=None,
                           images=None, pts_full=None, conf_full=None,
                           initial_conf_thresh=0.0,
                           edge_normal_threshold=5.0,
                           edge_depth_threshold=0.008):
    """
    Interactive Open3D viewer with keyboard controls:
      [Right]/[Left] : increase/decrease confidence threshold by 0.01 (Right=Increase, Left=Decrease)
      [Up]/[Down]  : increase/decrease confidence threshold by 0.1
      [+]/[-]      : increase/decrease point size
      [B]          : toggle background color (Black/White)
      [E]          : toggle edge mask (get_pts_mask)
      [0]/[1]      : switch between pcd0 (initial) and pcd (refined)
      [Space]      : print current camera parameters
      [Q]/[Esc]    : close

    Args:
        pts_raw: (N, 3) flattened world points
        colors_raw: (N, 3) flattened RGB colors  
        conf_raw: (N,) confidence values (sigmoid-transformed)
        pts_raw_0 / colors_raw_0 / conf_raw_0: optional alternative point cloud
        c2w: (T, 4, 4) or list of poses for refined trajectory (pcd)
        c2w_0: (T, 4, 4) or list of poses for initial trajectory (pcd0)
        images: (B, T, C, H, W) tensor for sky mask (for get_pts_mask)
        pts_full: (B*T, H, W, 3) array for edge detection (for get_pts_mask)
        conf_full: (B*T, H, W) array for edge detection (for get_pts_mask)
        initial_conf_thresh: initial confidence threshold
        edge_normal_threshold: angle tolerance (deg)
        edge_depth_threshold: relative depth tolerance
    """

    # --- State ---
    class State:
        conf_thresh = initial_conf_thresh
        edge_mask_enabled = False
        showing_pcd0 = False
        edge_mask_flat = None       # for pcd
        edge_mask_flat_0 = None     # for pcd0
        point_size = 1.0
        bg_black = True
        camera_geometries = []      # list of current camera geometries

    state = State()
    pcd_handle = o3d.geometry.PointCloud()

    def _get_camera_geometries(poses, color):
        geoms = []
        if poses is None:
            return geoms
        # Handle (B, T, 4, 4) or (T, 4, 4)
        if hasattr(poses, 'shape') and len(poses.shape) == 4:
            poses = poses.reshape(-1, 4, 4)
        elif hasattr(poses, 'ndim') and poses.ndim == 4: # numpy
            poses = poses.reshape(-1, 4, 4)

        for pose in poses:
            geoms.extend(draw_camera(pose, color=color))
        return geoms

    def _build_pcd():
        """Rebuild point cloud geometry from current state."""
        if state.showing_pcd0 and pts_raw_0 is not None:
            pts, colors, conf = pts_raw_0, colors_raw_0, conf_raw_0
            edge_mask = state.edge_mask_flat_0
        else:
            pts, colors, conf = pts_raw, colors_raw, conf_raw
            edge_mask = state.edge_mask_flat

        mask = conf >= state.conf_thresh
        if state.edge_mask_enabled and edge_mask is not None:
            mask = mask & edge_mask

        if mask.sum() == 0:
            pcd_handle.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            pcd_handle.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        else:
            pcd_handle.points = o3d.utility.Vector3dVector(pts[mask].reshape(-1, 3))
            pcd_handle.colors = o3d.utility.Vector3dVector(colors[mask].reshape(-1, 3))

        n = len(pcd_handle.points)
        src = "pcd0" if state.showing_pcd0 else "pcd"
        edge_str = " [edge mask ON]" if state.edge_mask_enabled else ""
        print(f"  conf_thresh={state.conf_thresh:.4f} | {src} | {n:,} points{edge_str} | pt_size={state.point_size:.1f}")

    def _refresh(vis):
        _build_pcd()
        vis.update_geometry(pcd_handle)
        
        opt = vis.get_render_option()
        opt.point_size = state.point_size
        opt.background_color = np.array([0, 0, 0]) if state.bg_black else np.array([1, 1, 1])
        
        vis.update_renderer()

    def _update_camera_vis(vis):
        """Update visualized cameras based on current source (pcd0 vs pcd)."""
        # Remove old
        for g in state.camera_geometries:
            vis.remove_geometry(g, reset_bounding_box=False)
        
        # Determine new poses
        if state.showing_pcd0:
            poses = c2w_0
            color = [1, 0, 0] # Red for initial
        else:
            poses = c2w
            color = [1, 0, 0] # Green for refined
            
        # Add new
        new_geoms = _get_camera_geometries(poses, color)
        for g in new_geoms:
            vis.add_geometry(g, reset_bounding_box=False)
        state.camera_geometries = new_geoms

    def _compute_edge_mask():
        if pts_full is None or conf_full is None:
            print("Warning: pts_full and conf_full required for edge mask.")
            return

        from amb3r.tools.pts_vis import get_pts_mask
        print("Computing edge mask...")
        mask, _ = get_pts_mask(
            pts_full,
            images=images,
            conf=conf_full,
            conf_threshold=state.conf_thresh,
            edge_normal_threshold=edge_normal_threshold,
            edge_depth_threshold=edge_depth_threshold
        )
        state.edge_mask_flat = mask.reshape(-1)
        print("Edge mask computed.")

    # --- Key Callbacks ---
    def thresh_up_small(vis):
        state.conf_thresh = min(1.0, state.conf_thresh + 0.01)
        _refresh(vis)
        return False

    def thresh_down_small(vis):
        state.conf_thresh = max(0.0, state.conf_thresh - 0.01)
        _refresh(vis)
        return False

    def thresh_up_large(vis):
        state.conf_thresh = min(1.0, state.conf_thresh + 0.1)
        _refresh(vis)
        return False

    def thresh_down_large(vis):
        state.conf_thresh = max(0.0, state.conf_thresh - 0.1)
        _refresh(vis)
        return False

    def toggle_edge_mask(vis):
        state.edge_mask_enabled = not state.edge_mask_enabled
        if state.edge_mask_enabled and state.edge_mask_flat is None:
            _compute_edge_mask()
        _refresh(vis)
        return False

    def switch_to_pcd0(vis):
        if pts_raw_0 is not None:
            state.showing_pcd0 = True
            _refresh(vis)
            _update_camera_vis(vis)
        return False

    def switch_to_pcd1(vis):
        state.showing_pcd0 = False
        _refresh(vis)
        _update_camera_vis(vis)
        return False

    def pt_size_up(vis):
        state.point_size += 1.0
        _refresh(vis)
        return False

    def pt_size_down(vis):
        state.point_size = max(1.0, state.point_size - 1.0)
        _refresh(vis)
        return False

    def toggle_bg(vis):
        state.bg_black = not state.bg_black
        _refresh(vis)
        return False

    def print_camera(vis):
        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        print("\nCamera intrinsics:")
        print(params.intrinsic.intrinsic_matrix)
        print("Camera extrinsics:")
        print(params.extrinsic)
        return False

    # --- Build initial point cloud ---
    _build_pcd()

    # --- Visualizer setup ---
    vis = o3d.visualization.VisualizerWithKeyCallback()
    
    # Show controls in the window title since we can't do 2D overlays easily in legacy mode
    instructions = "Controls: [Right/Left] Conf +/-0.01 | [Up/Down] Conf +/-0.1 | [+/-] Pt Size | [B] BG Color | [E] Edge Mask | [0/1] Switch PCD"
    vis.create_window(window_name=instructions, width=1920, height=1080)
    vis.add_geometry(pcd_handle)
    
    # Initial camera geometries
    _update_camera_vis(vis)

    opt = vis.get_render_option()
    opt.point_size = state.point_size
    opt.background_color = np.array([0, 0, 0])

    # Register key callbacks (ASCII codes)
    vis.register_key_callback(263, thresh_up_small)      # Right arrow
    vis.register_key_callback(262, thresh_down_small)    # Left arrow
    vis.register_key_callback(265, thresh_up_large)      # Up arrow
    vis.register_key_callback(264, thresh_down_large)    # Down arrow
    
    vis.register_key_callback(ord('='), pt_size_up)      # +/= key
    vis.register_key_callback(ord('+'), pt_size_up)      # + (numpad or shift)
    vis.register_key_callback(ord('-'), pt_size_down)    # -/- key
    vis.register_key_callback(ord('_'), pt_size_down)    # _ (shift -)

    vis.register_key_callback(ord('B'), toggle_bg)       # B key
    vis.register_key_callback(ord('b'), toggle_bg)       # b key

    vis.register_key_callback(ord('E'), toggle_edge_mask)    # E key
    vis.register_key_callback(ord('e'), toggle_edge_mask)    # e key
    
    vis.register_key_callback(ord('0'), switch_to_pcd0)      # 0 key
    vis.register_key_callback(ord('1'), switch_to_pcd1)      # 1 key
    vis.register_key_callback(32, print_camera)              # Spacebar

    print("\n--- Interactive PCD Viewer Controls ---")
    print("  [Right/Left]: confidence threshold ±0.01 (Right=+, Left=-)")
    print("  [Up/Down]   : confidence threshold ±0.1")
    print("  [+/-]       : point size increase/decrease")
    print("  [B]         : toggle background color (Black/White)")
    print("  [E]         : toggle edge mask (get_pts_mask)")
    print("  [0/1]       : switch pcd0 (initial) / pcd (refined)")
    print("  [Space]     : print camera parameters")
    print("  [Q / Esc]   : close")
    print("---------------------------------------\n")

    vis.run()
    vis.destroy_window()
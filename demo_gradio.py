import os
import sys
import glob
import time
import threading
import queue
import uuid
import gc
import shutil
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import gradio as gr
import gradio_client.utils as grc_utils
import numpy as np
import torch

from loop_utils.config_utils import load_config
from loop_utils.sim3utils import apply_sim3_direct
from vggt_long import VGGT_Long
from vggt_long_inc import VGGT_Long_Inc

# Make sure local VGGT package is discoverable
root_dir = os.path.dirname(os.path.abspath(__file__))
base_models_dir = os.path.join(root_dir, "base_models")
if root_dir not in sys.path:
    sys.path.append(root_dir)
if base_models_dir not in sys.path:
    sys.path.append(base_models_dir)

from loop_utils.visual_util import predictions_to_glb  # noqa: E402

def warmup_alignment(config):
    if config['Model']['align_lib'] == 'numba':
        from loop_utils.sim3utils import warmup_numba
        warmup_numba()
    if config['Model']['align_lib'] == 'triton':
        from loop_utils.alignment_triton import warmup_triton
        warmup_triton()
    if config['Model']['align_lib'] == 'torch':
        from loop_utils.alignment_torch import warmup_torch
        warmup_torch()

# -------------------------------------------------------------------------
# Patch gradio_client json schema helper to tolerate boolean schemas
# (gradio_client 1.5.x can emit `additionalProperties: False`, which
# triggers a TypeError in json_schema_to_python_type)
# -------------------------------------------------------------------------
_orig_get_type = grc_utils.get_type


def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "boolean" if schema else "false"
    return _orig_get_type(schema)


grc_utils.get_type = _safe_get_type


# -------------------------------------------------------------------------
# Runtime / config
# -------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "base_config.yaml")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TEMP_UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)
INCREMENTAL_SESSIONS = {}


# -------------------------------------------------------------------------
# Incremental helpers
# -------------------------------------------------------------------------
def build_partial_predictions_from_cache(save_dir: str) -> Optional[dict]:
    """
    Stitch already-written chunk results for incremental visualization.
    Mixes aligned and unaligned chunks to show latest progress.
    """
    aligned_dir = os.path.join(save_dir, "_tmp_results_aligned")
    unaligned_dir = os.path.join(save_dir, "_tmp_results_unaligned")

    # Find all chunks in unaligned (it's the superset)
    unaligned_files = sorted(glob.glob(os.path.join(unaligned_dir, "chunk_*.npy")))
    if not unaligned_files:
        return None

    # Get max chunk index
    max_chunk_idx = -1
    for f in unaligned_files:
        try:
            # chunk_0.npy -> 0
            idx = int(os.path.basename(f).split('_')[1].split('.')[0])
            max_chunk_idx = max(max_chunk_idx, idx)
        except:
            pass
            
    if max_chunk_idx < 0:
        return None

    all_wp: List[np.ndarray] = []
    all_wpc: List[np.ndarray] = []
    all_images: List[np.ndarray] = []
    all_extrinsic: List[np.ndarray] = []
    all_intrinsic: List[np.ndarray] = []
    all_depth: List[np.ndarray] = []
    all_depth_conf: List[np.ndarray] = []

    chunks_used = 0
    # Iterate all potential chunks
    for idx in range(max_chunk_idx + 1):
        aligned_path = os.path.join(aligned_dir, f"chunk_{idx}.npy")
        unaligned_path = os.path.join(unaligned_dir, f"chunk_{idx}.npy")
        
        # Prefer aligned, fallback to unaligned
        if os.path.exists(aligned_path):
            path = aligned_path
        elif os.path.exists(unaligned_path):
            path = unaligned_path
        else:
            continue # Should not happen if indices are contiguous

        try:
            data = np.load(path, allow_pickle=True).item()
        except:
            continue

        chunks_used += 1
        wp = data["world_points"]
        wpc = data.get("world_points_conf", np.ones(wp.shape[:-1], dtype=np.float32))
        imgs = data["images"]
        if imgs.ndim == 4 and imgs.shape[1] == 3:
            imgs = np.transpose(imgs, (0, 2, 3, 1))
        extr = data["extrinsic"]
        intr = data["intrinsic"]
        depth = data.get("depth")
        depth_conf = data.get("depth_conf")

        all_wp.append(wp)
        all_wpc.append(wpc)
        all_images.append(imgs)
        all_extrinsic.append(extr)
        all_intrinsic.append(intr)
        if depth is not None:
            all_depth.append(depth)
        if depth_conf is not None:
            all_depth_conf.append(depth_conf)

    if not all_wp:
        return None

    predictions = {
        "world_points": np.concatenate(all_wp, axis=0),
        "world_points_conf": np.concatenate(all_wpc, axis=0),
        "images": np.concatenate(all_images, axis=0),
        "extrinsic": np.concatenate(all_extrinsic, axis=0),
    }

    # Handle intrinsics carefully (can be None for some models)
    if all(x is not None for x in all_intrinsic):
        predictions["intrinsic"] = np.concatenate(all_intrinsic, axis=0)
    else:
        predictions["intrinsic"] = None

    if all_depth:
        predictions["depth"] = np.concatenate(all_depth, axis=0)
    if all_depth_conf:
        predictions["depth_conf"] = np.concatenate(all_depth_conf, axis=0)

    predictions["_chunk_count"] = chunks_used
    return predictions


def build_glb_from_cache(
    save_dir: str,
    conf_thres: float,
    frame_filter: str,
    mask_black_bg: bool,
    mask_white_bg: bool,
    show_cam: bool,
    mask_sky: bool,
    prediction_mode: str,
):
    predictions = build_partial_predictions_from_cache(save_dir)
    if predictions is None:
        return None, "No cached predictions yet."
    glbfile = os.path.join(
        save_dir,
        f"inc_glb_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}"
        f"_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=save_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)
    chunk_count = predictions.get("_chunk_count", 0)
    log = f"Merged {chunk_count} chunk(s)."
    return glbfile, log


def _run_incremental_worker(
    session_id: str,
    target_dir: str,
    model_name: str,
    sequence_name: str,
    config_path: str,
    resume: bool,
    update_trigger: Optional[queue.Queue] = None,
) -> None:
    """
    Background worker: uses VGGT_Long_Inc. Callbacks push events to update_trigger queue.
    """
    session = INCREMENTAL_SESSIONS.get(session_id)
    if session is None:
        return
    try:
        config = load_config(config_path)
        config["Model"]["delete_temp_files"] = False
        config['Weights']['model'] = model_name

        warmup_alignment(config)

        image_dir = os.path.join(target_dir, "images")
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if resume and session.get("save_dir"):
            save_dir = session["save_dir"]
            os.makedirs(save_dir, exist_ok=True)
        else:
            # results/Model/Images/Timestamp
            folder_name = sequence_name.strip() if sequence_name and sequence_name.strip() else target_dir.split('/')[-1]
            save_dir = os.path.join(
                RESULTS_DIR, 
                model_name,
                folder_name,  # Extract input_images_... folder name
                timestamp
            )
            os.makedirs(save_dir, exist_ok=True)

        session["save_dir"] = save_dir
        session["status"] = "running"

        # Copy images to save_dir/images (incremental: sync new uploads)
        dest_image_dir = os.path.join(save_dir, "images")
        if os.path.abspath(image_dir) != os.path.abspath(dest_image_dir):
            if TEMP_UPLOADS_DIR in os.path.abspath(image_dir):
                # Move contents from temp image_dir to dest_image_dir
                os.makedirs(dest_image_dir, exist_ok=True)
                for item in os.listdir(image_dir):
                    s = os.path.join(image_dir, item)
                    d = os.path.join(dest_image_dir, item)
                    if os.path.exists(d):
                        continue # Skip duplicates
                    shutil.move(s, d)
                print(f"Moved incremental images from temp to {dest_image_dir}")
                # Try to remove the temp image dir and its parent
                try:
                    os.rmdir(image_dir)
                    os.rmdir(os.path.dirname(image_dir))
                except:
                    pass
            else:
                shutil.copytree(image_dir, dest_image_dir, dirs_exist_ok=True)
                print(f"Synced images to {dest_image_dir}")
            image_dir = dest_image_dir

        def on_update(*args):
            if update_trigger:
                update_trigger.put("update")

        # Reuse retrieval object if it exists in session
        retrieval_instance = session.get("retrieval_instance")

        vlong = VGGT_Long_Inc(
            image_dir=image_dir,
            save_dir=save_dir,
            config=config,
            on_chunk_aligned=on_update,
            on_all_aligned=on_update,
            retrieval_instance=retrieval_instance,
        )
        
        # Store retrieval instance for next run
        if hasattr(vlong, "retrieval"):
            session["retrieval_instance"] = vlong.retrieval

        vlong.run(resume=resume)
        session["status"] = "done"
    except Exception as e:  # noqa: BLE001
        session["status"] = "error"
        session["error"] = str(e)
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def start_incremental_session(input_video, input_images, model_name, sequence_name, prev_dir, prev_session_id):
    """
    Start a fresh incremental session: copy inputs, create session, spawn backend worker.
    """
    if prev_session_id and prev_session_id in INCREMENTAL_SESSIONS:
        return (
            None,
            prev_dir or "None",
            None,
            "An incremental session is active. Please click 'End and Clean Cache' first.",
            prev_session_id,
        )
    target_dir, image_paths = handle_uploads(input_video, input_images, prev_dir=None)

    # Generate new session id
    session_id = str(uuid.uuid4())
    event_queue = queue.Queue()
    INCREMENTAL_SESSIONS[session_id] = {
        "target_dir": target_dir,
        "save_dir": None,
        "status": "init",
        "error": None,
        "images": image_paths,
        "queue": event_queue,
        "retrieval_instance": None,  # Store persistent loop detector here
        "model_name": model_name,
        "sequence_name": sequence_name,
    }

    worker = threading.Thread(
        target=_run_incremental_worker,
        args=(session_id, target_dir, model_name, sequence_name, DEFAULT_CONFIG, False, event_queue),
        daemon=True,
    )
    INCREMENTAL_SESSIONS[session_id]["thread"] = worker
    worker.start()

    log = f"Incremental reconstruction started ({model_name}). Assets: {target_dir}. Cache stays until you clean it."
    return None, target_dir, image_paths, log, session_id


def continue_incremental_session(input_video, input_images, target_dir, session_id):
    """
    Continue an existing incremental session: append uploads and rerun with existing cache (resume=True).
    """
    if not session_id or session_id not in INCREMENTAL_SESSIONS:
        return (
            None,
            target_dir or "None",
            None,
            "No incremental session. Please click 'Start Incremental Reconstruction' first.",
            session_id,
        )
    session = INCREMENTAL_SESSIONS[session_id]
    if session.get("status") == "running":
        return (
            None,
            session.get("target_dir", "None"),
            None,
            "Background is still running; please wait.",
            session_id,
        )

    tgt_dir = session.get("target_dir") or target_dir
    model_name = session.get("model_name", "VGGT") # Fallback if missing
    sequence_name = session.get("sequence_name", "")
    if not tgt_dir:
        return None, "None", None, "Session is missing asset directory. Please start again.", session_id

    # Append new uploads without clearing old ones
    image_paths = append_uploads(tgt_dir, input_video, input_images)
    session["images"] = image_paths

    # reuse existing queue if present, else create new
    if "queue" not in session:
        session["queue"] = queue.Queue()
    event_queue = session["queue"]

    worker = threading.Thread(
        target=_run_incremental_worker,
        args=(session_id, tgt_dir, model_name, sequence_name, DEFAULT_CONFIG, True, event_queue),
        daemon=True,
    )
    session["thread"] = worker
    worker.start()

    log = f"Continue reconstruction started ({model_name}). Assets: {tgt_dir}. Reusing existing cache."
    return None, tgt_dir, image_paths, log, session_id


def check_for_updates(
    session_id: str,
    target_dir: str,
    conf_thres: float,
    frame_filter: str,
    mask_black_bg: bool,
    mask_white_bg: bool,
    show_cam: bool,
    mask_sky: bool,
    prediction_mode: str,
):
    """
    Called by gr.Timer periodically. Checks if worker pushed an update event.
    If yes, refreshes visualization. If no, yields nothing (no update).
    """
    if not session_id or session_id not in INCREMENTAL_SESSIONS:
        # Stop timer or yield nothing
        return gr.update()

    session = INCREMENTAL_SESSIONS[session_id]
    q = session.get("queue")
    save_dir = session.get("save_dir")

    # If worker error, show it once
    if session.get("error"):
        return gr.update(), f"Error: {session['error']}"

    # Check if there is a pending update signal
    # Consume all pending signals to just update once
    has_update = False
    if q:
        while not q.empty():
            q.get_nowait()
            has_update = True

    if has_update and save_dir:
        glb, log_msg = build_glb_from_cache(
            save_dir,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        )
        final_log = f"{log_msg} (Status: {session.get('status')})"
        return glb, final_log

    return gr.update(), gr.update()


def manual_refresh_incremental_view(
    session_id: Optional[str],
    target_dir: str,
    conf_thres: float,
    frame_filter: str,
    mask_black_bg: bool,
    mask_white_bg: bool,
    show_cam: bool,
    mask_sky: bool,
    prediction_mode: str,
):
    """
    Manual trigger to force visualization refresh from cache.
    """
    if not session_id or session_id not in INCREMENTAL_SESSIONS:
        return None, "No incremental session."
    session = INCREMENTAL_SESSIONS[session_id]
    save_dir = session.get("save_dir")
    if not save_dir:
        return None, "Initializing..."
    
    glb, log_msg = build_glb_from_cache(
        save_dir,
        conf_thres,
        frame_filter,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        mask_sky,
        prediction_mode,
    )
    final_log = f"{log_msg} (Manual refresh. Status: {session.get('status')})"
    return glb, final_log


def show_cache_info(session_id: Optional[str]):
    if not session_id or session_id not in INCREMENTAL_SESSIONS:
        return "No incremental session."
    session = INCREMENTAL_SESSIONS[session_id]
    return (
        f"Asset dir: {session.get('target_dir', 'N/A')}\n"
        f"Cache dir: {session.get('save_dir', 'Not created yet (initializing)')}\n"
        "Click 'End and Clean Cache' when you are done."
    )


def cleanup_incremental_session(session_id: Optional[str], target_dir: str):
    """
    User-triggered teardown: delete INPUT images only, keep reconstruction results.
    """
    if not session_id or session_id not in INCREMENTAL_SESSIONS:
        return None, "None", None, "No incremental session to clean.", None

    session = INCREMENTAL_SESSIONS.pop(session_id, {})
    tgt_dir = session.get("target_dir")
    save_dir = session.get("save_dir")
    
    # Also clean up persistent retrieval object (processes)
    retrieval = session.get("retrieval_instance")
    if retrieval and hasattr(retrieval, "close"):
        try:
            retrieval.close()
        except:
            pass

    # Delete input images folder to save space, but keep the parent dir with results
    # (Since we moved images to save_dir/images, we should clean that if user wants to save space)
    # The original temp 'target_dir' might be empty now or already gone.
    
    if tgt_dir and os.path.isdir(os.path.join(tgt_dir, "images")):
        shutil.rmtree(os.path.join(tgt_dir, "images"), ignore_errors=True)
        
    # Also clean the images in the result directory per user request
    if save_dir and os.path.isdir(os.path.join(save_dir, "images")):
        shutil.rmtree(os.path.join(save_dir, "images"), ignore_errors=True)

    # Delete the entire temp_uploads directory
    if os.path.exists(TEMP_UPLOADS_DIR):
        shutil.rmtree(TEMP_UPLOADS_DIR, ignore_errors=True)
        # Recreate it to be ready for next usage
        os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)

    # Clean up generated glb files in save_dir if exists
    if save_dir and os.path.isdir(save_dir):
        for f in glob.glob(os.path.join(save_dir, "*.glb")):
            try:
                os.remove(f)
            except:
                pass

    return None, "None", None, "Input images removed. Reconstruction results preserved.", None


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images, prev_dir: Optional[str] = None) -> Tuple[str, List[str]]:
    """
    Create a new 'target_dir/images' subfolder, place user uploads, and clear previous temp dir.
    Videos are sampled at ~5 FPS.
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    if prev_dir and os.path.isdir(prev_dir):
        shutil.rmtree(prev_dir, ignore_errors=True)

    # Use uploaded folder name or video name if possible, else fallback to timestamp
    folder_name = f"input_images_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    if input_images and len(input_images) > 0:
        # Try to get common parent folder name or first image parent folder
        first_img_path = input_images[0]["name"] if isinstance(input_images[0], dict) else input_images[0]
        parent_dir = os.path.basename(os.path.dirname(first_img_path))
        if parent_dir and parent_dir != "images" and parent_dir != "temp": # Simple check to avoid generic names
             folder_name = parent_dir
    elif input_video:
        video_path = input_video["name"] if isinstance(input_video, dict) else input_video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        folder_name = video_name

    target_dir = os.path.join(TEMP_UPLOADS_DIR, folder_name)
    target_dir_images = os.path.join(target_dir, "images")
    os.makedirs(target_dir_images, exist_ok=True)

    image_paths: List[str] = []

    if input_images is not None:
        for file_data in input_images:
            file_path = file_data["name"] if isinstance(file_data, dict) else file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    if input_video is not None:
        video_path = input_video["name"] if isinstance(input_video, dict) else input_video
        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = max(int(round(fps / 5.0)), 1) if fps and fps > 0 else 1

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1
            count += 1

    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


def append_uploads(target_dir: str, input_video, input_images) -> List[str]:
    """
    Append uploads into an existing target_dir/images without clearing previous files.
    """
    if not target_dir:
        raise ValueError("target_dir is required for append_uploads")

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    target_dir_images = os.path.join(target_dir, "images")
    os.makedirs(target_dir_images, exist_ok=True)

    image_paths: List[str] = sorted(
        glob.glob(os.path.join(target_dir_images, "*.png"))
        + glob.glob(os.path.join(target_dir_images, "*.jpg"))
        + glob.glob(os.path.join(target_dir_images, "*.jpeg"))
    )

    def unique_path(base_name: str) -> str:
        dst = os.path.join(target_dir_images, base_name)
        if not os.path.exists(dst):
            return dst
        stem, ext = os.path.splitext(base_name)
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return os.path.join(target_dir_images, f"{stem}_{suffix}{ext}")

    if input_images is not None:
        for file_data in input_images:
            file_path = file_data["name"] if isinstance(file_data, dict) else file_data
            base = os.path.basename(file_path)
            dst_path = unique_path(base)
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    if input_video is not None:
        video_path = input_video["name"] if isinstance(input_video, dict) else input_video
        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = max(int(round(fps / 5.0)), 1) if fps and fps > 0 else 1

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            if count % frame_interval == 0:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                image_path = os.path.join(target_dir_images, f"{stamp}_{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1
            count += 1

    image_paths = sorted(image_paths)
    end_time = time.time()
    print(f"Files appended to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_video, input_images, current_dir):
    """
    Whenever user uploads or changes files, clear previous temp dir, re-import files, and show gallery.
    """
    if not input_video and not input_images:
        return None, None, None, "Please upload data first."
    target_dir, image_paths = handle_uploads(input_video, input_images, prev_dir=current_dir)
    return None, target_dir, image_paths, "Upload complete. Ready to start reconstruction."


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    theme = gr.themes.Ocean()
    theme.set(
        checkbox_label_background_fill_selected="*button_primary_background_fill",
        checkbox_label_text_color_selected="*button_primary_text_color",
    )

    with gr.Blocks(
        theme=theme,
        css="""
        .custom-log * {
            font-style: italic;
            font-size: 22px !important;
            background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
            -webkit-background-clip: text;
            background-clip: text;
            font-weight: bold !important;
            color: transparent !important;
            text-align: center !important;
        }

        .example-log * {
            font-style: italic;
            font-size: 16px !important;
            background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent !important;
        }

        #my_radio .wrap {
            display: flex;
            flex-wrap: nowrap;
            justify-content: center;
            align-items: center;
        }

        #my_radio .wrap label {
            display: flex;
            width: 50%;
            justify-content: center;
            align-items: center;
            margin: 0;
            padding: 10px 0;
            box-sizing: border-box;
        }

        .upload-images-box {
            max-height: 220px;
            overflow-y: auto;
        }
        """,
    ) as demo:
        gr.HTML(
            """
            <h1>🏛️ VGGT-Long: Chunk it, Loop it, Align it, Pushing VGGT's Limits on Kilometer-scale Long RGB Sequences</h1>
            <p><a href="https://github.com/DengKaiCQ/VGGT-Long">🐙 GitHub Repository</a></p>

            <div style="font-size: 16px; line-height: 1.5;">
            <p>Upload a video (sampled at 5 FPS) or a set of images to create a 3D reconstruction. VGGT-Long splits long sequences into chunks, aligns them, and renders a colored point cloud plus camera frustums.</p>

            <h3>Getting Started:</h3>
            <ol>
                <li><strong>Upload Your Data:</strong> Use the "Upload Video" or "Upload Images" buttons on the left. Videos are sampled at ~5 FPS.</li>
                <li><strong>Preview:</strong> Your uploaded images will appear in the gallery on the left.</li>
                <li><strong>Start Reconstruction:</strong> Click the "Start Incremental Reconstruction" button to start the incremental 3D reconstruction process.</li>
                <li><strong>Continue Reconstruction:</strong> Make sure to clear the previous upload records before the next images/video upload(you can click the cross icon in the upper-right corner of the upload box to do so). Then click the "Continue Reconstruction" button to continue the incremental 3D reconstruction process based on previous uploads.</li>
                <li><strong>Visualize:</strong> The 3D reconstruction will appear in the viewer on the right. You can rotate, pan, and zoom to explore the model, and download the GLB file. Note the visualization of 3D points may be slow for a large number of input images.</li>
                <li>
                <strong>Adjust Visualization (Optional):</strong>
                After reconstruction, you can fine-tune the visualization using the options below
                <details style="display:inline;">
                    <summary style="display:inline;">(<strong>click to expand</strong>):</summary>
                    <ul>
                    <li><em>Confidence Threshold:</em> Adjust the filtering of points based on confidence.</li>
                    <li><em>Show Points from Frame:</em> Select specific frames to display in the point cloud.</li>
                    <li><em>Show Camera:</em> Toggle the display of estimated camera positions.</li>
                    <li><em>Filter Sky / Filter Black Background:</em> Remove sky or black-background points.</li>
                    <li><em>Select a Prediction Mode:</em> Choose between "Predicted Pointmap" or "Depthmap and Camera Branch."</li>
                    </ul>
                </details>
                </li>
                <li><strong>End Reconstruction:</strong> If you want to finish this incremental 3D reconstruction process, click the "End and Clean Cache" button to clear all caches from this process while preserving the reconstruction results. By the way, you can click the "Show Cache Paths" button at any time to view the location where the caches and results are saved.</li>
            </ol>
            <p><strong style="color: #0ea5e9;">Note:</strong> Visualization of dense point clouds may take time. If localhost is blocked, set <code>GRADIO_SHARE=1</code> before launching.</p>
            </div>
            """
        )

        inc_session_state = gr.State(value=None)
        target_dir_output = gr.Textbox(label="Preview Target Dir", visible=False, value="None")
        target_dir_inc = gr.Textbox(label="Incremental Target Dir", visible=False, value="None")

        with gr.Row():
            with gr.Column(scale=2):
                input_video = gr.Video(label="Upload Video", interactive=True)
                input_images = gr.File(
                    file_count="multiple",
                    label="Upload Images",
                    interactive=True,
                    height=160,
                    elem_classes=["upload-images-box"],
                )
                
                # Model selection
                model_selector = gr.Dropdown(
                    choices=["VGGT", "Pi3", "Mapanything", "DA3"],
                    value="VGGT",
                    label="Select Model",
                    interactive=True,
                )
                
                sequence_name = gr.Textbox(
                    label="Sequence Name (Optional)",
                    placeholder="e.g. kitti00. If empty, uses temp folder name.",
                    interactive=True
                )
                
                image_gallery = gr.Gallery(
                    label="Preview",
                    columns=4,
                    height="240px",
                    show_download_button=True,
                    object_fit="contain",
                    preview=True,
                )
            with gr.Column(scale=4):
                log_inc = gr.Markdown("Click 'Start Incremental Reconstruction' to begin. Cache is kept until you clean it.", elem_classes=["custom-log"])
                reconstruction_inc = gr.Model3D(height=460, zoom_speed=0.5, pan_speed=0.5)
                with gr.Row():
                    start_inc_btn = gr.Button("Start Incremental Reconstruction", variant="primary")
                    continue_inc_btn = gr.Button("Continue Reconstruction", variant="secondary")
                with gr.Row():
                    show_cache_btn = gr.Button("Show Cache Paths")
                    cleanup_inc_btn = gr.Button("End and Clean Cache", variant="stop")
                with gr.Row():
                    conf_thres_inc = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                    frame_filter_inc = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                with gr.Row():
                    show_cam_inc = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky_inc = gr.Checkbox(label="Filter Sky", value=False)
                    mask_black_inc = gr.Checkbox(label="Filter Black Background", value=False)
                    mask_white_inc = gr.Checkbox(label="Filter White Background", value=False)
                prediction_mode_inc = gr.Radio(
                    ["Predicted Pointmap", "Depthmap and Camera Branch"],
                    label="Select a Prediction Mode",
                    value="Predicted Pointmap",
                    scale=1,
                    elem_id="my_radio"
                )

        # Event wiring
        input_video.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images, target_dir_output],
            outputs=[reconstruction_inc, target_dir_output, image_gallery, log_inc],
        )
        input_images.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images, target_dir_output],
            outputs=[reconstruction_inc, target_dir_output, image_gallery, log_inc],
        )

        # Buttons trigger backend worker
        start_inc_btn.click(
            fn=start_incremental_session,
            inputs=[input_video, input_images, model_selector, sequence_name, target_dir_inc, inc_session_state],
            outputs=[reconstruction_inc, target_dir_inc, image_gallery, log_inc, inc_session_state],
        )
        continue_inc_btn.click(
            fn=continue_incremental_session,
            inputs=[input_video, input_images, target_dir_inc, inc_session_state],
            outputs=[reconstruction_inc, target_dir_inc, image_gallery, log_inc, inc_session_state],
        )

        # Use a short-interval Timer to poll for event-based updates
        refresh_timer = gr.Timer(value=1.0)
        refresh_timer.tick(
            fn=check_for_updates,
            inputs=[
                inc_session_state,
                target_dir_inc,
                conf_thres_inc,
                frame_filter_inc,
                mask_black_inc,
                mask_white_inc,
                show_cam_inc,
                mask_sky_inc,
                prediction_mode_inc,
            ],
            outputs=[reconstruction_inc, log_inc],
        )

        show_cache_btn.click(
            fn=show_cache_info,
            inputs=[inc_session_state],
            outputs=[log_inc],
        )

        cleanup_inc_btn.click(
            fn=cleanup_incremental_session,
            inputs=[inc_session_state, target_dir_inc],
            outputs=[reconstruction_inc, target_dir_inc, image_gallery, log_inc, inc_session_state],
        )

        for ctrl in [
            conf_thres_inc,
            frame_filter_inc,
            mask_black_inc,
            mask_white_inc,
            show_cam_inc,
            mask_sky_inc,
            prediction_mode_inc,
        ]:
            ctrl.change(
                manual_refresh_incremental_view,
                [
                    inc_session_state,
                    target_dir_inc,
                    conf_thres_inc,
                    frame_filter_inc,
                    mask_black_inc,
                    mask_white_inc,
                    show_cam_inc,
                    mask_sky_inc,
                    prediction_mode_inc,
                ],
                [reconstruction_inc, log_inc],
            )

    return demo


demo = build_demo()


def launch(server_name: str = None, server_port: int = None, share: bool = None) -> None:
    """
    Launch the Gradio app. Environment variable overrides:
      GRADIO_SERVER_NAME (default: 0.0.0.0)
      GRADIO_SERVER_PORT (default: 8080)
      GRADIO_SHARE       (default: True if unset; set 0/false to disable)
    """
    if server_name is None:
        server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    if server_port is None:
        server_port = int(os.environ.get("GRADIO_SERVER_PORT", "8080"))
    if share is None:
        # Default to share=True to avoid "localhost not accessible" errors in restricted envs.
        share_env = os.environ.get("GRADIO_SHARE", "1")
        share = share_env in ("1", "true", "True")

    demo.queue(max_size=20).launch(
        show_error=True,
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_api=False,  # avoid schema generation issues in some gradio_client versions
    )


if __name__ == "__main__":
    launch()

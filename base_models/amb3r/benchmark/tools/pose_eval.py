# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/pose_eval.py

import os
import sys
import evo
import evo.tools.plot
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'thirdparty'))

from vggt.utils.rotation import mat_to_quat
from vggt.utils.geometry import closed_form_inverse_se3
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from evo.core import metrics
from evo.core.trajectory import PosePath3D

def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False, plot=True):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est_aligned = PosePath3D(poses_se3=poses_est)
    r_a, t_a, s = traj_est_aligned.align(traj_ref, correct_scale=monocular)


    info = {
        'r': r_a,
        't': t_a,
        's': s
    }
    
    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()

    print("RMSE ATE \[m]", ape_stat)

    if not plot:
        return ape_stat

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)

    return ape_stat, info


def build_pair_index(N, B=1):
    """
    Build indices for all possible pairs of frames.

    Args:
        N: Number of frames
        B: Batch size

    Returns:
        i1, i2: Indices for all possible pairs
    """
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    """
    Calculate rotation angle error between ground truth and predicted rotations.

    Args:
        rot_gt: Ground truth rotation matrices
        rot_pred: Predicted rotation matrices
        batch_size: Batch size for reshaping the result
        eps: Small value to avoid numerical issues

    Returns:
        Rotation angle error in degrees
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    """
    Calculate translation angle error between ground truth and predicted translations.

    Args:
        tvec_gt: Ground truth translation vectors
        tvec_pred: Predicted translation vectors
        batch_size: Batch size for reshaping the result
        ambiguity: Whether to handle direction ambiguity

    Returns:
        Translation angle error in degrees
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """
    Normalize the translation vectors and compute the angle between them.

    Args:
        t_gt: Ground truth translation vectors
        t: Predicted translation vectors
        eps: Small value to avoid division by zero
        default_err: Default error value for invalid cases

    Returns:
        Angular error between translation vectors in radians
    """
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using NumPy.

    Args:
        r_error: numpy array representing R error values (Degree)
        t_error: numpy array representing T error values (Degree)
        max_threshold: Maximum threshold value for binning the histogram

    Returns:
        AUC value and the normalized histogram
    """
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """
    Compute rotation and translation errors between predicted and ground truth poses.
    This function assumes the input poses are world-to-camera (w2c) transformations.

    Args:
        pred_se3: Predicted SE(3) transformations (w2c), shape (N, 4, 4)
        gt_se3: Ground truth SE(3) transformations (w2c), shape (N, 4, 4)
        num_frames: Number of frames (N)

    Returns:
        Rotation and translation angle errors in degrees
    """
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    relative_pose_gt = gt_se3[pair_idx_i1].bmm(
        closed_form_inverse_se3(gt_se3[pair_idx_i2])
    )
    relative_pose_pred = pred_se3[pair_idx_i1].bmm(
        closed_form_inverse_se3(pred_se3[pair_idx_i2])
    )

    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )

    return rel_rangle_deg, rel_tangle_deg



def get_results_from_pose_enc(pose_enc, gt_extrinsic, RRA_thresholds = [5, 15, 30], RTA_thresholds = [5, 15, 30]):
    """
    Convert pose encoding to camera extrinsics and intrinsics.
    """
       
    
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
    pose_enc, image_size_hw=(518, 392), pose_encoding_type="absT_quaR_FoV", build_intrinsics=True  # e.g., (256, 512)
    )

    # extrinsic: Bs, T, 3, 4 -> Bs, T, 4, 4
    if extrinsic.shape[-2] != 4:
        extrinsic = torch.cat([extrinsic, torch.tensor([0, 0, 0, 1], device=extrinsic.device).view(1, 1, 1, 4).expand(extrinsic.shape[0], extrinsic.shape[1], 1, 4)], dim=2)
    
    if gt_extrinsic.shape[-2] != 4:
        gt_extrinsic = torch.cat([gt_extrinsic, torch.tensor([0, 0, 0, 1], device=extrinsic.device).view(1, 1, 1, 4).expand(gt_extrinsic.shape[0], gt_extrinsic.shape[1], 1, 4)], dim=2)
    
    extrinsic = extrinsic.squeeze(0)
    gt_extrinsic = gt_extrinsic.squeeze(0)
    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(extrinsic, gt_extrinsic, len(extrinsic))


    results = {}

    for tau in RRA_thresholds:
        results[f"RRA_at_{tau}"] = (rel_rangle_deg < tau).float().mean().item()

    for tau in RTA_thresholds:
        results[f"RTA_at_{tau}"] = (rel_tangle_deg < tau).float().mean().item()

    results['mAA_30'] = calculate_auc_np(rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy(), max_threshold=30)[0]
    return results, extrinsic, gt_extrinsic



def get_results_from_camera_pose(pose, gt_pose, RRA_thresholds = [5, 15, 30], RTA_thresholds = [5, 15, 30]):
    """
    Convert pose encoding to camera extrinsics and intrinsics.
    """
    
    extrinsic = closed_form_inverse_se3(pose)
    gt_extrinsic = closed_form_inverse_se3(gt_pose)

    # extrinsic: Bs, T, 3, 4 -> Bs, T, 4, 4
    if extrinsic.shape[-2] != 4:
        extrinsic = torch.cat([extrinsic, torch.tensor([0, 0, 0, 1], device=extrinsic.device).view(1, 1, 1, 4).expand(extrinsic.shape[0], extrinsic.shape[1], 1, 4)], dim=2)
    
    if gt_extrinsic.shape[-2] != 4:
        gt_extrinsic = torch.cat([gt_extrinsic, torch.tensor([0, 0, 0, 1], device=extrinsic.device).view(1, 1, 1, 4).expand(gt_extrinsic.shape[0], gt_extrinsic.shape[1], 1, 4)], dim=2)
    
    extrinsic = extrinsic.squeeze(0)
    gt_extrinsic = gt_extrinsic.squeeze(0)
    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(extrinsic, gt_extrinsic, len(extrinsic))


    results = {}

    for tau in RRA_thresholds:
        results[f"RRA_at_{tau}"] = (rel_rangle_deg < tau).float().mean().item()

    for tau in RTA_thresholds:
        results[f"RTA_at_{tau}"] = (rel_tangle_deg < tau).float().mean().item()

    results['mAA_30'] = calculate_auc_np(rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy(), max_threshold=30)[0]
    return results, extrinsic, gt_extrinsic


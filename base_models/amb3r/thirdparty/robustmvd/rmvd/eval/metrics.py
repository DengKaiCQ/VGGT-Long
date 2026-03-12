# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/eval/metrics.py

import numpy as np
import pandas as pd
from tqdm import tqdm


def valid_mean(arr, mask, axis=None, keepdims=np._NoValue):
    """Compute mean of elements across given dimensions of an array, considering only valid elements.

    Args:
        arr: The array to compute the mean.
        mask: Array with numerical or boolean values for element weights or validity. For bool, False means invalid.
        axis: Dimensions to reduce.
        keepdims: If true, retains reduced dimensions with length 1.

    Returns:
        Mean array/scalar and a valid array/scalar that indicates where the mean could be computed successfully.
    """

    mask = mask.astype(arr.dtype) if mask.dtype == bool else mask
    num_valid = np.sum(mask, axis=axis, keepdims=keepdims)
    masked_arr = arr * mask
    masked_arr_sum = np.sum(masked_arr, axis=axis, keepdims=keepdims)

    with np.errstate(divide='ignore', invalid='ignore'):
        valid_mean = masked_arr_sum / num_valid
        is_valid = np.isfinite(valid_mean)
        valid_mean = np.nan_to_num(valid_mean, copy=False, nan=0, posinf=0, neginf=0)

    return valid_mean, is_valid


def thresh_inliers(gt, pred, thresh, mask=None, output_scaling_factor=1.0):
    """Computes the inlier (=error within a threshold) ratio for a predicted and ground truth depth map.

    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        thresh: Threshold for the relative difference between the prediction and ground truth.
        mask: Array of shape HxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]).

    Returns:
        Scalar that indicates the inlier ratio. Scalar is np.nan if the result is invalid.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_1 = np.nan_to_num(gt / pred, nan=thresh+1, posinf=thresh+1, neginf=thresh+1)  # pred=0 should be an outlier
        rel_2 = np.nan_to_num(pred / gt, nan=0, posinf=0, neginf=0)  # gt=0 is masked out anyways

    max_rel = np.maximum(rel_1, rel_2)
    inliers = ((0 < max_rel) & (max_rel < thresh)).astype(np.float32)  # 1 for inliers, 0 for outliers

    inlier_ratio, valid = valid_mean(inliers, mask)

    inlier_ratio = inlier_ratio * output_scaling_factor
    inlier_ratio = inlier_ratio if valid else np.nan

    return inlier_ratio


def m_rel_ae(gt, pred, mask=None, output_scaling_factor=1.0):
    """Computes the mean-relative-absolute-error for a predicted and ground truth depth map.

    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        mask: Array of shape HxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]).


    Returns:
        Scalar that indicates the mean-relative-absolute-error. Scalar is np.nan if the result is invalid.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)

    e = pred - gt
    ae = np.abs(e)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_ae = np.nan_to_num(ae / gt, nan=0, posinf=0, neginf=0)

    m_rel_ae, valid = valid_mean(rel_ae, mask)

    m_rel_ae = m_rel_ae * output_scaling_factor
    m_rel_ae = m_rel_ae if valid else np.nan

    return m_rel_ae


def pointwise_rel_ae(gt, pred, mask=None, output_scaling_factor=1.0):
    """Computes the pointwise relative-absolute-error for a predicted and ground truth depth map.

    Args:
        gt: Ground truth depth map as numpy array of shape 1HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape 1xHxW.
        mask: Array of shape 1xHxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]).

    Returns:
        Numpy array of shape 1xHxW with pointwise relative-absolute-error values.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)

    e = pred - gt
    ae = np.abs(e)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_ae = np.nan_to_num(ae / gt, nan=0, posinf=0, neginf=0)  # nan values are masked out anyways
    rel_ae *= mask

    rel_ae = rel_ae * output_scaling_factor

    return rel_ae


def sparsification(gt, pred, uncertainty, mask=None, error_fct=m_rel_ae, show_pbar=False, pbar_desc=None):
    """Computes the sparsification curve for a predicted and ground truth depth map and a given ranking.

    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        uncertainty: Uncertainty measure for the predicted depth map. Numpy array of shape HxW.
        mask: Array of shape HxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
        error_fct: Function that computes a metric between ground truth and prediction for the sparsification curve.
        show_pbar: Show progress bar.
        pbar_desc: Prefix for the progress bar.

    Returns:
        Pandas Series with (sparsification_ratio, error_ratio) values of the sparsification curve.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)

    y, x = np.unravel_index(np.argsort((uncertainty - uncertainty.min() + 1) * mask, axis=None), uncertainty.shape)
    # (masking out values that are anyways not considered for computing the error)
    ranking = np.flip(np.stack((x, y), axis=1), 0).tolist()

    num_valid = np.sum(mask.astype(bool))
    sparsification_steps = [int((num_valid / 100) * i) for i in range(100)]

    base_error = error_fct(gt=gt, pred=pred, mask=mask)
    sparsification_x, sparsification_y = [], []

    num_masked = 0
    pbar = tqdm(total=num_valid, desc=pbar_desc,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit}',
                disable=not show_pbar, unit="removed pixels", ncols=80)
    for x, y in ranking:
        if num_masked >= num_valid:
            break

        if mask[y, x] == 0:
            raise RuntimeError('This should never happen. If it happens, please open a GitHub issue.')

        if num_masked in sparsification_steps:
            cur_error = error_fct(gt=gt, pred=pred, mask=mask)
            sparsification_frac = num_masked / num_valid
            error_frac = cur_error / base_error
            if np.isfinite(cur_error):
                sparsification_x.append(sparsification_frac)
                sparsification_y.append(error_frac)

        mask[y, x] = 0
        num_masked += 1
        pbar.update(1)

    pbar.close()
    x = np.linspace(0, 0.99, 100)

    if len(sparsification_x) > 1:
        sparsification = np.interp(x, sparsification_x, sparsification_y)
    else:
        sparsification = np.array([np.nan] * 100, dtype=np.float64)
    sparsification = pd.Series(sparsification, index=x)

    return sparsification

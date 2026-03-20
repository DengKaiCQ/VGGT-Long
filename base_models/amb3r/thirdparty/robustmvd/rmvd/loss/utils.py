# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/loss/utils.py

import torch


def mae(gt, pred, mask=None, weight=None, eps=1e-9):

    e = pred - gt
    ae = torch.abs(e)

    if weight is not None:

        if isinstance(weight, torch.Tensor):
            while weight.ndim < ae.ndim:
                weight = weight.unsqueeze(-1)

        ae *= weight

    if mask is None:
        mae = ae.mean()
    else:
        mask = mask.float()
        num_valid = torch.sum(mask)
        mae = 1 / (num_valid + eps) * torch.sum(ae * mask)
        mae *= float((num_valid != 0))

    return mae


def pointwise_ae(gt, pred, mask=None, weight=None):

    e = pred - gt
    pointwise_ae = torch.abs(e)

    if mask is not None:
        pointwise_ae *= mask.float()

    if weight is not None:

        if isinstance(weight, torch.Tensor):
            while weight.ndim < pointwise_ae.ndim:
                weight = weight.unsqueeze(1)

        pointwise_ae *= weight

    return pointwise_ae


def m_univariate_laplace_nll(gt, pred_a, pred_log_b, mask=None, weight=None, eps=1e-9):

    e = pred_a - gt
    ae = torch.abs(e)
    pred_b = torch.exp(pred_log_b)
    nll = ae / pred_b + pred_log_b  # N, 1, H, W

    if weight is not None:

        if isinstance(weight, torch.Tensor):
            while weight.ndim < nll.ndim:
                weight = weight.unsqueeze(-1)

        nll *= weight

    if mask is None:
        mnll = nll.mean()
    else:
        mask = mask.float()
        num_valid = torch.sum(mask)
        mnll = 1 / (num_valid + eps) * torch.sum(nll * mask)
        mnll *= float((num_valid != 0))

    return mnll


def pointwise_univariate_laplace_nll(gt, pred_a, pred_log_b, mask=None, weight=None):

    e = pred_a - gt
    ae = torch.abs(e)  # torch.sqrt(torch.pow(e, 2) + 5e-3)
    pred_b = torch.exp(pred_log_b)
    pointwise_nll = ae / pred_b + pred_log_b  # N, 1, H, W

    if mask is not None:
        pointwise_nll *= mask.float()

    if weight is not None:

        if isinstance(weight, torch.Tensor):
            while weight.ndim < pointwise_nll.ndim:
                weight = weight.unsqueeze(1)

        pointwise_nll *= weight

    return pointwise_nll
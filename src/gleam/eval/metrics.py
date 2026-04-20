from __future__ import annotations

import flip_evaluator
import numpy as np
import torch
from pytorch_msssim import ssim as pt_ssim
from scipy.spatial.distance import directed_hausdorff
from skimage.feature import canny


def _to_unit_tensor(img_uint8: np.ndarray, device: str) -> torch.Tensor:
    """(H, W, 3) uint8 -> (1, 3, H, W) float32 in [0, 1] on device."""
    t = torch.from_numpy(img_uint8.copy()).permute(2, 0, 1).float().div_(255.0)
    return t.unsqueeze(0).to(device)


def _rgb_to_luma(img_uint8: np.ndarray) -> np.ndarray:
    r, g, b = img_uint8[..., 0], img_uint8[..., 1], img_uint8[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def compute_flip(pred_uint8: np.ndarray, gt_uint8: np.ndarray) -> float:
    """NVIDIA FLIP (LDR): inputs are sRGB float32 in [0, 1], shape (H, W, 3)."""
    pred = pred_uint8.astype(np.float32) / 255.0
    gt = gt_uint8.astype(np.float32) / 255.0
    _err_map, mean_flip, _params = flip_evaluator.evaluate(gt, pred, "LDR")
    return float(mean_flip)


def compute_lpips(
    pred_uint8: np.ndarray,
    gt_uint8: np.ndarray,
    model,
    device: str,
) -> float:
    pred = _to_unit_tensor(pred_uint8, device) * 2.0 - 1.0
    gt = _to_unit_tensor(gt_uint8, device) * 2.0 - 1.0
    with torch.no_grad():
        return float(model(pred, gt).item())


def compute_ssim(pred_uint8: np.ndarray, gt_uint8: np.ndarray, device: str) -> float:
    pred = _to_unit_tensor(pred_uint8, device)
    gt = _to_unit_tensor(gt_uint8, device)
    with torch.no_grad():
        return float(
            pt_ssim(pred, gt, data_range=1.0, size_average=True, nonnegative_ssim=True).item()
        )


def compute_hausdorff_canny(
    pred_uint8: np.ndarray,
    gt_uint8: np.ndarray,
    sigma: float = 1.0,
) -> float:
    """Symmetric Hausdorff distance between Canny-edge sets, in pixels."""
    edges_pred = canny(_rgb_to_luma(pred_uint8), sigma=sigma)
    edges_gt = canny(_rgb_to_luma(gt_uint8), sigma=sigma)
    pts_pred = np.argwhere(edges_pred)
    pts_gt = np.argwhere(edges_gt)
    if len(pts_pred) == 0 or len(pts_gt) == 0:
        h, w = edges_pred.shape
        return float(np.hypot(h, w))
    d1 = directed_hausdorff(pts_pred, pts_gt)[0]
    d2 = directed_hausdorff(pts_gt, pts_pred)[0]
    return float(max(d1, d2))

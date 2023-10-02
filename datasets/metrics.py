from typing import List, Union

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from torch import Tensor
from torch.nn import functional as F


def compute_valid_depth_rmse(prediction: Tensor, target: Tensor) -> float:
    """
    Computes the root mean squared error (RMSE) between the predicted and target depth values,
    only considering the valid rays (where target > 0).

    Args:
    - prediction (Tensor): predicted depth values
    - target (Tensor): target depth values

    Returns:
    - float: RMSE between the predicted and target depth values, only considering the valid rays
    """
    prediction, target = prediction.squeeze(), target.squeeze()
    valid_mask = target > 0
    prediction = prediction[valid_mask]
    target = target[valid_mask]
    return F.mse_loss(prediction, target).sqrt().item()


def compute_psnr(prediction: Tensor, target: Tensor) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between the prediction and target tensors.

    Args:
        prediction (torch.Tensor): The predicted tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        float: The PSNR value between the prediction and target tensors.
    """
    if not isinstance(prediction, Tensor):
        prediction = Tensor(prediction)
    if not isinstance(target, Tensor):
        target = Tensor(target).to(prediction.device)
    return (-10 * torch.log10(F.mse_loss(prediction, target))).item()


def compute_ssim(
    prediction: Union[Tensor, np.ndarray], target: Union[Tensor, np.ndarray]
) -> float:
    """
    Computes the Structural Similarity Index (SSIM) between the prediction and target images.

    Args:
        prediction (Union[Tensor, np.ndarray]): The predicted image.
        target (Union[Tensor, np.ndarray]): The target image.

    Returns:
        float: The SSIM value between the prediction and target images.
    """
    if isinstance(prediction, Tensor):
        prediction = prediction.cpu().numpy()
    if isinstance(target, Tensor):
        target = target.cpu().numpy()
    assert target.max() <= 1.0 and target.min() >= 0.0, "target must be in range [0, 1]"
    assert (
        prediction.max() <= 1.0 and prediction.min() >= 0.0
    ), "prediction must be in range [0, 1]"
    return ssim(target, prediction, data_range=1.0, channel_axis=-1)

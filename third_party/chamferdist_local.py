# Stolen from pytorch3d, and tweaked (very little).

from collections import namedtuple
from typing import Tuple

import torch

# Throws an error without this import
from chamferdist.chamfer import knn_points

_KNN = namedtuple("KNN", "dists idx knn")


class ChamferDistance(torch.nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(
        self,
        source_cloud: torch.Tensor,
        target_cloud: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the forward and backward Chamfer distance between two point clouds.

        Args:
            source_cloud (torch.Tensor): The source point cloud tensor of shape (batchsize, lengths, dim).
            target_cloud (torch.Tensor): The target point cloud tensor of shape (batchsize, lengths, dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the forward and backward Chamfer distance tensors of shape (batchsize, lengths).
        """
        batchsize_source, lengths_source, dim_source = source_cloud.shape
        batchsize_target, lengths_target, dim_target = target_cloud.shape

        lengths_source = (
            torch.ones(batchsize_source, dtype=torch.long, device=source_cloud.device)
            * lengths_source
        )
        lengths_target = (
            torch.ones(batchsize_target, dtype=torch.long, device=target_cloud.device)
            * lengths_target
        )
        if batchsize_source != batchsize_target:
            raise ValueError(
                "Source and target pointclouds must have the same batchsize."
            )
        if dim_source != dim_target:
            raise ValueError(
                "Source and target pointclouds must have the same dimensionality."
            )

        source_nn = knn_points(
            source_cloud,
            target_cloud,
            lengths1=lengths_source,
            lengths2=lengths_target,
            K=1,
        )

        target_nn = knn_points(
            target_cloud,
            source_cloud,
            lengths1=lengths_target,
            lengths2=lengths_source,
            K=1,
        )

        # Forward Chamfer distance (batchsize_source, lengths_source)
        chamfer_forward = source_nn.dists[..., 0]
        chamfer_backward = target_nn.dists[..., 0]
        chamfer_forward = chamfer_forward[0]
        chamfer_backward = chamfer_backward[0]

        return chamfer_forward, chamfer_backward

import logging
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import vedo
import vedo.utils
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim
from torch import Tensor
from tqdm import tqdm, trange

from datasets import WaymoSequenceLoader, WaymoSequenceSplit
from datasets.metrics import compute_psnr, compute_valid_depth_rmse
from radiance_fields.radiance_field import DensityField, RadianceField
from radiance_fields.render_utils import render_rays
from third_party.nerfacc_prop_net import PropNetEstimator
from utils.visualization_tools import (
    get_robust_pca,
    resize_five_views,
    scene_flow_to_rgb,
    to8b,
    visualize_depth,
)

logger = logging.getLogger()

depth_visualizer = lambda frame, opacity: visualize_depth(
    frame,
    opacity,
    lo=4.0,
    hi=120,
    depth_curve_fn=lambda x: -np.log(x + 1e-6),
)


def render_pixels(
    cfg: OmegaConf,
    model: RadianceField,
    prop_estimator: PropNetEstimator,
    dataset: WaymoSequenceLoader,
    proposal_networks: Optional[List[DensityField]] = None,
    compute_metrics: bool = False,
    vis_indices: Optional[List[int]] = None,
    custom_downscale: Optional[float] = None,
    return_decomposition: bool = False,
):
    model.eval()
    if proposal_networks is not None:
        for p in proposal_networks:
            p.eval()
    if prop_estimator is not None:
        prop_estimator.eval()
    render_func = lambda data_dict: render_rays(
        radiance_field=model,
        prop_estimator=prop_estimator,
        proposal_networks=proposal_networks,
        data_dict=data_dict,
        cfg=cfg,
        return_decomposition=return_decomposition,  # return static-dynamic decomposition
    )
    render_results = render(
        dataset,
        render_func,
        compute_metrics=compute_metrics,
        vis_indices=vis_indices,
        custom_downscale=custom_downscale,
    )
    if compute_metrics:
        num_samples = len(dataset) if vis_indices is None else len(vis_indices)
        logger.info(f"Eval over {num_samples} images:")
        logger.info(f"\tPSNR: {render_results['psnr']:.4f}")
        logger.info(f"\tDepthRMSE: {render_results['depth_rmse']:.4f}")
        logger.info(f"\tSSIM: {render_results['ssim']:.4f}")
        logger.info(f"\tDino PSNR: {render_results['dino_psnr']:.4f}")
        logger.info(f"\tMasked PSNR: {render_results['masked_psnr']:.4f}")
        logger.info(f"\tMasked DepthRMSE: {render_results['masked_depth_rmse']:.4f}")
        logger.info(f"\tMasked SSIM: {render_results['masked_ssim']:.4f}")
        logger.info(f"\tMasked Dino PSNR: {render_results['masked_dino_psnr']:.4f}")

    return render_results


def render(
    dataset: WaymoSequenceSplit,
    render_func: Callable,
    compute_metrics: bool = False,
    vis_indices: Optional[List[int]] = None,
    custom_downscale: Optional[float] = None,
):
    """
    Renders a dataset utilizing a specified render function. This function is designed to address memory limitations associated
    with large datasets, which may not fit entirely in GPU memory. To mitigate this, it renders images individually, storing
    them in CPU memory.

    For efficiency and space-saving reasons, this function doesn't store the original 'dino' features; instead, it keeps
    the colors reduced via PCA.
    TODO: clean up the code for computing PCA colors. It's a bit messy.

    Parameters:
        dataset: A sequence loader for Waymo data or a dictionary of PyTorch tensors.
        render_func: Callable function used for rendering the dataset.
        compute_metrics: Optional; if True, the function will compute and return metrics. Default is False.
    """
    rgbs, gt_rgbs = [], []
    static_rgbs, dynamic_rgbs = [], []
    shadow_reduced_static_rgbs, shadow_only_static_rgbs = [], []
    shadows = []
    depths, gt_lidar_depths = [], []
    static_depths, static_opacities = [], []
    dynamic_depths, dynamic_opacities = [], []
    opacities, sky_masks = [], []
    pred_dinos, gt_dinos = [], []
    pred_dinos_pe_free, pred_dino_pe = [], []
    static_dinos, dynamic_dinos = [], []  # should we also render this?
    dynamic_dino_on_static_rgbs, dynamic_rgb_on_static_dinos = [], []
    forward_flows, backward_flows = [], []
    flows = []
    if compute_metrics:
        psnrs, depth_rmses, ssim_scores, dino_psnrs = [], [], [], []
        masked_psnrs, masked_depth_rmses, masked_ssim_scores = [], [], []
        masked_dino_psnrs = []
    with torch.no_grad():
        indices = vis_indices if vis_indices is not None else range(len(dataset))
        computed = False
        for i in tqdm(indices, desc=f"rendering {dataset.split}"):
            data_dict = dataset.fetch_render_pixel_data(
                i, custom_downscale=custom_downscale
            )
            render_results = render_func(data_dict)
            # ------------- rgb ------------- #
            rgb = render_results["rgb"]
            rgbs.append(rgb.cpu().numpy())
            if "pixels" in data_dict:
                gt_rgbs.append(data_dict["pixels"].cpu().numpy())
            if "static_rgb" in render_results:
                static_rgbs.append(render_results["static_rgb"].cpu().numpy())
            if "dynamic_rgb" in render_results:
                green_background = torch.tensor([0.0, 177, 64]) / 255.0
                green_background = green_background.to(
                    render_results["dynamic_rgb"].device
                )
                dy_rgb = render_results["dynamic_rgb"] * render_results[
                    "dynamic_opacity"
                ] + green_background * (1 - render_results["dynamic_opacity"])
                dynamic_rgbs.append(dy_rgb.cpu().numpy())
            if "shadow_reduced_static_rgb" in render_results:
                shadow_reduced_static_rgbs.append(
                    render_results["shadow_reduced_static_rgb"].cpu().numpy()
                )
            if "shadow_only_static_rgb" in render_results:
                shadow_only_static_rgbs.append(
                    render_results["shadow_only_static_rgb"].cpu().numpy()
                )
            if "shadow" in render_results:
                shadows.append(render_results["shadow"].squeeze().cpu().numpy())
            if "forward_flow" in render_results:
                forward_flows.append(
                    scene_flow_to_rgb(
                        render_results["forward_flow"],
                        background="bright",
                        flow_max_radius=1.0,
                    )
                    .cpu()
                    .numpy()
                )
            if "backward_flow" in render_results:
                backward_flows.append(
                    scene_flow_to_rgb(
                        render_results["backward_flow"],
                        background="bright",
                        flow_max_radius=1.0,
                    )
                    .cpu()
                    .numpy()
                )
            if "flow" in render_results:
                flows.append(
                    scene_flow_to_rgb(
                        render_results["flow"],
                        background="bright",
                        flow_max_radius=1.0,
                    )
                    .cpu()
                    .numpy(),
                )
            # ------------- depth ------------- #
            depth = render_results["depth"]
            depths.append(depth.squeeze().cpu().numpy())
            # ------------- opacity ------------- #
            opacities.append(render_results["opacity"].squeeze().cpu().numpy())
            if "static_depth" in render_results:
                static_depths.append(
                    render_results["static_depth"].squeeze().cpu().numpy()
                )
                static_opacities.append(
                    render_results["static_opacity"].squeeze().cpu().numpy()
                )
            if "dynamic_depth" in render_results:
                dynamic_depths.append(
                    render_results["dynamic_depth"].squeeze().cpu().numpy()
                )
                dynamic_opacities.append(
                    render_results["dynamic_opacity"].squeeze().cpu().numpy()
                )
            # gt depth
            if "depth" in data_dict:
                gt_lidar_depths.append(data_dict["depth"].squeeze().cpu().numpy())
                if compute_metrics:
                    depth_rmse = compute_valid_depth_rmse(depth, data_dict["depth"])
                    depth_rmses.append(depth_rmse)
            # -------- / sky -------- #
            if "sky_mask" in data_dict:
                sky_masks.append(data_dict["sky_mask"].squeeze().cpu().numpy())

            if compute_metrics:
                psnrs.append(compute_psnr(rgb, data_dict["pixels"]))
                ssim_scores.append(
                    ssim(
                        rgb.cpu().numpy(),
                        data_dict["pixels"].cpu().numpy(),
                        data_range=1.0,
                        channel_axis=-1,
                    )
                )
                if "dynamic_mask" in data_dict:
                    dynamic_mask = (
                        data_dict["dynamic_mask"].squeeze().bool().cpu().numpy()
                    )
                    if dynamic_mask.sum() > 0:
                        masked_psnrs.append(
                            compute_psnr(
                                rgb[dynamic_mask], data_dict["pixels"][dynamic_mask]
                            )
                        )
                        if "depth" in data_dict:
                            masked_depth_rmses.append(
                                compute_valid_depth_rmse(
                                    depth[dynamic_mask],
                                    data_dict["depth"][dynamic_mask],
                                )
                            )
                        masked_ssim_scores.append(
                            ssim(
                                rgb.cpu().numpy(),
                                data_dict["pixels"].cpu().numpy(),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][dynamic_mask].mean()
                        )

            # -------------- dino ------------- #
            if "dino_feat" in render_results:
                pred_dino_feat = render_results["dino_feat"]
                if "dino_feat" in data_dict:
                    gt_dino_feat = data_dict["dino_feat"]
                    if compute_metrics:
                        dino_psnrs.append(compute_psnr(pred_dino_feat, gt_dino_feat))
                        if "dynamic_mask" in data_dict:
                            dynamic_mask = data_dict["dynamic_mask"].squeeze()
                            if dynamic_mask.sum() > 0:
                                masked_dino_error = compute_psnr(
                                    pred_dino_feat[dynamic_mask],
                                    gt_dino_feat[dynamic_mask],
                                )
                                masked_dino_psnrs.append(masked_dino_error)

                else:
                    gt_dino_feat = None
                pred_dino_feat = (
                    pred_dino_feat
                    @ dataset.dino_dimension_reduction_mat.to(pred_dino_feat)
                )
                pred_dino_feat = (
                    pred_dino_feat - dataset.color_norm_min.to(pred_dino_feat)
                ) / (
                    dataset.color_norm_max.to(pred_dino_feat)
                    - dataset.color_norm_min.to(pred_dino_feat)
                )
                if gt_dino_feat is not None:
                    gt_dino_feat = (
                        gt_dino_feat
                        @ dataset.dino_dimension_reduction_mat.to(pred_dino_feat)
                    )
                    gt_dino_feat = (
                        gt_dino_feat - dataset.color_norm_min.to(pred_dino_feat)
                    ) / (
                        dataset.color_norm_max.to(pred_dino_feat)
                        - dataset.color_norm_min.to(pred_dino_feat)
                    )
                if "dino_pe_free" in render_results:
                    if not computed:
                        computed = True
                        non_sky_dino_pe_free = render_results["dino_pe_free"] * (
                            ~data_dict["sky_mask"].unsqueeze(-1)
                        ).to(render_results["dino_pe_free"])
                        (
                            dino_pe_free_reduction_mat,
                            dino_pe_free_color_min,
                            dino_pe_free_color_max,
                        ) = get_robust_pca(
                            non_sky_dino_pe_free.reshape(
                                -1,
                                render_results["dino_pe_free"].shape[-1],
                            ),
                            m=2.5,
                        )
                        (
                            pe_reduction_mat,
                            pe_color_min,
                            pe_color_max,
                        ) = get_robust_pca(
                            render_results["dino_pe"].reshape(
                                -1, render_results["dino_pe"].shape[-1]
                            ),
                            m=2.5,
                        )
                    dino_pe_free = (
                        render_results["dino_pe_free"] @ dino_pe_free_reduction_mat
                    )
                    dino_pe_free = (
                        dino_pe_free - dino_pe_free_color_min.to(pred_dino_feat)
                    ) / (
                        dino_pe_free_color_max.to(pred_dino_feat)
                        - dino_pe_free_color_min.to(pred_dino_feat)
                    )
                    dino_pe_free = torch.clamp(dino_pe_free, 0, 1)
                    dino_pe_free *= render_results["opacity"]
                    pred_dinos_pe_free.append(dino_pe_free.cpu().numpy())

                    dino_pe = render_results["dino_pe"] @ pe_reduction_mat
                    dino_pe = (dino_pe - pe_color_min.to(pred_dino_feat)) / (
                        pe_color_max - pe_color_min
                    ).to(pred_dino_feat)
                    dino_pe = torch.clamp(dino_pe, 0, 1)
                    # dino_pe_free *= render_results["opacity"]
                    pred_dino_pe.append(dino_pe.cpu().numpy())
                    if "static_dino" in render_results:
                        static_dino_feat = (
                            render_results["static_dino"] @ dino_pe_free_reduction_mat
                        )
                        static_dino_feat = (
                            static_dino_feat - dino_pe_free_color_min.to(pred_dino_feat)
                        ) / (dino_pe_free_color_max - dino_pe_free_color_min).to(
                            pred_dino_feat
                        )
                        static_dino_feat = torch.clamp(static_dino_feat, 0, 1)
                        # dino_pe_free *= render_results["opacity"]
                        static_dinos.append(static_dino_feat.cpu().numpy())
                        # get dynamic_rgb on static_dino
                        dynamic_rgb_on_static_dino = render_results[
                            "dynamic_rgb"
                        ].cpu().numpy() * dynamic_opacities[-1][
                            ..., None
                        ] + static_dinos[
                            -1
                        ] * (
                            1 - dynamic_opacities[-1][..., None]
                        )
                        dynamic_rgb_on_static_dino = np.clip(
                            dynamic_rgb_on_static_dino, 0, 1
                        )
                        dynamic_rgb_on_static_dinos.append(dynamic_rgb_on_static_dino)

                    if "dynamic_dino" in render_results:
                        dynamic_dino_feat = (
                            render_results["dynamic_dino"] @ dino_pe_free_reduction_mat
                        )
                        dynamic_dino_feat = (
                            dynamic_dino_feat
                            - dino_pe_free_color_min.to(pred_dino_feat)
                        ) / (dino_pe_free_color_max - dino_pe_free_color_min).to(
                            pred_dino_feat
                        )
                        dynamic_dino_feat = torch.clamp(dynamic_dino_feat, 0, 1)
                        # dino_pe_free *= render_results["opacity"]
                        dynamic_dinos.append(dynamic_dino_feat.cpu().numpy())
                        # get dynamic_dino on static_rgb
                        dynamic_dino_on_static_rgb = dynamic_dinos[
                            -1
                        ] * dynamic_opacities[-1][..., None] + static_rgbs[-1] * (
                            1 - dynamic_opacities[-1][..., None]
                        )
                        dynamic_dino_on_static_rgb = np.clip(
                            dynamic_dino_on_static_rgb, 0, 1
                        )
                        dynamic_dino_on_static_rgbs.append(dynamic_dino_on_static_rgb)
                else:
                    if "static_dino" in render_results:
                        # use dataset dataset.dino_dimension_reduction_mat
                        static_dino_feat = render_results[
                            "static_dino"
                        ] @ dataset.dino_dimension_reduction_mat.to(pred_dino_feat)
                        static_dino_feat = (
                            static_dino_feat - dataset.color_norm_min.to(pred_dino_feat)
                        ) / (
                            dataset.color_norm_max.to(pred_dino_feat)
                            - dataset.color_norm_min.to(pred_dino_feat)
                        )
                        static_dino_feat = torch.clamp(static_dino_feat, 0, 1)
                        static_dinos.append(static_dino_feat.cpu().numpy())
                        # get dynamic_rgb on static_dino
                        dynamic_rgb_on_static_dino = render_results[
                            "dynamic_rgb"
                        ].cpu().numpy() * dynamic_opacities[-1][
                            ..., None
                        ] + static_dinos[
                            -1
                        ] * (
                            1 - dynamic_opacities[-1][..., None]
                        )
                        dynamic_rgb_on_static_dino = np.clip(
                            dynamic_rgb_on_static_dino, 0, 1
                        )
                        dynamic_rgb_on_static_dinos.append(dynamic_rgb_on_static_dino)
                    if "dynamic_dino" in render_results:
                        # use dataset dataset.dino_dimension_reduction_mat
                        dynamic_dino_feat = (
                            render_results["dynamic_dino"]
                            @ dataset.dino_dimension_reduction_mat
                        )
                        dynamic_dino_feat = (
                            dynamic_dino_feat
                            - dataset.color_norm_min.to(pred_dino_feat)
                        ) / (
                            dataset.color_norm_max.to(pred_dino_feat)
                            - dataset.color_norm_min.to(pred_dino_feat)
                        )
                        dynamic_dino_feat = torch.clamp(dynamic_dino_feat, 0, 1)
                        dynamic_dinos.append(dynamic_dino_feat.cpu().numpy())
                        # get dynamic_dino on static_rgb
                        dynamic_dino_on_static_rgb = dynamic_dinos[
                            -1
                        ] * dynamic_opacities[-1][..., None] + static_rgbs[-1] * (
                            1 - dynamic_opacities[-1][..., None]
                        )
                        dynamic_dino_on_static_rgb = np.clip(
                            dynamic_dino_on_static_rgb, 0, 1
                        )
                        dynamic_dino_on_static_rgbs.append(dynamic_dino_on_static_rgb)

                pred_dino_feat = torch.clamp(pred_dino_feat, 0, 1)
                # pred_dino_feat *= render_results["opacity"]
                pred_dinos.append(pred_dino_feat.squeeze().cpu().numpy())
                if gt_dino_feat is not None:
                    gt_dino_feat = torch.clamp(gt_dino_feat, 0, 1)
                    gt_dinos.append(gt_dino_feat.squeeze().cpu().numpy())
    results_dict = {}
    if compute_metrics:
        psnr_avg = sum(psnrs) / len(psnrs) if len(psnrs) > 0 else -1
        depth_rmse_avg = (
            sum(depth_rmses) / len(depth_rmses) if len(depth_rmses) > 0 else -1
        )
        ssim_avg = sum(ssim_scores) / len(ssim_scores) if len(ssim_scores) > 0 else -1
        dino_psnr_avg = sum(dino_psnrs) / len(dino_psnrs) if len(dino_psnrs) > 0 else -1
        masked_psnr_avg = (
            sum(masked_psnrs) / len(masked_psnrs) if len(masked_psnrs) > 0 else -1
        )
        masked_depth_rmse_avg = (
            sum(masked_depth_rmses) / len(masked_depth_rmses)
            if len(masked_depth_rmses) > 0
            else -1
        )
        masked_ssim_avg = (
            sum(masked_ssim_scores) / len(masked_ssim_scores)
            if len(masked_ssim_scores) > 0
            else -1
        )
        masked_dino_psnr_avg = (
            sum(masked_dino_psnrs) / len(masked_dino_psnrs)
            if len(masked_dino_psnrs) > 0
            else -1
        )
        results_dict["psnr"] = psnr_avg
        results_dict["depth_rmse"] = depth_rmse_avg
        results_dict["ssim"] = ssim_avg
        results_dict["dino_psnr"] = dino_psnr_avg
        results_dict["masked_psnr"] = masked_psnr_avg
        results_dict["masked_depth_rmse"] = masked_depth_rmse_avg
        results_dict["masked_ssim"] = masked_ssim_avg
        results_dict["masked_dino_psnr"] = masked_dino_psnr_avg
    else:
        results_dict["psnr"] = -1
        results_dict["depth_rmse"] = -1
        results_dict["ssim"] = -1
        results_dict["dino_psnr"] = -1
        results_dict["masked_psnr"] = -1
        results_dict["masked_depth_rmse"] = -1
        results_dict["masked_ssim"] = -1
        results_dict["masked_dino_psnr"] = -1
    results_dict["rgbs"] = rgbs
    results_dict["static_rgbs"] = static_rgbs
    results_dict["dynamic_rgbs"] = dynamic_rgbs
    results_dict["depths"] = depths
    results_dict["opacities"] = opacities
    results_dict["static_depths"] = static_depths
    results_dict["static_opacities"] = static_opacities
    results_dict["dynamic_depths"] = dynamic_depths
    results_dict["dynamic_opacities"] = dynamic_opacities
    if len(gt_rgbs) > 0:
        results_dict["gt_rgbs"] = gt_rgbs
    if len(gt_lidar_depths) > 0:
        results_dict["gt_lidar_depths"] = gt_lidar_depths
    if len(sky_masks) > 0:
        results_dict["gt_sky_masks"] = sky_masks
    if len(pred_dinos) > 0:
        results_dict["dino_feats"] = pred_dinos
    if len(gt_dinos) > 0:
        results_dict["gt_dino_feats"] = gt_dinos
    if len(pred_dinos_pe_free) > 0:
        results_dict["dino_feats_pe_free"] = pred_dinos_pe_free
    if len(pred_dino_pe) > 0:
        results_dict["dino_pe"] = pred_dino_pe
    if len(static_dinos) > 0:
        results_dict["static_dino_feats"] = static_dinos
    if len(dynamic_dinos) > 0:
        results_dict["dynamic_dino_feats"] = dynamic_dinos
    if len(dynamic_dino_on_static_rgbs) > 0:
        results_dict["dynamic_dino_on_static_rgbs"] = dynamic_dino_on_static_rgbs
    if len(dynamic_rgb_on_static_dinos) > 0:
        results_dict["dynamic_rgb_on_static_dinos"] = dynamic_rgb_on_static_dinos
    if len(shadow_reduced_static_rgbs) > 0:
        results_dict["shadow_reduced_static_rgbs"] = shadow_reduced_static_rgbs
    if len(shadow_only_static_rgbs) > 0:
        results_dict["shadow_only_static_rgbs"] = shadow_only_static_rgbs
    if len(shadows) > 0:
        results_dict["shadows"] = shadows
    if len(forward_flows) > 0:
        results_dict["forward_flows"] = forward_flows
    if len(backward_flows) > 0:
        results_dict["backward_flows"] = backward_flows
    if len(flows) > 0:
        results_dict["flows"] = flows
    return results_dict


def render_lidars(
    cfg: OmegaConf,
    model: RadianceField,
    dataset: WaymoSequenceLoader,
    prop_estimator: Optional[PropNetEstimator] = None,
    proposal_networks: Optional[List[DensityField]] = None,
    vis_indices: Optional[List[int]] = None,
    compute_metrics: bool = False,
    render_rays_on_image_only: bool = True,
    render_lidar_id: Optional[int] = None,
):
    model.eval()
    if proposal_networks is not None:
        for p in proposal_networks:
            p.eval()
    if prop_estimator is not None:
        prop_estimator.eval()
    render_func = lambda data_dict: render_rays(
        radiance_field=model,
        proposal_networks=proposal_networks,
        prop_estimator=prop_estimator,
        data_dict=data_dict,
        cfg=cfg,
        proposal_requires_grad=False,
        # return static-dynamic decomposition
        return_decomposition=True,
        prefix="lidar_",
    )
    render_results = render_lidar_ray(
        dataset,
        render_func,
        vis_indices=vis_indices,
        compute_metrics=compute_metrics,
        render_rays_on_image_only=render_rays_on_image_only,
        render_lidar_id=render_lidar_id,
    )
    if compute_metrics:
        num_samples = (
            dataset.num_timestamps if vis_indices is None else len(vis_indices)
        )
        logger.info(f"Eval over {num_samples} timestamps:")
        for k, v in render_results.items():
            if "avg_chamfer" in k or "avg_depth" in k:
                logger.info(f"\t{k}: {v:.4f}")
    return render_results


def render_lidar_ray(
    dataset: WaymoSequenceSplit,
    render_func: Callable,
    vis_indices: Optional[List[int]] = None,
    compute_metrics: bool = False,
    render_rays_on_image_only: bool = True,
    render_lidar_id: Optional[int] = None,
):
    pred_lidar_ranges, gt_lidar_ranges = [], []
    static_lidar_ranges, static_opacities = [], []
    dynamic_lidar_ranges, dynamic_opacities = [], []
    gt_lidar_xyz_list, lidar_xyz_list = [], []
    gt_cham_scores, pred_cham_scores = [], []
    gt_chamfer_list, pred_chamfer_list = [], []
    depth_rmse_list, depth_rmse_scores = [], []

    opacities = []
    if compute_metrics:
        from third_party.chamferdist_local import ChamferDistance
        chamferDist = ChamferDistance()
        chamfers_pred, chamfers_pred_99 = [], []
        chamfers_gt, chamfers_gt_99 = [], []
        chamfers_all, chamfers_all_99 = [], []
        two_way_chamfers = []
        depth_errors, depth_errors_99 = [], []
    with torch.no_grad():
        indices = (
            vis_indices if vis_indices is not None else range(dataset.num_timestamps)
        )
        for i in tqdm(indices, desc=f"rendering {dataset.split} lidar"):
            data_dict = dataset.fetch_render_lidar_data(
                i,
                return_rays_on_image=render_rays_on_image_only,
                lidar_id=render_lidar_id,
            ).copy()
            render_results = render_func(data_dict)
            render_results.pop("extras")
            opacity = render_results["opacity"].squeeze()
            pred_lidar_ranges.append(render_results["depth"].squeeze().cpu().numpy())
            opacities.append(render_results["opacity"].squeeze().cpu().numpy())
            gt_lidar_ranges.append(data_dict["lidar_ranges"].squeeze().cpu().numpy())
            if "static_depth" in render_results:
                static_lidar_ranges.append(
                    render_results["static_depth"].squeeze().cpu().numpy()
                )
                static_opacities.append(
                    render_results["static_opacity"].squeeze().cpu().numpy()
                )
            if "dynamic_depth" in render_results:
                dynamic_lidar_ranges.append(
                    render_results["dynamic_depth"].squeeze().cpu().numpy()
                )
                dynamic_opacities.append(
                    render_results["dynamic_opacity"].squeeze().cpu().numpy()
                )
            # gt depth
            if "lidar_ranges" in data_dict:
                gt_lidar_xyz = data_dict["lidar_xyz"]
                pred_lidar_xyz = (
                    data_dict["lidar_origins"]
                    + data_dict["lidar_viewdirs"] * render_results["depth"]
                )

                if compute_metrics:
                    # follow street surf to compute chamfer distance
                    cham_pred, cham_gt = chamferDist(
                        pred_lidar_xyz[None, ...], gt_lidar_xyz[None, ...]
                    )
                    cham_pred_sorted = torch.sort(cham_pred).values
                    cham_gt_sorted = torch.sort(cham_gt).values
                    mean_cham_pred = cham_pred.mean().item()
                    mean_cham_gt = cham_gt.mean().item()

                    depth_err_each_abs = (
                        render_results["depth"].squeeze()
                        - data_dict["lidar_ranges"].squeeze()
                    ).abs()
                    depth_rmse = depth_err_each_abs.square().mean().sqrt().item()
                    depth_err_each_abs_sorted = torch.sort(depth_err_each_abs).values

                    mean_cham_pred_99 = (
                        cham_pred_sorted[0 : int(cham_pred_sorted.numel() * 0.99)]
                        .mean()
                        .item()
                    )
                    mean_cham_gt_99 = (
                        cham_gt_sorted[0 : int(cham_gt_sorted.numel() * 0.99)]
                        .mean()
                        .item()
                    )
                    depth_err_rmse_99 = (
                        depth_err_each_abs_sorted[
                            0 : int(depth_err_each_abs_sorted.numel() * 0.99)
                        ]
                        .mean()
                        .sqrt()
                        .item()
                    )

                    depth_errors.append(depth_rmse)
                    depth_errors_99.append(depth_err_rmse_99)

                    chamfers_pred.append(mean_cham_pred)
                    chamfers_pred_99.append(mean_cham_pred_99)

                    chamfers_gt.append(mean_cham_gt)
                    chamfers_gt_99.append(mean_cham_gt_99)

                    chamfers_all.append(mean_cham_pred + mean_cham_gt)
                    chamfers_all_99.append(mean_cham_pred_99 + mean_cham_gt_99)

                    two_way_chamfers.append((cham_pred + cham_gt).cpu().numpy())

                    gt_lidar_xyz_list.append(gt_lidar_xyz.cpu().numpy())
                    lidar_xyz_list.append(pred_lidar_xyz.cpu().numpy())
                    gt_chamfer_list.append(cham_gt.cpu().numpy())
                    pred_chamfer_list.append(cham_pred.cpu().numpy())
                    gt_cham_scores.append(mean_cham_gt)
                    pred_cham_scores.append(mean_cham_pred)
                    depth_rmse_list.append(depth_err_each_abs.cpu().numpy())
                    depth_rmse_scores.append(depth_rmse)

    results_dict = {}
    if compute_metrics:
        avg_chamfer_pred = sum(chamfers_pred) / len(chamfers_pred)
        avg_chamfer_pred_99 = sum(chamfers_pred_99) / len(chamfers_pred_99)
        avg_chamfer_gt = sum(chamfers_gt) / len(chamfers_gt)
        avg_chamfer_gt_99 = sum(chamfers_gt_99) / len(chamfers_gt_99)
        avg_chamfer_all = sum(chamfers_all) / len(chamfers_all)
        avg_chamfer_all_99 = sum(chamfers_all_99) / len(chamfers_all_99)
        avg_depth_error = sum(depth_errors) / len(depth_errors)
        avg_depth_error_99 = sum(depth_errors_99) / len(depth_errors_99)
        results_dict["avg_chamfer_pred"] = avg_chamfer_pred
        results_dict["avg_chamfer_pred_99"] = avg_chamfer_pred_99
        results_dict["avg_chamfer_gt"] = avg_chamfer_gt
        results_dict["avg_chamfer_gt_99"] = avg_chamfer_gt_99
        results_dict["avg_chamfer_all"] = avg_chamfer_all
        results_dict["avg_chamfer_all_99"] = avg_chamfer_all_99
        results_dict["avg_depth_rmse"] = avg_depth_error
        results_dict["avg_depth_rmse_99"] = avg_depth_error_99

        results_dict["gt_chamfer_list"] = gt_chamfer_list
        results_dict["pred_chamfer_list"] = pred_chamfer_list
        results_dict["gt_cham_scores"] = gt_cham_scores
        results_dict["pred_cham_scores"] = pred_cham_scores
        results_dict["two_way_chamfers"] = two_way_chamfers
        results_dict["gt_lidar_xyz_list"] = gt_lidar_xyz_list
        results_dict["lidar_xyz_list"] = lidar_xyz_list
        results_dict["depth_rmse_list"] = depth_rmse_list
        results_dict["depth_rmse_scores"] = depth_rmse_scores

    else:
        for k in [
            "chamfer_pred",
            "chamfer_pred_99",
            "chamfer_gt",
            "chamfer_gt_99",
            "chamfer",
            "chamfer_99",
            "depth_rmse",
            "depth_rmse_99",
        ]:
            results_dict[k] = -1

    results_dict["gt_lidar_ranges"] = gt_lidar_ranges
    results_dict["pred_lidar_ranges"] = pred_lidar_ranges
    results_dict["static_lidar_ranges"] = static_lidar_ranges
    results_dict["dynamic_lidar_ranges"] = dynamic_lidar_ranges
    results_dict["opacities"] = opacities
    if len(gt_lidar_ranges) > 0:
        results_dict["gt_lidar_ranges"] = gt_lidar_ranges
    if len(pred_lidar_ranges) > 0:
        results_dict["pred_lidar_ranges"] = pred_lidar_ranges
    if len(static_lidar_ranges) > 0:
        results_dict["static_lidar_ranges"] = static_lidar_ranges
    if len(dynamic_lidar_ranges) > 0:
        results_dict["dynamic_lidar_ranges"] = dynamic_lidar_ranges
    return results_dict


def save_video_or_image(
    uri: str, frames: List[np.array], fps: int, verbose: bool = False
):
    """Save frames as a video or image based on their count."""
    if len(frames) > 1:
        if ".mp4" not in uri:
            uri = f"{uri}.mp4"
        imageio.mimwrite(uri, frames, fps=fps)
        if verbose:
            logger.info(f"Video saved to {uri}")
    else:
        if ".mp4" in uri:
            uri = uri.replace(".mp4", ".png")
        imageio.imwrite(uri, frames[0])
        if verbose:
            logger.info(f"Image saved to {uri}")


def save_lidar_simulation(
    render_results: Dict[str, List[Tensor]],
    dataset: WaymoSequenceSplit,
    save_pth: str,
    fps: int = 10,
    num_cams: int = 3,
    verbose: bool = False,
):
    def vis_lidar_pcl_vedo(
        pcl_world: Tensor,
        pcl_val: Tensor,
        min: float = None,
        max: float = None,
        index: int = 0,
    ):
        # NOTE: Convert to a common coordinate system (OpenCV pinhole camera in this case)
        pcl_cam_ref = (
            dataset.world_to_cam(pcl_world, num_cams * index + 1).cpu().numpy()
        )
        if min is None:
            min = pcl_val.min().item()
        if max is None:
            max = pcl_val.max().item()
        pts_c = (
            (vedo.color_map(pcl_val, "rainbow", vmin=min, vmax=max) * 255.0)
            .clip(0, 255)
            .astype(np.uint8)
        )
        pts_c = np.concatenate(
            [pts_c, np.full_like(pts_c[:, :1], 255)], axis=-1
        )  # RGBA is ~50x faster
        lidar_pts = vedo.Points(pcl_cam_ref, c=pts_c, r=2)
        # Top view
        plt_top.clear()
        plt_top.show(
            lidar_pts,
            resetcam=True,
            size=[W_lidar_vis, H_lidar_vis],
            camera={
                "focal_point": [0.0, 0.0, 15.0],
                "pos": [0.0, -100.0, 15.0],
                "viewup": [-1, 0, 0],
            },
        )
        im1 = plt_top.topicture().tonumpy()

        # Front view
        plt_front.clear()
        plt_front.show(
            lidar_pts,
            resetcam=True,
            size=[W_lidar_vis, H_lidar_vis],
            camera={
                "focal_point": [0.0, 0.0, 50.0],
                "pos": [0.0, -5, -19.82120022],
                "viewup": [0.0, -0.99744572, 0.07142857],
            },
        )
        im2 = plt_front.topicture().tonumpy()
        return im1, im2

    def draw_text(im, content):
        im = Image.fromarray(im)
        img_draw = ImageDraw.Draw(im)

        img_draw.text(
            (int(W_lidar_vis * 0.05), int(W_lidar_vis * 0.05)),
            content,
            fill=font_color,
        )
        im = np.asarray(im)
        return im

    video_bg = "white"
    bg_color = np.array(vedo.get_color("white")) * 255.0
    font_color = (0, 0, 0)
    bg_color = tuple(bg_color.clip(0, 255).astype(np.uint8).tolist())
    # W_lidar_vis = 600
    # H_lidar_vis = W_lidar_vis * 9 // 16
    W_lidar_vis, H_lidar_vis = 800, 600
    plt_top = vedo.Plotter(
        interactive=False,
        offscreen=True,
        size=[W_lidar_vis, H_lidar_vis],
        bg=video_bg,
    )
    plt_front = vedo.Plotter(
        interactive=False,
        offscreen=True,
        size=[W_lidar_vis, H_lidar_vis],
        bg=video_bg,
    )
    plt_demo = vedo.Plotter(
        interactive=False,
        offscreen=True,
        shape=(1, 2),
        size=[1600, 600],
        bg=video_bg,
    )
    pcl_imgs = {}
    for i in trange(
        len(render_results["gt_lidar_xyz_list"]),
        desc="Building lidar videos",
    ):
        gt_lidar_xyz = render_results["gt_lidar_xyz_list"][i]
        pred_lidar_xyz = render_results["lidar_xyz_list"][i]
        gt_range = render_results["gt_lidar_ranges"][i]
        gt_range = torch.from_numpy(gt_range).cuda()
        pred_range = render_results["pred_lidar_ranges"][i]
        pred_range = torch.from_numpy(pred_range).cuda()
        depth_rmse = render_results["depth_rmse_list"][i]
        depth_rmse_score = render_results["depth_rmse_scores"][i]
        cham_pred_score = render_results["pred_cham_scores"][i]
        cham_gt_score = render_results["gt_cham_scores"][i]
        cham_pred = render_results["pred_chamfer_list"][i]
        cham_gt = render_results["gt_chamfer_list"][i]

        lidar_xyz_ego = (
            dataset.world_to_cam(gt_lidar_xyz, num_cams * i + 1).cpu().numpy()
        )
        pcl_gt_ego = vedo.Points(lidar_xyz_ego, c="gray3", r=2)
        anno_gt = vedo.Text2D("GT", s=2)
        gt_plots = [pcl_gt_ego, anno_gt]

        pred_lidar_xyz_ego = (
            dataset.world_to_cam(pred_lidar_xyz, num_cams * i + 1).cpu().numpy()
        )
        pcl_pred = vedo.Points(pred_lidar_xyz_ego, r=2)
        pcl_pred.cmap("rainbow", cham_pred + cham_gt, vmin=0.0, vmax=5.0)
        pcl_pred.add_scalarbar(title="", font_size=24, nlabels=6, c="black")
        anno_pred = vedo.Text2D(
            f"Rendered/simulated\nChamfer Distance={cham_pred_score:.3f}", s=2
        )
        pred_plots = [anno_pred, pcl_pred]
        plt_demo.clear(at=0, deep=True)
        plt_demo.clear(at=1, deep=True)
        plt_demo.show(
            gt_plots,
            at=0,
            resetcam=False,
            camera={
                "focal_point": [-6.1, 6.2, 34.1],
                "pos": [18.0, -22.1, -26.7],
                "viewup": [0.006176, -0.905337, 0.424648],
            },
        )
        plt_demo.show(pred_plots, at=1, resetcam=False)
        im_demo = plt_demo.topicture().tonumpy()
        pcl_imgs.setdefault("demo", []).append(im_demo)

        min_range = max(
            gt_range.quantile(0.05).item(),
            pred_range.quantile(0.05).item(),
        )
        max_range = max(
            gt_range.quantile(0.95).item(),
            pred_range.quantile(0.95).item(),
        )

        im1, im2 = vis_lidar_pcl_vedo(
            gt_lidar_xyz,
            gt_range.squeeze().cpu().numpy(),
            min=min_range,
            max=max_range,
            index=i,
        )
        im1 = draw_text(im1, "Ground Truth")
        im2 = draw_text(im2, "Ground Truth")
        pcl_imgs.setdefault("gt", []).append([im1, im2])

        im1, im2 = vis_lidar_pcl_vedo(
            pred_lidar_xyz,
            pred_range.squeeze().cpu().numpy(),
            min=min_range,
            max=max_range,
            index=i,
        )
        im1 = draw_text(im1, "Predicted")
        im2 = draw_text(im2, "Predicted")
        pcl_imgs.setdefault("pred", []).append([im1, im2])

        im1, im2 = vis_lidar_pcl_vedo(
            gt_lidar_xyz,
            cham_gt,
            min=0.0,
            max=1.0,
            index=i,
        )
        im1 = draw_text(im1, f"GT Chamfer: {cham_gt_score:.3f}")
        im2 = draw_text(im2, f"GT Chamfer: {cham_gt_score:.3f}")
        pcl_imgs.setdefault("gt_err_chamfer", []).append([im1, im2])

        im1, im2 = vis_lidar_pcl_vedo(
            pred_lidar_xyz,
            cham_pred,
            min=0.0,
            max=1.0,
            index=i,
        )
        im1 = draw_text(im1, f"Pred Chamfer: {cham_pred_score:.3f}")
        im2 = draw_text(im2, f"Pred Chamfer: {cham_pred_score:.3f}")
        pcl_imgs.setdefault("pred_err_chamfer", []).append([im1, im2])

        im1, im2 = vis_lidar_pcl_vedo(
            pred_lidar_xyz,
            depth_rmse,
            min=0.0,
            max=5.0,
            index=i,
        )
        im1 = draw_text(im1, f"RMSE: {depth_rmse_score:.3f}")
        im2 = draw_text(im2, f"RMSE: {depth_rmse_score:.3f}")
        pcl_imgs.setdefault("rmse_on_pred_pts", []).append([im1, im2])

        im1, im2 = vis_lidar_pcl_vedo(
            gt_lidar_xyz,
            depth_rmse,
            min=0.0,
            max=5.0,
            index=i,
        )
        im1 = draw_text(im1, f"RMSE: {depth_rmse_score:.3f}")
        im2 = draw_text(im2, f"RMSE: {depth_rmse_score:.3f}")
        pcl_imgs.setdefault("rmse_on_gt_pts", []).append([im1, im2])

    video_frames = []
    video_frames_err = []
    for i in range(len(pcl_imgs["gt"])):
        video_frames.append(
            np.concatenate(
                [
                    np.concatenate(
                        [pcl_imgs[key][i][0] for key in ["gt", "pred"]], axis=1
                    ),
                    np.concatenate(
                        [
                            pcl_imgs[key][i][0]
                            for key in ["gt_err_chamfer", "pred_err_chamfer"]
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
            )
        )
        video_frames_err.append(
            np.concatenate(
                [
                    pcl_imgs[key][i][0]
                    for key in [
                        "pred_err_chamfer",
                        "rmse_on_pred_pts",
                        "rmse_on_gt_pts",
                    ]
                ],
                axis=1,
            )
        )
    return_frames = {}
    save_video_or_image(
        os.path.join(save_pth, f"TOP_topdown.mp4"),
        video_frames,
        fps=fps,
        verbose=verbose,
    )
    return_frames["TOP_topdown"] = video_frames[len(video_frames) // 2]
    save_video_or_image(
        os.path.join(save_pth, f"TOP_topdown_err.mp4"),
        video_frames_err,
        fps=fps,
        verbose=verbose,
    )
    return_frames["TOP_topdown_err"] = video_frames_err[len(video_frames_err) // 2]
    for key in pcl_imgs:
        if key == "demo":
            continue
        video_frames = [item[0] for item in pcl_imgs[key]]
        save_video_or_image(
            os.path.join(save_pth, f"TOP_topdown_{key}.mp4"),
            video_frames,
            fps=fps,
            verbose=verbose,
        )
        return_frames[f"TOP_topdown_{key}"] = video_frames[len(video_frames) // 2]

    video_frames = []
    video_frames_err = []
    for i in range(len(pcl_imgs["gt"])):
        video_frames.append(
            np.concatenate(
                [
                    np.concatenate(
                        [pcl_imgs[key][i][1] for key in ["gt", "pred"]], axis=1
                    ),
                    np.concatenate(
                        [
                            pcl_imgs[key][i][1]
                            for key in ["gt_err_chamfer", "pred_err_chamfer"]
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
            )
        )
        video_frames_err.append(
            np.concatenate(
                [
                    pcl_imgs[key][i][1]
                    for key in [
                        "pred_err_chamfer",
                        "rmse_on_pred_pts",
                        "rmse_on_gt_pts",
                    ]
                ],
                axis=1,
            )
        )
    save_video_or_image(
        os.path.join(save_pth, f"TOP_front.mp4"),
        video_frames,
        fps=fps,
        verbose=verbose,
    )
    return_frames["TOP_front"] = video_frames[len(video_frames) // 2]
    save_video_or_image(
        os.path.join(save_pth, f"TOP_front_err.mp4"),
        video_frames_err,
        fps=fps,
        verbose=verbose,
    )
    return_frames["TOP_front_err"] = video_frames_err[len(video_frames_err) // 2]
    for key in pcl_imgs:
        if key == "demo":
            continue
        video_frames = [item[1] for item in pcl_imgs[key]]
        save_video_or_image(
            os.path.join(save_pth, f"TOP_front_{key}.mp4"),
            video_frames,
            fps=fps,
            verbose=verbose,
        )
        return_frames[f"TOP_front_{key}"] = video_frames[len(video_frames) // 2]

    if "demo" in pcl_imgs.keys():
        save_video_or_image(
            os.path.join(save_pth, f"TOP_demo.mp4"),
            pcl_imgs["demo"],
            fps=fps,
            verbose=verbose,
        )
        return_frames["TOP_demo"] = pcl_imgs["demo"][len(pcl_imgs["demo"]) // 2]
    return return_frames


def save_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "gt_lidar_depths", "depths"],
    num_cams: int = 3,
    save_seperate_video: bool = False,
    save_images: bool = False,
    fps: int = 10,
    verbose: bool = True,
):
    if save_seperate_video:
        return_frame = save_seperate_videos(
            render_results,
            save_pth,
            num_timestamps=num_timestamps,
            keys=keys,
            num_cams=num_cams,
            save_images=save_images,
            fps=fps,
            verbose=verbose,
        )
    else:
        return_frame = save_concatenated_videos(
            render_results,
            save_pth,
            num_timestamps=num_timestamps,
            keys=keys,
            num_cams=num_cams,
            save_images=save_images,
            fps=fps,
            verbose=verbose,
        )
    return return_frame


def save_concatenated_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "gt_lidar_depths", "depths"],
    num_cams: int = 3,
    save_images: bool = False,
    fps: int = 10,
    verbose: bool = True,
):
    if num_timestamps == 1:  # it's an image
        writer = imageio.get_writer(save_pth, mode="I")
        return_frame_id = 0
    else:
        return_frame_id = num_timestamps // 2
        writer = imageio.get_writer(save_pth, mode="I", fps=fps)
    for i in range(num_timestamps):
        merged_list = []
        for key in keys:
            if key == "sky_masks":
                try:
                    frames = render_results["opacities"][
                        i * num_cams : (i + 1) * num_cams
                    ]
                except:
                    continue
            elif key == "gt_lidar_on_images":
                if "gt_lidar_depths" not in render_results:
                    continue
                frames = render_results["gt_rgbs"][i * num_cams : (i + 1) * num_cams]
                depth_colors = [
                    depth_visualizer(frame, frame > 0)
                    for frame in render_results["gt_lidar_depths"][
                        i * num_cams : (i + 1) * num_cams
                    ]
                ]
                valid_depth_masks = [
                    (frame > 0)[..., None]
                    for frame in render_results["gt_lidar_depths"][
                        i * num_cams : (i + 1) * num_cams
                    ]
                ]
                # plot lidar on images
                frames = [
                    frame * (1 - valid_depth_mask) + depth * valid_depth_mask
                    for frame, depth, valid_depth_mask in zip(
                        frames, depth_colors, valid_depth_masks
                    )
                ]
            # elif key == "depths_on_images":
            #     frames = render_results["rgbs"][i * num_cams : (i + 1) * num_cams]
            #     depth_colors = [
            #         depth_visualizer(frame, frame > 0)
            #         for frame in render_results["depths"][
            #             i * num_cams : (i + 1) * num_cams
            #         ]
            #     ]
            #     alpha = 0.5
            #     frames = [
            #         frame * (1 - alpha) + depth * alpha
            #         for frame, depth in zip(frames, depth_colors)
            #     ]
            elif key == "rgb_on_d":
                if "dynamic_depths" not in render_results:
                    continue
                static_opacities = render_results["static_opacities"][
                    i * num_cams : (i + 1) * num_cams
                ]
                static_depths = render_results["static_depths"][
                    i * num_cams : (i + 1) * num_cams
                ]
                frames = [
                    depth_visualizer(frame, opacity)
                    for frame, opacity in zip(static_depths, static_opacities)
                ]
                dynamic_rgbs = render_results["dynamic_rgbs"][
                    i * num_cams : (i + 1) * num_cams
                ]
                dynamic_opacities = render_results["dynamic_opacities"][
                    i * num_cams : (i + 1) * num_cams
                ]
                for do in dynamic_opacities:
                    do[do < 0.5] = 0
                frames = [
                    dynamic_rgb * opacity[..., None] + f * (1 - opacity[..., None])
                    for dynamic_rgb, f, opacity in zip(
                        dynamic_rgbs, frames, dynamic_opacities
                    )
                ]

            elif key == "depth_on_rgb":
                if "dynamic_rgbs" not in render_results:
                    continue
                dynamic_depths = render_results["dynamic_depths"][
                    i * num_cams : (i + 1) * num_cams
                ]
                dynamic_depths = [
                    depth_visualizer(frame, frame > 0) for frame in dynamic_depths
                ]
                static_rgbs = render_results["static_rgbs"][
                    i * num_cams : (i + 1) * num_cams
                ]
                opacities = render_results["static_opacities"][
                    i * num_cams : (i + 1) * num_cams
                ]
                frames = [
                    static_rgb * (1 - opacity[..., None]) + depth * opacity[..., None]
                    for static_rgb, depth, opacity in zip(
                        static_rgbs, dynamic_depths, opacities
                    )
                ]
            else:
                if key not in render_results or len(render_results[key]) == 0:
                    continue
                frames = render_results[key][i * num_cams : (i + 1) * num_cams]
                if key == "rgbs" and save_images:
                    if i == 0:
                        os.makedirs(save_pth.replace(".mp4", ""), exist_ok=True)
                    for j, frame in enumerate(frames):
                        cam_id_mapping = {0: 1, 1: 0, 2: 2}
                        imageio.imwrite(
                            save_pth.replace(
                                ".mp4", f"/{i:03d}_{cam_id_mapping[j]}.jpg"
                            ),
                            to8b(frame),
                        )

            if key == "gt_lidar_depths":
                frames = [depth_visualizer(frame, frame > 0) for frame in frames]
            elif key == "gt_sky_masks":
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif key == "sky_masks":
                frames = [
                    1 - np.stack([frame, frame, frame], axis=-1) for frame in frames
                ]
            elif "depth" in key and key != "depths_on_images":
                try:
                    opacities = render_results[key.replace("depths", "opacities")][
                        i * num_cams : (i + 1) * num_cams
                    ]
                except:
                    continue
                frames = [
                    depth_visualizer(frame, opacity)
                    for frame, opacity in zip(frames, opacities)
                ]
            elif key == "shadows":
                frames = [
                    visualize_depth(
                        frame,
                        frame > 0,
                        lo=None,
                        hi=None,
                        depth_curve_fn=lambda x: x,
                    )
                    for frame in frames
                ]
            frames = resize_five_views(frames)
            frames = np.concatenate(frames, axis=1)
            merged_list.append(frames)
        merged_frame = to8b(np.concatenate(merged_list, axis=0))
        if i == return_frame_id:
            return_frame = merged_frame
        writer.append_data(merged_frame)
    writer.close()
    if verbose:
        logger.info(f"saved video to {save_pth}")
    del render_results
    return {"concatenated_frame": return_frame}


def save_seperate_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "gt_lidar_depths", "depths"],
    num_cams: int = 3,
    fps: int = 10,
    verbose: bool = False,
    save_images: bool = False,
):
    return_frame_id = num_timestamps // 2
    return_frame_dict = {}
    for key in keys:
        tmp_save_pth = save_pth.replace(".mp4", f"_{key}.mp4")
        tmp_save_pth = tmp_save_pth.replace(".png", f"_{key}.png")
        if num_timestamps == 1:  # it's an image
            writer = imageio.get_writer(tmp_save_pth, mode="I")
        else:
            writer = imageio.get_writer(tmp_save_pth, mode="I", fps=fps)
        if key not in render_results or len(render_results[key]) == 0:
            continue
        for i in range(num_timestamps):
            if key == "sky_masks":
                frames = render_results["opacities"][i * num_cams : (i + 1) * num_cams]
            else:
                frames = render_results[key][i * num_cams : (i + 1) * num_cams]
            if key == "gt_lidar_depths":
                frames = [depth_visualizer(frame, frame > 0) for frame in frames]
            elif key == "gt_sky_masks":
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif key == "sky_masks":
                frames = [
                    1 - np.stack([frame, frame, frame], axis=-1) for frame in frames
                ]
            elif "depth" in key:
                opacities = render_results[key.replace("depths", "opacities")][
                    i * num_cams : (i + 1) * num_cams
                ]
                frames = [
                    depth_visualizer(frame, opacity)
                    for frame, opacity in zip(frames, opacities)
                ]
            frames = resize_five_views(frames)
            if save_images:
                if i == 0:
                    os.makedirs(tmp_save_pth.replace(".mp4", ""), exist_ok=True)
                for j, frame in enumerate(frames):
                    imageio.imwrite(
                        tmp_save_pth.replace(".mp4", f"_{i*3 + j:03d}.png"),
                        to8b(frame),
                    )
            frames = to8b(np.concatenate(frames, axis=1))
            writer.append_data(frames)
            if i == return_frame_id:
                return_frame_dict[key] = frames
        # close the writer
        writer.close()
        if verbose:
            logger.info(f"saved video to {tmp_save_pth}")
    del render_results
    return return_frame_dict

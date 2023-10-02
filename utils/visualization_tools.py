import logging
import os
from collections import namedtuple
from typing import Dict, Iterable, List, Optional, Union

import cv2
import matplotlib.cm as cm
import numpy as np
import plotly.graph_objects as go
import torch
from omegaconf import OmegaConf
from scipy import ndimage
from torch import Tensor
from tqdm import tqdm, trange

from datasets.utils import point_sampling, voxel_coords_to_world_coords
from radiance_fields import DensityField, RadianceField
from itertools import accumulate

DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)


logger = logging.getLogger()

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def fast_pca_HWC(X, V=None, return_v=False):
    H, W, C = X.shape
    X = X.reshape(-1, C)
    if return_v:
        X, V = fast_pca_NC(X, V, return_v)
    else:
        X = fast_pca_NC(X, V, return_v)
    X = X.reshape(H, W, 3)
    if return_v:
        return X, V
    else:
        return X


def fast_pca_NC(X, V=None, return_v=False):
    X = (X - X.mean(dim=0, keepdim=True)) / X.std(dim=0, keepdim=True)
    if V is None:
        U, S, V = torch.pca_lowrank(X)
        V = V[:, :3]
    X = X @ V
    X = (X - X.min(0, keepdim=True)[0]) / (
        X.max(0, keepdim=True)[0] - X.min(0, keepdim=True)[0]
    )
    X = X.clip(0, 1)
    if return_v:
        return X, V
    else:
        return X


def get_robust_pca(features: Tensor, m: float = 2, remove_first_component=False):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    rins = colors[fg_mask][s[:, 0] < m, 0]
    gins = colors[fg_mask][s[:, 1] < m, 1]
    bins = colors[fg_mask][s[:, 2] < m, 2]

    rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
    rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def fast_pca_HWC_v2(X):
    reduction_mat, rgb_min, rgb_max = fast_pca_v2(X)
    X = X @ reduction_mat
    X = (X - rgb_min) / (rgb_max - rgb_min)
    X = X.clip(0, 1)
    return X


def fast_pca_v2(features):
    features_reshape = features.view(-1, features.shape[-1])
    reduction_mat = torch.pca_lowrank(features_reshape, q=3, niter=20)[2]
    colors = features_reshape @ reduction_mat
    rgb_min = colors.min(dim=0).values
    rgb_max = colors.max(dim=0).values
    return reduction_mat, rgb_min, rgb_max


def to8b(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def resize_five_views(imgs: np.array):
    if len(imgs) != 5:
        return imgs
    for idx in [0, -1]:
        img = imgs[idx]
        new_shape = [int(img.shape[1] * 0.46), img.shape[1], 3]
        new_img = np.zeros_like(img)
        new_img[-new_shape[0] :, : new_shape[1], :] = ndimage.zoom(
            img, [new_shape[0] / img.shape[0], new_shape[1] / img.shape[1], 1]
        )
        # clip the image to 0-1
        new_img = np.clip(new_img, 0, 1)
        imgs[idx] = new_img
    return imgs


def sinebow(h):
    """A cyclic and uniform colormap, see http://basecase.org/env/on-rainbows."""
    f = lambda x: np.sin(np.pi * x) ** 2
    return np.stack([f(3 / 6 - h), f(5 / 6 - h), f(7 / 6 - h)], -1)


def matte(vis, acc, dark=0.8, light=1.0, width=8):
    """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
    bg_mask = np.logical_xor(
        (np.arange(acc.shape[0]) % (2 * width) // width)[:, None],
        (np.arange(acc.shape[1]) % (2 * width) // width)[None, :],
    )
    bg = np.where(bg_mask, light, dark)
    return vis * acc[:, :, None] + (bg * (1 - acc))[:, :, None]


def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)


def visualize_cmap(
    value,
    weight,
    colormap,
    lo=None,
    hi=None,
    percentile=99.0,
    curve_fn=lambda x: x,
    modulus=None,
    matte_background=True,
):
    """Visualize a 1D image and a 1D weighting according to some colormap.
    from mipnerf

    Args:
      value: A 1D image.
      weight: A weight map, in [0, 1].
      colormap: A colormap function.
      lo: The lower bound to use when rendering, if None then use a percentile.
      hi: The upper bound to use when rendering, if None then use a percentile.
      percentile: What percentile of the value map to crop to when automatically
        generating `lo` and `hi`. Depends on `weight` as well as `value'.
      curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
        before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
      modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
        `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
      matte_background: If True, matte the image over a checkerboard.

    Returns:
      A colormap rendering.
    """
    # Identify the values that bound the middle of `value' according to `weight`.
    if lo is None or hi is None:
        lo_auto, hi_auto = weighted_percentile(
            value, weight, [50 - percentile / 2, 50 + percentile / 2]
        )
        # If `lo` or `hi` are None, use the automatically-computed bounds above.
        eps = np.finfo(np.float32).eps
        lo = lo or (lo_auto - eps)
        hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = np.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = np.nan_to_num(
            np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)
        )
    if weight is not None:
        value *= weight
    else:
        weight = np.ones_like(value)
    if colormap:
        colorized = colormap(value)[..., :3]
    else:
        assert len(value.shape) == 3 and value.shape[-1] == 3
        colorized = value

    return matte(colorized, weight) if matte_background else colorized


def visualize_depth(
    x, acc=None, lo=None, hi=None, depth_curve_fn=lambda x: -np.log(x + 1e-6)
):
    """Visualizes depth maps."""
    return visualize_cmap(
        x,
        acc,
        cm.get_cmap("turbo"),
        curve_fn=depth_curve_fn,
        lo=lo,
        hi=hi,
        matte_background=False,
    )


def vis_occ_plotly(
    vis_aabb: List[Union[int, float]],
    coords: np.array,
    colors: np.array,
    dynamic_coords: List[np.array] = None,
    dynamic_colors: List[np.array] = None,
    x_ratio: float = 1.0,
    y_ratio: float = 1.0,
    z_ratio: float = 0.125,
    size: int = 5,
) -> go.Figure:  # type: ignore
    fig = go.Figure()  # start with an empty figure

    # Add static trace
    static_trace = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=dict(
            size=size,
            color=colors,
            symbol="square",
        ),
    )
    fig.add_trace(static_trace)

    # Add temporal traces
    if dynamic_coords is not None:
        for i in range(len(dynamic_coords)):
            fig.add_trace(
                go.Scatter3d(
                    x=dynamic_coords[i][:, 0],
                    y=dynamic_coords[i][:, 1],
                    z=dynamic_coords[i][:, 2],
                    mode="markers",
                    marker=dict(
                        size=size,
                        color=dynamic_colors[i],
                        symbol="diamond",
                    ),
                )
            )
        steps = []
        for i in range(len(dynamic_coords)):
            step = dict(
                method="restyle",
                args=[
                    "visible",
                    [False] * (len(dynamic_coords) + 1),
                ],  # Include the static trace
                label="Timestep {}".format(i),
            )
            step["args"][1][0] = True  # Make the static trace always visible
            step["args"][1][i + 1] = True  # Toggle i'th temporal trace to "visible"
            steps.append(step)

        sliders = [dict(active=0, pad={"t": 1}, steps=steps)]
        fig.update_layout(sliders=sliders)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x", showspikes=False, range=[vis_aabb[0], vis_aabb[3]]),
            yaxis=dict(title="y", showspikes=False, range=[vis_aabb[1], vis_aabb[4]]),
            zaxis=dict(title="z", showspikes=False, range=[vis_aabb[2], vis_aabb[5]]),
            aspectmode="manual",
            aspectratio=dict(x=x_ratio, y=y_ratio, z=z_ratio),
        ),
        margin=dict(r=0, b=10, l=0, t=10),
        hovermode=False,
    )

    return fig


def visualize_3d_voxels(
    cfg: OmegaConf,
    model: RadianceField,
    proposal_networks: DensityField,
    device: str = "cuda",
    save_html: bool = True,
):
    model.eval()
    for p in proposal_networks:
        p.eval()
    vis_voxel_aabb = torch.tensor(cfg.voxel_visualization.vis_aabb, device=device)
    aabb_min, aabb_max = torch.split(vis_voxel_aabb, 3, dim=-1)
    voxel_resolution = torch.ceil(
        (aabb_max - aabb_min) / cfg.voxel_visualization.vis_voxel_size
    ).long()
    aabb_length = aabb_max - aabb_min
    raw_points = voxel_coords_to_world_coords(aabb_min, aabb_max, voxel_resolution)
    raw_points = raw_points.reshape(-1, 3).to(device)
    dummy_pca_reduction = None
    with torch.no_grad():
        chunk = (
            2**24 if cfg.voxel_visualization.vis_query_key != "dino_feat" else 2**16
        )
        pca_colors = []
        occupied_points = []
        # query static fields
        for i in trange(0, raw_points.shape[0], chunk, desc="querying features"):
            density_list = []
            for p in proposal_networks:
                density_list.append(p(raw_points[i : i + chunk]).squeeze(-1))
            results = model.query_attributes(
                raw_points[i : i + chunk], query_timestamp=0.0
            )
            density_list.append(results["density"])
            density = torch.stack(density_list, dim=0)
            density = torch.mean(density, dim=0)
            selector = density > cfg.voxel_visualization.vis_density_threshold
            occupied_points_chunk = raw_points[i : i + chunk][selector]
            if len(occupied_points_chunk) == 0:
                continue
            # query some features
            feats = model.query_attributes(
                occupied_points_chunk,
                query_keys=[cfg.voxel_visualization.vis_query_key],
                query_timestamp=0.01,
            )[cfg.voxel_visualization.vis_query_key]
            if cfg.voxel_visualization.vis_query_key == "dino_feat":
                colors = feats @ model.dino_feats_reduction_mat
                colors = (colors - model.color_norm_min) / (
                    model.color_norm_max - model.color_norm_min
                )
            else:
                if dummy_pca_reduction is None:
                    dummy_pca_reduction, color_min, color_max = get_robust_pca(feats)
                colors = feats @ dummy_pca_reduction
                colors = (colors - color_min) / (color_max - color_min)
            pca_colors.append(torch.clamp(colors, 0, 1))
            occupied_points.append(occupied_points_chunk)
        pca_colors = torch.cat(pca_colors, dim=0)
        occupied_points = torch.cat(occupied_points, dim=0)
        logger.info(f"Raw points: {raw_points.shape[0]}")
        logger.info(f"Occupied points: {occupied_points.shape[0]}")
        figure = vis_occ_plotly(
            vis_aabb=cfg.voxel_visualization.vis_aabb,
            coords=occupied_points.cpu().numpy(),
            colors=pca_colors.cpu().numpy(),
            x_ratio=1,
            y_ratio=(aabb_length[1] / aabb_length[0]).item(),
            z_ratio=(aabb_length[2] / aabb_length[0]).item(),
            size=5,
        )
        if save_html:
            figure.write_html(
                os.path.join(
                    cfg.log_dir,
                    f"occ_field_{cfg.voxel_visualization.vis_density_threshold}.html",
                )
            )
            logger.info(
                f"Query result saved to {os.path.join(cfg.log_dir, f'occ_field_{cfg.voxel_visualization.vis_density_threshold}.html')}"
            )
        return figure


def visualize_3d_voxels_clean(
    cfg: OmegaConf,
    model: RadianceField,
    proposal_networks: DensityField,
    vis_dataset,
    device: str = "cuda",
    save_html: bool = True,
):
    model.eval()
    for p in proposal_networks:
        p.eval()
    vis_voxel_aabb = torch.tensor(cfg.voxel_visualization.vis_aabb, device=device)
    aabb_min, aabb_max = torch.split(vis_voxel_aabb, 3, dim=-1)
    voxel_resolution = torch.ceil(
        (aabb_max - aabb_min) / cfg.voxel_visualization.vis_voxel_size
    ).long()
    aabb_length = aabb_max - aabb_min

    from datasets.utils import world_coords_to_voxel_coords
    from radiance_fields.render_utils import render_rays

    render_func = lambda data_dict: render_rays(
        radiance_field=model,
        proposal_networks=proposal_networks,
        estimator=None,
        data_dict=data_dict,
        num_samples=cfg.nerf.sampling.num_samples,
        num_samples_per_prop=cfg.nerf.sampling.num_samples_per_prop,
        near_plane=cfg.nerf.sampling.near_plane,
        far_plane=cfg.nerf.sampling.far_plane,
        sampling_type=cfg.nerf.sampling.sampling_type,
        proposal_requires_grad=False,
        test_chunk_size=cfg.render.render_chunk_size,
        # return static-dynamic decomposition
    )
    # let's query the depth at first.
    empty_voxels = torch.zeros(*voxel_resolution, device=device)
    for i in trange(len(vis_dataset), desc="querying depth"):
        data_dict = vis_dataset[i]
        data_dict["lidar_origins"] = data_dict["origins"]
        data_dict["lidar_viewdirs"] = data_dict["viewdirs"]
        render_results = render_func(data_dict)
        depth = render_results["depth"]
        world_coords = data_dict["origins"] + data_dict["viewdirs"] * depth
        world_coords = world_coords[depth.squeeze() < 80]
        voxel_coords = world_coords_to_voxel_coords(
            world_coords, aabb_min, aabb_max, voxel_resolution
        )
        voxel_coords = voxel_coords.long()
        selector = (
            (voxel_coords[..., 0] >= 0)
            & (voxel_coords[..., 0] < voxel_resolution[0])
            & (voxel_coords[..., 1] >= 0)
            & (voxel_coords[..., 1] < voxel_resolution[1])
            & (voxel_coords[..., 2] >= 0)
            & (voxel_coords[..., 2] < voxel_resolution[2])
        )
        # Split the voxel_coords into separate dimensions
        voxel_coords_x = voxel_coords[..., 0][selector]
        voxel_coords_y = voxel_coords[..., 1][selector]
        voxel_coords_z = voxel_coords[..., 2][selector]
        # Index into empty_voxels using the separated coordinates
        empty_voxels[voxel_coords_x, voxel_coords_y, voxel_coords_z] = 1
    # now let's query the features
    occupied_voxel_coords = torch.nonzero(empty_voxels)
    all_occupied_points = voxel_coords_to_world_coords(
        aabb_min, aabb_max, voxel_resolution, occupied_voxel_coords
    )
    dummy_pca_reduction = None
    with torch.no_grad():
        chunk = (
            2**26 if cfg.voxel_visualization.vis_query_key != "dino_feat" else 2**20
        )
        pca_colors = []
        occupied_points = []
        # query static fields
        for i in trange(
            0, all_occupied_points.shape[0], chunk, desc="querying features"
        ):
            occupied_points_chunk = all_occupied_points[i : i + chunk]
            density_list = []
            for p in proposal_networks:
                density_list.append(p(occupied_points_chunk).squeeze(-1))
            results = model.query_attributes(occupied_points_chunk, query_timestamp=0.0)
            density_list.append(results["density"])
            density = torch.stack(density_list, dim=0)
            density = torch.mean(density, dim=0)
            selector = density > cfg.voxel_visualization.vis_density_threshold
            occupied_points_chunk = occupied_points_chunk[selector]
            if len(occupied_points_chunk) == 0:
                continue
            # query some features
            feats = model.query_attributes(
                occupied_points_chunk,
                query_keys=[cfg.voxel_visualization.vis_query_key],
                query_timestamp=0.01,
            )[cfg.voxel_visualization.vis_query_key]
            if cfg.voxel_visualization.vis_query_key == "dino_feat2":
                colors = feats @ model.dino_feats_reduction_mat
                colors = (colors - model.color_norm_min) / (
                    model.color_norm_max - model.color_norm_min
                )
            else:
                if dummy_pca_reduction is None:
                    print("using robust pca")
                    dummy_pca_reduction, color_min, color_max = get_robust_pca(feats)
                colors = feats @ dummy_pca_reduction
                colors = (colors - color_min) / (color_max - color_min)
            pca_colors.append(torch.clamp(colors, 0, 1))
            occupied_points.append(occupied_points_chunk)
        pca_colors = torch.cat(pca_colors, dim=0)
        occupied_points = torch.cat(occupied_points, dim=0)
        figure = vis_occ_plotly(
            vis_aabb=cfg.voxel_visualization.vis_aabb,
            coords=occupied_points.cpu().numpy(),
            colors=pca_colors.cpu().numpy(),
            x_ratio=1,
            y_ratio=(aabb_length[1] / aabb_length[0]).item(),
            z_ratio=(aabb_length[2] / aabb_length[0]).item(),
            size=5,
        )
        if save_html:
            figure.write_html(
                os.path.join(
                    cfg.log_dir,
                    f"occ_field_{cfg.voxel_visualization.vis_density_threshold}.html",
                )
            )
            logger.info(
                f"Query result saved to {os.path.join(cfg.log_dir, f'occ_field_{cfg.voxel_visualization.vis_density_threshold}.html')}"
            )
        return figure


def visualize_4d_voxels(
    cfg: OmegaConf,
    model: RadianceField,
    proposal_networks: DensityField,
    device: str = "cuda",
    save_html: bool = True,
    query_timestamp: Union[float, Iterable[float]] = None,
):
    model.eval()
    for p in proposal_networks:
        p.eval()
    vis_voxel_aabb = torch.tensor(cfg.voxel_visualization.vis_aabb, device=device)
    aabb_min, aabb_max = torch.split(vis_voxel_aabb, 3, dim=-1)
    voxel_resolution = torch.ceil(
        (aabb_max - aabb_min) / cfg.voxel_visualization.vis_voxel_size
    ).long()
    aabb_length = aabb_max - aabb_min
    raw_points = voxel_coords_to_world_coords(aabb_min, aabb_max, voxel_resolution)
    raw_points = raw_points.reshape(-1, 3).to(device)
    dummy_pca_reduction = None
    with torch.no_grad():
        chunk = 2**24
        pca_colors = []
        occupied_points = []
        static_query_key = f"static_{cfg.voxel_visualization.vis_query_key}"
        dynamic_query_key = f"dynamic_{cfg.voxel_visualization.vis_query_key}"

        # query static fields
        all_dynamic_pca_colors = []
        all_dynamic_occupied_points = []
        if query_timestamp is None:
            # query all timestamps
            if cfg.data.num_timestamps == -1:
                num_timestamps = 200
            else:
                num_timestamps = cfg.data.num_timestamps
            all_timestamps = torch.linspace(0, 1, num_timestamps, device=device)
        else:
            if isinstance(query_timestamp, float):
                query_timestamp = [query_timestamp]
            for t in query_timestamp:
                assert 0 <= t <= 1, "query_timestamp should be in [0, 1]"
            all_timestamps = query_timestamp
        for t in tqdm(all_timestamps, desc="Querying dynamic fields"):
            dynamic_pca_colors = []
            dynamic_occupied_points = []
            for i in range(0, raw_points.shape[0], chunk):
                if t == 0:
                    density_list = []
                    for p in proposal_networks:
                        density_list.append(p(raw_points[i : i + chunk]).squeeze(-1))
                    results = model.query_attributes(
                        raw_points[i : i + chunk], query_timestamp=0.0
                    )
                    density_list.append(results["static_density"])
                    static_density = torch.stack(density_list, dim=0)
                    static_density = torch.mean(static_density, dim=0)
                    selector = (
                        static_density > cfg.voxel_visualization.vis_density_threshold
                    )
                    occupied_points_chunk = raw_points[i : i + chunk][selector]
                    if len(occupied_points_chunk) > 0:
                        # query some features
                        feats = model.query_attributes(
                            occupied_points_chunk,
                            query_keys=[cfg.voxel_visualization.vis_query_key],
                            query_timestamp=0.0,
                        )[static_query_key]
                        if static_query_key == "static_dino_feat":
                            colors = feats @ model.dino_feats_reduction_mat
                            colors = (colors - model.color_norm_min) / (
                                model.color_norm_max - model.color_norm_min
                            )
                        else:
                            if dummy_pca_reduction is None:
                                (
                                    dummy_pca_reduction,
                                    color_min,
                                    color_max,
                                ) = get_robust_pca(feats)
                            colors = feats @ dummy_pca_reduction
                            colors = (colors - color_min) / (color_max - color_min)
                        pca_colors.append(torch.clamp(colors, 0, 1))
                        occupied_points.append(occupied_points_chunk)
                    else:
                        continue

                dynamic_density = model.query_attributes(
                    raw_points[i : i + chunk], query_timestamp=t
                )["dynamic_density"]
                dynamic_selector = (
                    dynamic_density > cfg.voxel_visualization.vis_density_threshold * 2
                )
                occupied_dynamic_points_chunk = raw_points[i : i + chunk][
                    dynamic_selector
                ]

                if len(occupied_dynamic_points_chunk) > 0:
                    # query some features
                    feats = model.query_attributes(
                        occupied_dynamic_points_chunk,
                        query_keys=[cfg.voxel_visualization.vis_query_key],
                        query_timestamp=t,
                    )[dynamic_query_key]
                    if dynamic_query_key == "dynamic_dino_feat":
                        colors = feats @ model.dino_feats_reduction_mat
                        colors = (colors - model.color_norm_min) / (
                            model.color_norm_max - model.color_norm_min
                        )
                    else:
                        if dummy_pca_reduction is None:
                            dummy_pca_reduction, color_min, color_max = get_robust_pca(
                                feats
                            )
                        colors = feats @ dummy_pca_reduction
                        colors = (colors - color_min) / (color_max - color_min)
                    dynamic_pca_colors.append(torch.clamp(colors, 0, 1))
                    dynamic_occupied_points.append(occupied_dynamic_points_chunk)
                    ###### modify here if you want to return features ########
                    # storing features for all timestamps and all locations will be very memory consuming
                    del feats
                    torch.cuda.empty_cache()
                    ###### modify here if you want to return features ########
                else:
                    continue
            dynamic_pca_colors = torch.cat(dynamic_pca_colors, dim=0)
            dynamic_occupied_points = torch.cat(dynamic_occupied_points, dim=0)
            all_dynamic_pca_colors.append(dynamic_pca_colors)
            all_dynamic_occupied_points.append(dynamic_occupied_points)
        pca_colors = torch.cat(pca_colors, dim=0)
        occupied_points = torch.cat(occupied_points, dim=0)
        logger.info(f"Raw points: {raw_points.shape[0]}")
        logger.info(f"Occupied points: {occupied_points.shape[0]}")
        figure = vis_occ_plotly(
            vis_aabb=cfg.voxel_visualization.vis_aabb,
            coords=occupied_points.cpu().numpy(),
            colors=pca_colors.cpu().numpy(),
            dynamic_coords=[x.cpu().numpy() for x in all_dynamic_occupied_points],
            dynamic_colors=[x.cpu().numpy() for x in all_dynamic_pca_colors],
            x_ratio=1,
            y_ratio=(aabb_length[1] / aabb_length[0]).item(),
            z_ratio=(aabb_length[2] / aabb_length[0]).item(),
            size=5,
        )
        if save_html:
            figure.write_html(
                os.path.join(
                    cfg.log_dir,
                    f"occ_field_{cfg.voxel_visualization.vis_density_threshold}.html",
                )
            )
            logger.info(
                f"Query result saved to {os.path.join(cfg.log_dir, f'occ_field_{cfg.voxel_visualization.vis_density_threshold}.html')}"
            )
        return figure


def visualize_data_sample(
    data_dict: Dict[str, Tensor], aabb: Union[Tensor, List[float]]
) -> np.ndarray:
    # dict_keys(['src_img', 'src_w2c', 'origins', 'viewdirs',
    # 'timestamp', 'pixels', 'depth', 'sky_mask', 'dino_feat'])
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)

    # num_cams, H, W, 3
    src_img = data_dict["src_img"].cpu().numpy()
    img_height, img_width = src_img.shape[1], src_img.shape[2]
    src_w2c = data_dict["src_w2c"]
    rgbs = data_dict["pixels"]

    src_img = resize_five_views(src_img)
    src_img = np.concatenate(src_img, axis=1)

    # visualize the first frame
    merged_list = [src_img]
    rgb = rgbs[0].cpu().numpy()
    rgb = resize_five_views(rgb)
    rgb = np.concatenate(rgb, axis=1)
    merged_list.append(rgb)

    if "depth" in data_dict:
        depth = data_dict["depth"][0].cpu().numpy()
        depth = visualize_depth(depth, depth > 0, lo=0.5, hi=120)
        depth = resize_five_views(depth)
        depth = np.concatenate(depth, axis=1)
        merged_list.append(depth)
    if "sky_mask" in data_dict:
        sky_mask = data_dict["sky_mask"][0].cpu().numpy()
        sky_mask = np.stack([sky_mask] * 3, axis=-1)
        sky_mask = resize_five_views(sky_mask)
        sky_mask = np.concatenate(sky_mask, axis=1)
        merged_list.append(sky_mask)
    if "dino_feat" in data_dict:
        # TODO: find a way to visualize dino_feat
        pass

    # visualize sampling points
    world_coords = (
        voxel_coords_to_world_coords(aabb[:3], aabb[3:], [64, 64, 64])
        .reshape(-1, 3)
        .cuda()
    )
    ref_pts, valid_mask, depth = point_sampling(
        world_coords.unsqueeze(0),
        src_w2c.cuda().unsqueeze(0),
        img_height,
        img_width,
    )
    # point_sampling is batched, so we need to squeeze the batch dim
    ref_pts, valid_mask, depth = (
        ref_pts.squeeze(0),
        valid_mask.squeeze(0),
        depth.squeeze(0),
    )
    ref_pts[..., 0] = (ref_pts[..., 0] + 1) / 2 * (img_width - 1)
    ref_pts[..., 1] = (ref_pts[..., 1] + 1) / 2 * (img_height - 1)
    ref_pts = ref_pts.round().long().cpu().numpy()
    valid_mask = valid_mask.cpu().numpy()
    src_rgb = rgbs[0].cpu().numpy()
    for cam_id in range(src_rgb.shape[0]):
        ## ---- visualize projected points ---- ##
        valid_points = ref_pts[cam_id, valid_mask[cam_id]]
        for point in valid_points:
            src_rgb[cam_id] = cv2.circle(
                src_rgb[cam_id], (int(point[0]), int(point[1])), 1, (0, 0, 1), -1
            )

    src_rgb = resize_five_views(src_rgb)
    src_rgb = np.concatenate(src_rgb, axis=1)
    merged_list.append(src_rgb)
    merged = np.concatenate(merged_list, axis=0)
    merged = to8b(merged)
    return merged


def _make_colorwheel(transitions: tuple = DEFAULT_TRANSITIONS) -> torch.Tensor:
    """Creates a colorwheel (borrowed/modified from flowpy).
    A colorwheel defines the transitions between the six primary hues:
    Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).
    Args:
        transitions: Contains the length of the six transitions, based on human color perception.
    Returns:
        colorwheel: The RGB values of the transitions in the color space.
    Notes:
        For more information, see:
        https://web.archive.org/web/20051107102013/http://members.shaw.ca/quadibloc/other/colint.htm
        http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    """
    colorwheel_length = sum(transitions)
    # The red hue is repeated to make the colorwheel cyclic
    base_hues = map(
        np.array,
        (
            [255, 0, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
            [255, 0, 255],
            [255, 0, 0],
        ),
    )
    colorwheel = np.zeros((colorwheel_length, 3), dtype="uint8")
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, accumulate(transitions)):
        transition_length = end_index - start_index
        colorwheel[start_index:end_index] = np.linspace(
            hue_from, hue_to, transition_length, endpoint=False
        )
        hue_from = hue_to
        start_index = end_index
    return torch.FloatTensor(colorwheel)


WHEEL = _make_colorwheel()
N_COLS = len(WHEEL)
WHEEL = torch.vstack((WHEEL, WHEEL[0]))  # Make the wheel cyclic for interpolation


# Adapted from https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior/blob/main/visualize.py
def scene_flow_to_rgb(
    flow: torch.Tensor,
    flow_max_radius: Optional[float] = None,
    background: Optional[str] = "dark",
) -> torch.Tensor:
    """Creates a RGB representation of an optical flow (borrowed/modified from flowpy).
    Args:
        flow: scene flow.
            flow[..., 0] should be the x-displacement
            flow[..., 1] should be the y-displacement
            flow[..., 2] should be the z-displacement
        flow_max_radius: Set the radius that gives the maximum color intensity, useful for comparing different flows.
            Default: The normalization is based on the input flow maximum radius.
        background: States if zero-valued flow should look 'bright' or 'dark'.
    Returns: An array of RGB colors.
    """
    valid_backgrounds = ("bright", "dark")
    if background not in valid_backgrounds:
        raise ValueError(
            f"background should be one the following: {valid_backgrounds}, not {background}."
        )

    # For scene flow, it's reasonable to assume displacements in x and y directions only for visualization pursposes.
    complex_flow = flow[..., 0] + 1j * flow[..., 1]
    radius, angle = torch.abs(complex_flow), torch.angle(complex_flow)
    if flow_max_radius is None:
        # flow_max_radius = torch.max(radius)
        flow_max_radius = torch.quantile(radius, 0.99)
    if flow_max_radius > 0:
        radius /= flow_max_radius
    # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
    angle[angle < 0] += 2 * np.pi
    angle = angle * ((N_COLS - 1) / (2 * np.pi))

    # Interpolate the hues
    angle_fractional, angle_floor, angle_ceil = (
        torch.fmod(angle, 1),
        angle.trunc(),
        torch.ceil(angle),
    )
    angle_fractional = angle_fractional.unsqueeze(-1)
    wheel = WHEEL.to(angle_floor.device)
    float_hue = (
        wheel[angle_floor.long()] * (1 - angle_fractional)
        + wheel[angle_ceil.long()] * angle_fractional
    )
    ColorizationArgs = namedtuple(
        "ColorizationArgs",
        ["move_hue_valid_radius", "move_hue_oversized_radius", "invalid_color"],
    )

    def move_hue_on_V_axis(hues, factors):
        return hues * factors.unsqueeze(-1)

    def move_hue_on_S_axis(hues, factors):
        return 255.0 - factors.unsqueeze(-1) * (255.0 - hues)

    if background == "dark":
        parameters = ColorizationArgs(
            move_hue_on_V_axis, move_hue_on_S_axis, torch.FloatTensor([255, 255, 255])
        )
    else:
        parameters = ColorizationArgs(
            move_hue_on_S_axis, move_hue_on_V_axis, torch.zeros(3)
        )
    colors = parameters.move_hue_valid_radius(float_hue, radius)
    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask], 1 / radius[oversized_radius_mask]
    )
    return colors / 255.0

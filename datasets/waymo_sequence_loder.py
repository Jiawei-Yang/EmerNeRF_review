import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor
from tqdm import tqdm, trange

from datasets.utils import get_projection_matrix, point_sampling
from utils.visualization_tools import get_robust_pca

logger = logging.getLogger()
ORIGINAL_SIZE = [[1280, 1920], [1280, 1920], [1280, 1920], [884, 1920], [884, 1920]]
OPENCV2WAYMO = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


def get_mover_c2w(
    center_frame_cam_to_worlds: Tensor,
    num_frames_per_step: int,
    move_range: float,
    device: torch.device,
) -> Tensor:
    """
    Generates camera-to-world transformation matrices for a sequence of frames that move in a specific pattern.

    Args:
        center_frame_cam_to_worlds (Tensor): Camera-to-world transformation matrix for the center frame.
        num_frames_per_step (int): Number of frames to generate for each movement step.
        move_range (float): Distance to move the frames in each direction.
        device (torch.device): Device to use for the generated tensors.

    Returns:
        Tensor: Camera-to-world transformation matrices for the generated frames.
    """
    # Move to left
    left_mover_c2w = center_frame_cam_to_worlds.repeat_interleave(
        num_frames_per_step, dim=0
    )
    left_mover_c2w[..., 1, 3] += torch.linspace(
        0, move_range, num_frames_per_step, device=device
    ).unsqueeze(-1)

    # Move back to center
    center_mover_c2w = left_mover_c2w[-1:].repeat_interleave(num_frames_per_step, dim=0)
    center_mover_c2w[..., 1, 3] -= torch.linspace(
        0, move_range, num_frames_per_step, device=device
    ).unsqueeze(-1)

    # Move to right
    right_mover_c2w = center_mover_c2w[-1:].repeat_interleave(
        num_frames_per_step, dim=0
    )
    right_mover_c2w[..., 1, 3] -= torch.linspace(
        0, move_range, num_frames_per_step, device=device
    ).unsqueeze(-1)

    # Move back to center
    center_mover_c2w2 = right_mover_c2w[-1:].repeat_interleave(
        num_frames_per_step, dim=0
    )
    center_mover_c2w2[..., 1, 3] += torch.linspace(
        0, move_range, num_frames_per_step, device=device
    ).unsqueeze(-1)

    # Move up
    up_mover_c2w = center_mover_c2w2[-1:].repeat_interleave(num_frames_per_step, dim=0)
    up_mover_c2w[..., 2, 3] += torch.linspace(
        0, move_range, num_frames_per_step, device=device
    ).unsqueeze(-1)

    # Move back to center
    center_mover_c2w3 = up_mover_c2w[-1:].repeat_interleave(num_frames_per_step, dim=0)
    center_mover_c2w3[..., 2, 3] -= torch.linspace(
        0, move_range, num_frames_per_step, device=device
    ).unsqueeze(-1)

    mover_c2ws = torch.cat(
        [
            left_mover_c2w,
            center_mover_c2w,
            right_mover_c2w,
            center_mover_c2w2,
            up_mover_c2w,
            center_mover_c2w3,
        ],
        dim=0,
    )
    return mover_c2ws


def idx_to_3d(idx, H, W):
    """
    Converts a 1D index to a 3D index (img_id, row_id, col_id)

    Args:
        idx (int): The 1D index to convert.
        H (int): The height of the 3D space.
        W (int): The width of the 3D space.

    Returns:
        tuple: A tuple containing the 3D index (i, j, k),
                where i is the image index, j is the row index,
                and k is the column index.
    """
    i = idx // (H * W)
    j = (idx % (H * W)) // W
    k = idx % W
    return i, j, k


class WaymoSequenceLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    def __init__(
        self,
        root: str,
        sequence_idx: int,
        start_timestamp: int,
        num_timestamps: int,
        num_cams: int = 3,
        num_rays: int = None,
        load_size: Tuple[int, int] = None,
        # downscale factor for evaluation and rendering
        downscale: float = 1.0,
        device: torch.device = torch.device("cpu"),
        load_rgb: bool = True,
        load_lidar: bool = False,
        load_sky_mask: bool = False,
        load_dino: bool = False,
        load_dynamic_mask: bool = False,
        # Use every Nth image for the test set.
        # if test_holdout=0, use all images for training and none for testing.
        test_holdout: int = 0,
        # if None, use the original dimension,
        # otherwise, use PCA to reduce the dimension to target_dino_dim
        target_dino_dim: int = None,
        dino_model_type: str = "dino_vitb8",
        scene_cfg: Optional[OmegaConf] = None,
        buffer_ratio: float = -1,
        buffer_downscale: int = 8,
        only_use_first_return: bool = True,
        only_use_top_lidar: bool = False,
        only_keep_lidar_rays_on_images: bool = False,
        only_keep_lidar_rays_within_truncated_range: float = -1.0,
    ):
        super().__init__()
        self.downscale = downscale
        self.num_cams = num_cams
        self.test_holdout = test_holdout
        self.start_idx = start_timestamp
        self.data_dir = f"{root}/{sequence_idx:03d}"
        self.sequence_idx = sequence_idx
        self.target_dino_dim = target_dino_dim
        self.cam_idx_list = {3: [1, 0, 2], 5: [3, 1, 0, 2, 4]}[num_cams]
        self.device = device
        self.only_use_first_return = only_use_first_return
        self.only_use_top_lidar = only_use_top_lidar
        self.only_keep_lidar_rays_on_images = only_keep_lidar_rays_on_images
        self.only_keep_lidar_rays_within_truncated_range = (
            only_keep_lidar_rays_within_truncated_range
        )
        self.end_idx = (
            len(os.listdir(os.path.join(self.data_dir, "ego_pose")))
            if num_timestamps == -1
            else start_timestamp + num_timestamps
        )
        self.num_timestamps = self.end_idx - self.start_idx
        if load_size is None:
            self.load_size = ORIGINAL_SIZE[0]
            self.HEIGHT, self.WIDTH = ORIGINAL_SIZE[0]
        else:
            self.load_size = load_size
            self.HEIGHT, self.WIDTH = load_size
        # self.load_potential_dynamic_mask()
        if self.test_holdout == 0:
            self.test_indices = []
        else:
            self.test_indices = torch.arange(self.num_timestamps)[
                self.test_holdout :: self.test_holdout
            ].tolist()
        # compute train set indices
        self.train_indices = [
            i for i in range(self.num_timestamps) if i not in self.test_indices
        ]
        logger.info(
            f"Train timestamp indices: {[x + self.start_idx for x in self.train_indices]}"
        )
        logger.info(
            f"Test timestamp indices: {[x + self.start_idx for x in self.test_indices]}"
        )
        self.load_extrinsics_and_intrinsics()
        self.load_rgb(load_rgb)
        self.load_dynamic_mask(load_dynamic_mask)
        self.load_lidar(load_lidar)
        self.load_sky_mask(load_sky_mask)
        self.load_dino(load_dino, dino_model_type)

        if scene_cfg is not None and load_lidar:
            self.aabb = self.get_auto_aabb(scene_cfg)
        else:
            self.aabb = None

        self.train_set = WaymoSequenceSplit(
            images=self.train_images,
            img_ids=self.train_img_ids,
            cam_to_worlds=self.train_cam_to_worlds,
            intrinsics=self.train_intrinsics,
            cam_to_egos=self.cam_to_egos,
            ego_to_worlds=self.train_ego_to_worlds,
            cam_ids=self.train_cam_ids,
            timestamps=self.train_timestamps,
            time_indices=self.train_time_indices,
            lidar_rays=self.train_lidar_rays,
            lidar_time_idx=self.train_lidar_time_idx,
            lidar_ids=self.train_lidar_ids,
            sky_masks=self.train_sky_masks,
            dynamic_masks=self.train_dynamic_masks,
            dino_features=self.train_dino_features,
            dino_scale=self.dino_scale,
            dino_dimension_reduction_mat=self.dino_dimension_reduction_mat,
            color_norm_min=self.color_norm_min,
            color_norm_max=self.color_norm_max,
            num_cams=num_cams,
            num_rays=num_rays,
            image_size=(self.HEIGHT, self.WIDTH),
            downscale=self.downscale,
            device=device,
            split="train",
            buffer_ratio=buffer_ratio,
            buffer_downscale=buffer_downscale,
        )

        if len(self.test_indices) > 0:
            self.test_set = WaymoSequenceSplit(
                images=self.test_images,
                img_ids=self.test_img_ids,
                cam_to_worlds=self.test_cam_to_worlds,
                intrinsics=self.test_intrinsics,
                cam_to_egos=self.cam_to_egos,
                ego_to_worlds=self.test_ego_to_worlds,
                cam_ids=self.test_cam_ids,
                timestamps=self.test_timestamps,
                time_indices=self.test_time_indices,
                lidar_rays=self.test_lidar_rays,
                lidar_time_idx=self.test_lidar_time_idx,
                lidar_ids=self.test_lidar_ids,
                sky_masks=self.test_sky_masks,
                dynamic_masks=self.test_dynamic_masks,
                dino_features=self.test_dino_features,
                dino_scale=self.dino_scale,
                dino_dimension_reduction_mat=self.dino_dimension_reduction_mat,
                color_norm_min=self.color_norm_min,
                color_norm_max=self.color_norm_max,
                num_cams=num_cams,
                num_rays=num_rays,
                image_size=(self.HEIGHT, self.WIDTH),
                downscale=self.downscale,
                device=device,
                split="test",
            )
        else:
            self.test_set = None

        self.full_set = WaymoSequenceSplit(
            self.images,
            self.img_ids,
            self.cam_to_worlds,
            self.intrinsics,
            self.cam_to_egos,
            self.ego_to_worlds,
            self.cam_ids,
            self.timestamps,
            self.time_indices,
            self.lidar_rays,
            self.lidar_time_idx,
            self.lidar_ids,
            self.sky_masks,
            self.dynamic_masks,
            self.dino_features,
            dino_scale=self.dino_scale,
            dino_dimension_reduction_mat=self.dino_dimension_reduction_mat,
            color_norm_min=self.color_norm_min,
            color_norm_max=self.color_norm_max,
            num_cams=num_cams,
            num_rays=None,
            image_size=(self.HEIGHT, self.WIDTH),
            downscale=self.downscale,
            device=device,
            split="full",
        )

    def get_auto_aabb(self, scene_cfg: OmegaConf) -> Tensor:
        if scene_cfg.auto_aabb_based_on_lidar:
            logger.info("Computing auto AABB based on lidar....")
            lidar_pts = (
                self.lidar_rays[:, :3]
                + self.lidar_rays[:, 3:6] * self.lidar_rays[:, 6:7]
            )
            lidar_pts = lidar_pts[
                torch.randperm(len(lidar_pts))[
                    : int(len(lidar_pts) / scene_cfg.lidar_downsample_ratio)
                ]
            ]
            aabb_min = torch.quantile(lidar_pts, scene_cfg.lidar_percentile, dim=0)
            aabb_max = torch.quantile(lidar_pts, 1 - scene_cfg.lidar_percentile, dim=0)
            del lidar_pts
            torch.cuda.empty_cache()
            if aabb_max[-1] < 20:
                aabb_max[-1] = 20.0
            aabb = torch.tensor([*aabb_min, *aabb_max]).to(self.device)
            logger.info(f"Auto AABB: {aabb}")
            return aabb
        else:
            return None

    def sync_error_maps(self):
        self.sync_pixel_error_maps()

    def sync_pixel_error_maps(self):
        self.full_set.pixel_error_maps = self.train_set.pixel_error_maps

    def load_extrinsics_and_intrinsics(self):
        _intrinsics = []
        cam_to_egos = []
        for i in range(self.num_cams):
            # load camera intrinsics
            intrinsic = np.loadtxt(
                os.path.join(self.data_dir, "intrinsics", f"{i}.txt")
            )
            fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
            # scale intrinsics w.r.t. load size
            fx, fy = (
                fx * self.load_size[1] / ORIGINAL_SIZE[i][1],
                fy * self.load_size[0] / ORIGINAL_SIZE[i][0],
            )
            cx, cy = (
                cx * self.load_size[1] / ORIGINAL_SIZE[i][1],
                cy * self.load_size[0] / ORIGINAL_SIZE[i][0],
            )
            # Load camera extrinsics
            # It's frame.context.camera_calibrations.extrinsic.transform from Waymo Open Dataset
            cam_to_ego = np.loadtxt(
                os.path.join(self.data_dir, "extrinsics", f"{i}.txt")
            )
            cam_to_egos.append(cam_to_ego @ OPENCV2WAYMO)
            intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            _intrinsics.append(intrinsic)

        # now pre-compute per-image poses and intrinsics
        cam_to_worlds, ego_to_worlds = [], []
        intrinsics, timestamps, cam_ids = [], [], []
        time_idx = []

        # We calibrate the camera poses w.r.t. the first timestamp,
        # so we need to load the first start_timestamp.
        ego_to_world_start = np.loadtxt(
            os.path.join(self.data_dir, "ego_pose", f"{self.start_idx:03d}.txt")
        )
        for t in range(self.start_idx, self.end_idx):
            # rotate the camera poses w.r.t. the first timestamp
            ego_to_world_current = np.loadtxt(
                os.path.join(self.data_dir, "ego_pose", f"{t:03d}.txt")
            )
            ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
            ego_to_worlds.append(ego_to_world)
            for cam_id in self.cam_idx_list:
                # pose
                # we generate camera rays in opencv coordinate system.
                # opencv coordinate system: x right, y down, z front
                # waymo coordinate system: x front, y left, z up
                # transformtion: opencv_cam -> waymo_cam -> waymo_vehicle -> waymo_world
                cam2world = ego_to_world @ cam_to_egos[cam_id]
                cam_to_worlds.append(cam2world)
                intrinsics.append(_intrinsics[cam_id])
                timestamps.append(t)
                cam_ids.append(cam_id)
                time_idx.append(t - self.start_idx)

        intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0))
        cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0))
        ego_to_worlds = torch.from_numpy(np.stack(ego_to_worlds, axis=0))
        cam_to_egos = torch.from_numpy(np.stack(cam_to_egos, axis=0))
        timestamps = torch.from_numpy(np.stack(timestamps, axis=0))
        timestamps = (timestamps - timestamps[0]) / (
            timestamps[-1] - timestamps[0] + 1e-6
        )
        time_idx = torch.from_numpy(np.stack(time_idx, axis=0))
        self.time_indices = time_idx

        self.intrinsics = intrinsics.to(self.device).float()
        self.cam_to_worlds = cam_to_worlds.to(self.device).float()
        self.ego_to_worlds = ego_to_worlds.to(self.device).float()
        self.timestamps = timestamps.to(self.device).float()
        self.cam_to_egos = cam_to_egos.to(self.device).float()

        self.train_intrinsics = self.intrinsics.reshape(
            self.num_timestamps, self.num_cams, 3, 3
        )[self.train_indices].reshape(-1, 3, 3)
        self.train_cam_to_worlds = self.cam_to_worlds.reshape(
            self.num_timestamps, self.num_cams, 4, 4
        )[self.train_indices].reshape(-1, 4, 4)
        self.train_ego_to_worlds = self.ego_to_worlds.reshape(
            self.num_timestamps, 4, 4
        )[self.train_indices].reshape(-1, 4, 4)
        self.train_timestamps = self.timestamps.reshape(
            self.num_timestamps, self.num_cams
        )[self.train_indices].reshape(-1)
        self.unique_train_timestamps = torch.unique(self.train_timestamps)
        self.train_time_indices = self.time_indices.reshape(
            self.num_timestamps, self.num_cams
        )[self.train_indices].reshape(-1)

        self.test_intrinsics = self.intrinsics.reshape(
            self.num_timestamps, self.num_cams, 3, 3
        )[self.test_indices].reshape(-1, 3, 3)
        self.test_cam_to_worlds = self.cam_to_worlds.reshape(
            self.num_timestamps, self.num_cams, 4, 4
        )[self.test_indices].reshape(-1, 4, 4)
        self.test_ego_to_worlds = self.ego_to_worlds.reshape(self.num_timestamps, 4, 4)[
            self.test_indices
        ].reshape(-1, 4, 4)
        self.test_timestamps = self.timestamps.reshape(
            self.num_timestamps, self.num_cams
        )[self.test_indices].reshape(-1)
        self.unique_test_timestamps = torch.unique(self.test_timestamps)
        self.test_time_indices = self.time_indices.reshape(
            self.num_timestamps, self.num_cams
        )[self.test_indices].reshape(-1)

        self.cam_ids = torch.from_numpy(np.array(cam_ids)).to(self.device).long()
        self.train_cam_ids = self.cam_ids.reshape(self.num_timestamps, self.num_cams)[
            self.train_indices
        ].reshape(-1)
        self.test_cam_ids = self.cam_ids.reshape(self.num_timestamps, self.num_cams)[
            self.test_indices
        ].reshape(-1)

    def load_rgb(self, load: bool):
        if not load:
            self.images = None
            self.img_ids = None
            self.train_images = None
            self.test_images = None
            self.train_img_ids = None
            self.test_img_ids = None
            return
        images = []
        for t in trange(self.start_idx, self.end_idx, desc="Loading images"):
            for cam_id in self.cam_idx_list:
                fname = os.path.join(self.data_dir, "images", f"{t:03d}_{cam_id}.jpg")
                rgb = Image.open(fname).convert("RGB")
                rgb = rgb.resize((self.load_size[1], self.load_size[0]), Image.BILINEAR)
                images.append(rgb)
                fname = os.path.join(self.data_dir, "velocity", f"{t:03d}_{cam_id}.png")
        self.images = torch.from_numpy(np.stack(images, axis=0)).to(self.device) / 255
        self.img_ids = torch.arange(len(self.images), device=self.device)

        self.train_images = self.images.reshape(
            self.num_timestamps, self.num_cams, self.HEIGHT, self.WIDTH, 3
        )[self.train_indices].reshape(-1, self.HEIGHT, self.WIDTH, 3)
        self.train_img_ids = self.img_ids.reshape(self.num_timestamps, self.num_cams)[
            self.train_indices
        ].reshape(-1)

        self.test_images = self.images.reshape(
            self.num_timestamps, self.num_cams, self.HEIGHT, self.WIDTH, 3
        )[self.test_indices].reshape(-1, self.HEIGHT, self.WIDTH, 3)
        self.test_img_ids = self.img_ids.reshape(self.num_timestamps, self.num_cams)[
            self.test_indices
        ].reshape(-1)

    def load_dynamic_mask(self, load: bool):
        if not load:
            self.dynamic_masks = None
            self.train_dynamic_masks = None
            self.test_dynamic_masks = None
            return
        dynamic_masks = []
        for t in trange(self.start_idx, self.end_idx, desc="Loading velocity"):
            for cam_id in self.cam_idx_list:
                fname = os.path.join(self.data_dir, "velocity", f"{t:03d}_{cam_id}.png")
                velocity_map = Image.open(fname).convert("L")
                velocity_map = velocity_map.resize(
                    (self.load_size[1], self.load_size[0]), Image.BILINEAR
                )
                dynamic_masks.append(velocity_map)
        self.dynamic_masks = torch.from_numpy(np.stack(dynamic_masks, axis=0))
        # we set the threshold to 10 to filter out the static pixels (> 1m/s)
        # velocity map is a map of the magnitude of the velocity vector * 10,
        # i.e., true_velocity = velocity_map / 10
        # dynamic_object has a velocity > 1m/s
        self.dynamic_masks = self.dynamic_masks.to(self.device) > 10

        self.train_dynamic_masks = self.dynamic_masks.reshape(
            self.num_timestamps, self.num_cams, self.HEIGHT, self.WIDTH
        )[self.train_indices].reshape(-1, self.HEIGHT, self.WIDTH)
        self.test_dynamic_masks = self.dynamic_masks.reshape(
            self.num_timestamps, self.num_cams, self.HEIGHT, self.WIDTH
        )[self.test_indices].reshape(-1, self.HEIGHT, self.WIDTH)

    def load_lidar(self, load_lidar: bool):
        if not load_lidar:
            self.lidar_rays = None
            self.train_lidar_rays, self.test_lidar_rays = None, None
            self.lidar_time_idx = None
            self.train_lidar_time_idx, self.test_lidar_time_idx = None, None
            self.lidar_ids = None
            self.train_lidar_ids, self.test_lidar_ids = None, None
            return
        lidar_rays_o, lidar_rays_d = [], []
        lidar_ranges, lidar_timestamps = [], []
        time_idx = []
        lidar_ids = []
        accumulated_num_original_rays = 0
        accumulated_num_rays = 0
        for t in trange(self.start_idx, self.end_idx, desc="Loading lidar rays"):
            lidar_xyz = np.memmap(
                os.path.join(
                    # zipping a huge training folder is too slow,
                    # so we store the lidar data in a separate folder.
                    self.data_dir.replace("training", "training_lidar"),
                    "lidar",
                    f"{t:03d}.bin",
                ),
                dtype=np.float32,
                mode="r",
            )
            # concatenate origins, points, intensity, elongation, return_indices, laser_ids (12 dims)
            #          2          5             6              7                8                  9                  10           11
            # origins: 3, points: 3, intensity: 1, elongation: 1, ground_label: 1, real_timestamp: 1, return_indices: 1, laser_ids: 1
            # laser_ids: 0: TOP, 1: FRONT, 2: SIDE_LEFT, 3: SIDE_RIGHT, 4: REAR
            lidar_info = lidar_xyz.reshape(-1, 12)
            original_length = len(lidar_info)
            accumulated_num_original_rays += original_length
            if self.only_use_top_lidar:
                lidar_info = lidar_info[lidar_info[:, 11] == 0]
            # keep the first return only
            if self.only_use_first_return:
                lidar_info = lidar_info[lidar_info[:, 10] == 0]
                accumulated_num_rays += len(lidar_info)
            lidar_origins = torch.from_numpy(lidar_info[:, :3]).float().to(self.device)
            lidar_xyz = torch.from_numpy(lidar_info[:, 3:6]).float().to(self.device)
            lidar_id = torch.from_numpy(lidar_info[:, 11]).long().to(self.device)
            # transform lidar points from lidar coordinate system to world coordinate system
            lidar_points = (
                self.ego_to_worlds[t - self.start_idx][:3, :3] @ lidar_xyz.T
                + self.ego_to_worlds[t - self.start_idx][:3, 3:4]
            ).T
            lidar_origins = (
                self.ego_to_worlds[t - self.start_idx][:3, :3] @ lidar_origins.T
                + self.ego_to_worlds[t - self.start_idx][:3, 3:4]
            ).T
            rays_d = lidar_points - lidar_origins
            ranges = torch.norm(rays_d, dim=-1, keepdim=True)
            lidar_viewdirs = rays_d / ranges
            lidar_timestamp = torch.ones(ranges.shape, device=self.device) * t
            lidar_timestamp = (lidar_timestamp - self.start_idx) / (
                self.end_idx - 1 - self.start_idx
            )
            lidar2cams = [
                get_projection_matrix(
                    self.intrinsics[(t - self.start_idx) * self.num_cams + i],
                    self.cam_to_worlds[(t - self.start_idx) * self.num_cams + i],
                )
                for i in range(self.num_cams)
            ]
            lidar2cams = torch.stack(lidar2cams).float().to(ranges.device)
            ref_pts, mask, depths = point_sampling(
                lidar_points.unsqueeze(0),
                lidar2cams.unsqueeze(0),
                img_height=self.HEIGHT,
                img_width=self.WIDTH,
            )
            ref_pts, mask, depths = (
                ref_pts.squeeze(0),
                mask.squeeze(0),
                depths.squeeze(0),
            )
            # convert to image coordinates in (w, h) format (not (h, w))
            ref_pts[..., 0] = (ref_pts[..., 0] + 1) / 2 * (self.WIDTH - 1)
            ref_pts[..., 1] = (ref_pts[..., 1] + 1) / 2 * (self.HEIGHT - 1)
            ref_pts = ref_pts.round().long()
            if self.only_keep_lidar_rays_on_images:
                valid_mask = mask.any(dim=0)
            else:
                valid_mask = torch.ones_like(mask[0]).bool()
            if self.only_keep_lidar_rays_within_truncated_range is not None:
                # To be studied: use lidar_points or lidar_xyz?
                valid_mask = valid_mask & (
                    lidar_xyz[..., 0] > self.only_keep_lidar_rays_within_truncated_range
                )
            lidar_ids.append(lidar_id[valid_mask])
            lidar_rays_o.append(lidar_origins[valid_mask])
            lidar_rays_d.append(lidar_viewdirs[valid_mask])
            lidar_ranges.append(ranges[valid_mask])
            lidar_timestamps.append(lidar_timestamp[valid_mask])
            time_idx.append(torch.ones_like(ranges[valid_mask]) * t)
        time_idx = torch.cat(time_idx, dim=0).long().to(self.device) - self.start_idx
        self.lidar_time_idx = time_idx.squeeze(-1)
        self.lidar_ids = torch.cat(lidar_ids, dim=0).long().to(self.device)
        self.lidar_rays = torch.cat(
            [
                torch.concat(lidar_rays_o, dim=0),
                torch.concat(lidar_rays_d, dim=0),
                torch.concat(lidar_ranges, dim=0),
                torch.concat(lidar_timestamps, dim=0),
            ],
            dim=-1,
        ).to(self.device)
        logger.info(
            f"Accumulated {accumulated_num_rays} rays out of {accumulated_num_original_rays} rays ({accumulated_num_rays / accumulated_num_original_rays * 100:.2f}%)."
        )
        if self.only_keep_lidar_rays_on_images:
            logger.info(
                f"{len(self.lidar_rays)} lidar rays ({len(self.lidar_rays) / accumulated_num_rays * 100:.2f}%) are within the images."
            )
        else:
            logger.info(
                f"{len(self.lidar_rays)} lidar rays ({len(self.lidar_rays) / accumulated_num_rays * 100:.2f}%) are loaded (including those outside the images)."
            )

        # Ensure self.train_indices is a tensor
        train_indices_tensor = torch.tensor(self.train_indices).to(time_idx.device)
        train_mask = (time_idx == train_indices_tensor).any(dim=-1)
        # Use the mask to index self.lidar_rays
        self.train_lidar_rays = self.lidar_rays[train_mask]
        self.train_lidar_time_idx = self.lidar_time_idx[train_mask]
        self.train_lidar_ids = self.lidar_ids[train_mask]

        test_indices_tensor = torch.tensor(self.test_indices).to(time_idx.device)
        test_mask = (time_idx == test_indices_tensor).any(dim=-1)
        self.test_lidar_rays = self.lidar_rays[test_mask]
        self.test_lidar_time_idx = self.lidar_time_idx[test_mask]
        self.test_lidar_ids = self.lidar_ids[test_mask]

    def load_sky_mask(self, load: bool = True):
        if not load:
            self.sky_masks = None
            self.train_sky_masks = None
            self.test_sky_masks = None
            return
        sky_masks = []
        for t in trange(self.start_idx, self.end_idx, desc="Loading sky masks"):
            for cam_id in self.cam_idx_list:
                fname = os.path.join(
                    self.data_dir, "sky_masks", f"{t:03d}_{cam_id}.png"
                )
                sky_mask = Image.open(fname).convert("L")
                sky_mask = sky_mask.resize(
                    (self.load_size[1], self.load_size[0]), Image.NEAREST
                )
                sky_masks.append(np.array(sky_mask) > 0)
        sky_masks = torch.from_numpy(np.stack(sky_masks, axis=0)).bool()
        self.sky_masks = sky_masks.to(self.device)

        self.train_sky_masks = self.sky_masks.reshape(
            self.num_timestamps, self.num_cams, self.HEIGHT, self.WIDTH
        )[self.train_indices].reshape(-1, self.HEIGHT, self.WIDTH)
        self.test_sky_masks = self.sky_masks.reshape(
            self.num_timestamps, self.num_cams, self.HEIGHT, self.WIDTH
        )[self.test_indices].reshape(-1, self.HEIGHT, self.WIDTH)

    def load_dino(self, load: bool, dino_model_type: str):
        if not load:
            self.dino_features = None
            self.dino_scale = None
            self.dino_dimension_reduction_mat = None
            self.color_norm_min = None
            self.color_norm_max = None
            self.train_dino_features = None
            self.test_dino_features = None
            self.dino_original_to_reduced = None
            return
        # Like the lidar data, we store the DINO features in a separate folde,
        # so that we can zip/delete/move the this large folder easily.
        if "v2" in dino_model_type:
            dino_folder = "training_dinov2"
        elif "clip" in dino_model_type:
            dino_folder = "training_clip"
        else:
            dino_folder = "training_dino"
        dino_features = []
        for t in trange(self.start_idx, self.end_idx, desc="Loading DINO features"):
            for cam_id in self.cam_idx_list:
                fname = os.path.join(
                    self.data_dir.replace("training", dino_folder),
                    "dino",
                    f"{t:03d}_{cam_id}.npy",
                )
                # mmap_mode="r" is necessary to avoid memory overflow
                dino_feature = np.load(fname, mmap_mode="r")
                dino_features.append(dino_feature)
        self.dino_features = torch.from_numpy(np.stack(dino_features, axis=0)).float()
        # dino_scale is used to convert the image coordinates to DINO feature coordinates.
        # resize dino features to (H, W) using bilinear interpolation is infeasible.
        # imagine a feature array of shape (200, 640, 960, 768), it's too large to fit in GPU memory.
        self.dino_scale = (
            self.dino_features.shape[1] / self.load_size[0],
            self.dino_features.shape[2] / self.load_size[1],
        )
        # compute dino visualization matrix
        # we compute the first 3 principal components of the DINO features as the color
        logger.info(f"Loaded {self.dino_features.shape} DINO features.")
        logger.info(f"DINO feature scale: {self.dino_scale}")
        logger.info(f"Computing DINO PCA...")
        C = self.dino_features.shape[-1]
        temp_feats = self.dino_features.reshape(-1, C)
        # no need to compute PCA on the entire dataset
        max_elements_to_compute_pca = min(100000, temp_feats.shape[0])
        selected_features = temp_feats[
            np.random.choice(
                temp_feats.shape[0], max_elements_to_compute_pca, replace=False
            )
        ]
        if self.target_dino_dim is not None:
            logger.info(f"Reducing DINO features to {self.target_dino_dim} dimensions.")
            U, S, reduction_mat = torch.pca_lowrank(
                selected_features, q=self.target_dino_dim, niter=20
            )
            selected_features = selected_features @ reduction_mat
            self.dino_features = self.dino_features @ reduction_mat
            C = self.dino_features.shape[-1]
            # normalize the reduced features to [0, 1]
            dino_min = self.dino_features.reshape(-1, C).min(dim=0)[0]
            dino_max = self.dino_features.reshape(-1, C).max(dim=0)[0]
            self.dino_features = (self.dino_features - dino_min) / (dino_max - dino_min)
            selected_features = (selected_features - dino_min) / (dino_max - dino_min)
            self.dino_original_to_reduced = reduction_mat
        else:
            self.dino_original_to_reduced = None
        reduction_mat, color_norm_min, color_norm_max = get_robust_pca(
            selected_features.to(self.device)
        )
        # save this reduction matrix for visualizing other
        # DINO features (e.g., predicted/novel-view dinos)
        self.dino_dimension_reduction_mat = reduction_mat
        self.color_norm_min = color_norm_min
        self.color_norm_max = color_norm_max
        del temp_feats, selected_features
        logger.info(
            "DINO PCA computed, shape: {}".format(
                self.dino_dimension_reduction_mat.shape
            )
        )
        dino_hwc = self.dino_features.shape[1:]
        self.train_dino_features = self.dino_features.reshape(
            self.num_timestamps,
            self.num_cams,
            *dino_hwc,
        )[self.train_indices].reshape(-1, *dino_hwc)
        self.test_dino_features = self.dino_features.reshape(
            self.num_timestamps,
            self.num_cams,
            *dino_hwc,
        )[self.test_indices].reshape(-1, *dino_hwc)

    def __len__(self):
        return len(self.cam_to_worlds)


class WaymoSequenceSplit(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    def __init__(
        self,
        images: Tensor,
        img_ids: Tensor,
        cam_to_worlds: Tensor,
        intrinsics: Tensor,
        cam_to_egos: Optional[Tensor] = None,
        ego_to_worlds: Optional[Tensor] = None,
        cam_ids: Optional[Tensor] = None,
        timestamps: Optional[Tensor] = None,
        time_indices: Optional[Tensor] = None,
        lidar_rays: Optional[Tensor] = None,
        lidar_time_idx: Optional[Tensor] = None,
        lidar_ids: Optional[Tensor] = None,
        sky_masks: Optional[Tensor] = None,
        dynamic_masks: Optional[Tensor] = None,
        dino_features: Optional[Tensor] = None,
        dino_scale: Optional[Tuple[float, float]] = None,
        dino_dimension_reduction_mat: Optional[Tensor] = None,
        color_norm_min: Optional[Tensor] = None,
        color_norm_max: Optional[Tensor] = None,
        num_cams: int = 5,
        num_rays: int = None,
        batch_over_images: bool = True,
        image_size: Tuple[int, int] = (1280, 1920),
        downscale: float = 1.0,
        device: torch.device = torch.device("cpu"),
        split: str = "train",
        buffer_ratio: float = -1,
        buffer_downscale: int = 8,
    ):
        super().__init__()
        self.images = images
        self.img_ids = img_ids
        self.cam_to_worlds = cam_to_worlds
        self.ego_to_worlds = ego_to_worlds
        self.num_timestamps = len(cam_to_worlds) // num_cams
        self.num_cams = num_cams
        self.intrinsics = intrinsics
        self.cam_to_egos = cam_to_egos
        self.cam_ids = cam_ids
        self.timestamps = timestamps
        self.time_indices = time_indices
        self.unique_time_indices = torch.unique(self.time_indices).sort()[0]
        self.lidar_rays = lidar_rays
        self.lidar_time_idx = lidar_time_idx
        self.lidar_ids = lidar_ids
        self.sky_masks = sky_masks
        self.dynamic_masks = dynamic_masks
        self.dino_features = dino_features
        self.dino_scale = dino_scale
        self.dino_dimension_reduction_mat = dino_dimension_reduction_mat
        self.color_norm_min = color_norm_min
        self.color_norm_max = color_norm_max
        self.num_rays = num_rays
        self.num_lidar_rays = num_rays
        self.batch_over_images = batch_over_images
        self.device = device
        self.training = (num_rays is not None) and split == "train"
        self.split = split
        self.HEIGHT, self.WIDTH = image_size
        self.downscale = downscale
        self.downscaled_height = int(self.HEIGHT / self.downscale)
        self.downscaled_width = int(self.WIDTH / self.downscale)
        self.buffer_downscale = buffer_downscale
        self.buffer_ratio = buffer_ratio
        self.build_pixel_error_buffer()

    def build_pixel_error_buffer(self):
        if self.buffer_ratio > 0:
            self.pixel_error_maps = torch.ones(
                (
                    len(self.cam_to_worlds),
                    self.HEIGHT // self.buffer_downscale,
                    self.WIDTH // self.buffer_downscale,
                ),
                dtype=torch.float32,
                device=self.device,
            ).flatten()
            self.pixel_error_buffered = False
            logger.info(
                "Successfully built pixel error buffer (log2(num_pixels) = {:.2f}).".format(
                    np.log2(len(self.pixel_error_maps))
                )
            )
        else:
            self.pixel_error_maps = None
            logger.info(
                "Not building pixel error buffer because buffer_ratio is too small."
            )

    def __len__(self):
        return len(self.cam_to_worlds)

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def update_num_lidar_rays(self, num_lidar_rays):
        self.num_lidar_rays = num_lidar_rays

    def update_pixel_error_maps(self, render_results):
        if self.pixel_error_maps is None:
            logger.info("Not updating pixel error buffer because it's not built.")
            return
        gt_rgbs = render_results["gt_rgbs"]
        pred_rgbs = render_results["rgbs"]
        gt_rgbs = torch.from_numpy(np.stack(gt_rgbs, axis=0)).to(self.device)
        pred_rgbs = torch.from_numpy(np.stack(pred_rgbs, axis=0)).to(self.device)
        gt_rgbs = gt_rgbs.reshape(-1, 3)
        pred_rgbs = pred_rgbs.reshape(-1, 3)
        pixel_error_maps = torch.abs(gt_rgbs - pred_rgbs).mean(dim=-1)
        assert pixel_error_maps.shape == self.pixel_error_maps.shape
        if "dynamic_opacities" in render_results:
            if len(render_results["dynamic_opacities"]) > 0:
                dynamic_opacity = render_results["dynamic_opacities"]
                dynamic_opacity = (
                    torch.from_numpy(np.stack(dynamic_opacity, axis=0))
                    .to(self.device)
                    .reshape(-1)
                )
                # we prioritize the dynamic objects by multiplying the error by 10
                pixel_error_maps[dynamic_opacity > 0.2] *= 10
        self.pixel_error_maps: Tensor = pixel_error_maps
        self.pixel_error_maps = (
            self.pixel_error_maps - self.pixel_error_maps.min()
        ) / (self.pixel_error_maps.max() - self.pixel_error_maps.min())
        self.pixel_error_buffered = True
        logger.info("Successfully updated pixel error buffer")

    def visualize_pixel_sample_weights(self, index):
        frames = (
            self.pixel_error_maps.detach()
            .cpu()
            .numpy()
            .reshape(len(self), self.HEIGHT // self.buffer_downscale, -1)[index]
        )
        min_value = self.pixel_error_maps.min().item()
        max_value = self.pixel_error_maps.max().item()
        frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
        frames = np.concatenate(frames, axis=1)
        frames = (frames - min_value) / (max_value - min_value)
        return np.uint8(frames * 255)

    def get_pixel_sample_weights_video(self):
        assert self.buffer_ratio > 0, "buffer_ratio must be > 0"
        maps = []
        loss_maps = (
            self.pixel_error_maps.detach()
            .cpu()
            .numpy()
            .reshape(len(self), self.HEIGHT // self.buffer_downscale, -1)
        )
        for i in range(len(self)):
            maps.append(loss_maps[i])
        return maps

    @torch.no_grad()
    def __getitem__(self, index):
        if self.training:
            return self.fetch_train_pixel_data(index)
        else:
            return self.fetch_render_pixel_data(index)

    def get_rays(
        self, x: Tensor, y: Tensor, c2w: Tensor, intrinsic: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate rays from the camera center to the pixel"""
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - intrinsic[:, 0, 2] + 0.5) / intrinsic[:, 0, 0],
                    (y - intrinsic[:, 1, 2] + 0.5) / intrinsic[:, 1, 1],
                ],
                dim=-1,
            ),
            (0, 1),
            value=1.0,
        )  # [num_rays, 3]
        # rotate the camera rays w.r.t. the camera pose
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        # TODO: not sure if we still need direction_norm
        direction_norm = torch.linalg.norm(directions, dim=-1, keepdims=True)
        # normalize the ray directions
        viewdirs = directions / (direction_norm + 1e-8)
        return origins, viewdirs, direction_norm

    def get_dino_feat(
        self, x: Tensor, y: Tensor, img_id: Tensor, downscale: float = 1.0
    ) -> Tensor:
        # we compute the nearest DINO feature for each pixel
        # map (x, y) in the (W, H) space to (x * dino_scale[0], y * dino_scale[1]) in the (W//patch_size, H//patch_size) space
        dino_y = (y * self.dino_scale[0] * downscale).long()
        dino_x = (x * self.dino_scale[1] * downscale).long()
        # dino_feats are in CPU memory (because they are huge), so we need to move them to GPU
        dino_feat = self.dino_features[img_id.cpu(), dino_y.cpu(), dino_x.cpu()].cuda()
        return dino_feat

    def sample_pixels_with_high_loss(self, num_rays):
        sampled_indices = torch.multinomial(self.pixel_error_maps, num_rays, False)
        img_idx, y, x = idx_to_3d(
            sampled_indices,
            self.HEIGHT // self.buffer_downscale,
            self.WIDTH // self.buffer_downscale,
        )
        # Upscale to the original resolution
        y, x = (y * self.buffer_downscale).long(), (x * self.buffer_downscale).long()

        # Add a random offset
        y_offset = torch.randint(
            0, self.buffer_downscale, (num_rays,), device=self.images.device
        )
        x_offset = torch.randint(
            0, self.buffer_downscale, (num_rays,), device=self.images.device
        )

        y += y_offset
        x += x_offset

        # Clamp to ensure coordinates don't exceed the image bounds
        y = torch.clamp(y, 0, self.HEIGHT - 1)
        x = torch.clamp(x, 0, self.WIDTH - 1)
        return img_idx, y, x

    def sample_random_pixels(self, num_rays: int, sample_interval: float = 1.0):
        img_id = torch.randint(
            0,
            len(self.images),
            size=(num_rays,),
            device=self.images.device,
        )
        # sample pixels
        x = (
            torch.randint(
                0,
                int(self.WIDTH // sample_interval),
                size=(num_rays,),
                device=self.images.device,
            )
            * sample_interval
        )
        y = (
            torch.randint(
                0,
                int(self.HEIGHT // sample_interval),
                size=(num_rays,),
                device=self.images.device,
            )
            * sample_interval
        )
        x, y = x.long(), y.long()
        return img_id, y, x

    def fetch_train_pixel_data(self, index, sample_interval: float = 1.0):
        assert self.training, "This function is only used for training."
        num_rays = self.num_rays
        if self.buffer_ratio > 0 and self.pixel_error_buffered:
            num_prioritized_rays = int(num_rays * self.buffer_ratio)
            num_random_rays = num_rays - num_prioritized_rays
            random_img_idx, random_y, random_x = self.sample_random_pixels(
                num_random_rays, sample_interval
            )
            loss_img_idx, loss_y, loss_x = self.sample_pixels_with_high_loss(
                num_prioritized_rays
            )
            img_id = torch.cat([random_img_idx, loss_img_idx], dim=0)
            y = torch.cat([random_y, loss_y], dim=0)
            x = torch.cat([random_x, loss_x], dim=0)
        else:
            img_id, y, x = self.sample_random_pixels(num_rays, sample_interval)

        rgb = self.images[img_id, y, x]  # (num_rays, 3)
        pix_coords = torch.stack([y / self.HEIGHT, x / self.WIDTH], dim=-1)
        sky_mask = (
            self.sky_masks[img_id, y, x].float() if self.sky_masks is not None else None
        )
        dino_feat = (
            self.get_dino_feat(x, y, img_id) if self.dino_features is not None else None
        )
        # generate rays
        c2w = self.cam_to_worlds[img_id]  # (num_rays, 3, 4)
        intrinsic = self.intrinsics[img_id]  # (num_rays, 3, 3)
        origins, viewdirs, direction_norm = self.get_rays(x, y, c2w, intrinsic)
        timestamp = self.timestamps[img_id]
        cam_idx = self.cam_ids[img_id]
        img_indexs = img_id * torch.ones_like(cam_idx)
        if self.buffer_ratio > 0:
            # compute 1d index for each ray
            buffer_indices = (
                img_id
                * (self.HEIGHT // self.buffer_downscale)
                * (self.WIDTH // self.buffer_downscale)
                + (y // self.buffer_downscale) * (self.WIDTH // self.buffer_downscale)
                + (x // self.buffer_downscale)
            )
        else:
            buffer_indices = None
        data = {
            "pixels": rgb,  # [h, w, 3] or [num_rays, 3]
            "origins": origins,  # [h, w, 3] or [num_rays, 3]
            "viewdirs": viewdirs,  # [h, w, 3] or [num_rays, 3]
            "direction_norm": direction_norm,  # [h, w] or [num_rays,]
            "timestamp": timestamp,  # [h, w] or [num_rays,]
            "cam_idx": cam_idx,  # [h, w] or [num_rays,]
            "img_idx": img_indexs,  # [h, w] or [num_rays,]
            "sky_mask": sky_mask,  # [h, w] or [num_rays,]
            "dino_feat": dino_feat,  # [h, w, dino_feat_dim] (which is huge) or [num_rays, dino_feat_dim]
            "pix_coords": pix_coords,
            "buffer_indices": buffer_indices,
        }
        return {k: v.cuda() for k, v in data.items() if v is not None}

    def fetch_render_pixel_data(self, index, custom_downscale: float = None):
        img_id = torch.tensor([index], device=self.images.device).long()
        rgb = self.images[img_id].squeeze()  # [H, W, 3]
        if custom_downscale is not None:
            downscaled_height = int(self.HEIGHT / custom_downscale)
            downscaled_width = int(self.WIDTH / custom_downscale)
            downscale = custom_downscale
        else:
            downscaled_height = self.downscaled_height
            downscaled_width = self.downscaled_width
            downscale = self.downscale
        if downscale != 1.0:
            rgb = (
                TF.resize(
                    rgb.unsqueeze(0).permute(0, 3, 1, 2),
                    (downscaled_height, downscaled_width),
                    antialias=True,
                    interpolation=Image.BILINEAR,
                )
                .squeeze(0)
                .permute(1, 2, 0)
            )
        x, y = torch.meshgrid(
            torch.arange(downscaled_width),
            torch.arange(downscaled_height),
            indexing="xy",
        )
        x = x.flatten().to(self.images.device)
        y = y.flatten().to(self.images.device)
        pix_coords = torch.stack(
            [y / downscaled_height, x / downscaled_width], dim=-1
        ).reshape(downscaled_height, downscaled_width, 2)

        if self.lidar_rays is not None:
            # resizing is not good. we reproject lidar points to the image plane
            lidar_dict = self.fetch_render_lidar_data(
                index // self.num_cams,
                return_rays_on_image=True,
            )
            lidar_xyz = (
                lidar_dict["lidar_origins"]
                + lidar_dict["lidar_viewdirs"] * lidar_dict["lidar_ranges"][:, None]
            )
            projection = torch.eye(4, device=self.device).float()
            projection[:3, :3] = self.intrinsics[index] / downscale
            projection[2, 2] = 1.0
            projection = projection @ self.cam_to_worlds[index].inverse()
            # project lidar onto image plane:
            ref_pts, mask, depths = point_sampling(
                lidar_xyz.unsqueeze(0),
                projection.unsqueeze(0).unsqueeze(0),
                img_height=downscaled_height,
                img_width=downscaled_width,
            )
            ref_pts, mask, depths = (
                ref_pts.squeeze(),
                mask.squeeze(),
                depths.squeeze(),
            )
            # convert to image coordinates in (w, h) format (not (h, w))
            ref_pts[..., 0] = (ref_pts[..., 0] + 1) / 2 * (downscaled_width - 1)
            ref_pts[..., 1] = (ref_pts[..., 1] + 1) / 2 * (downscaled_height - 1)
            ref_pts = ref_pts.round().long()
            # only keep the points that are visible in the image
            valid_pts = ref_pts[mask]
            # initialize the range_img with -1 (invalid depth)
            range_img = torch.full(
                (downscaled_height, downscaled_width),
                -1,
                dtype=torch.float32,
                device=depths.device,
            )
            # fill in the range_img
            range_img[valid_pts[:, 1], valid_pts[:, 0]] = depths[mask]
        else:
            range_img = None

        if self.sky_masks is not None:
            sky_mask = self.sky_masks[img_id]
            sky_mask = TF.resize(
                sky_mask, (downscaled_height, downscaled_width)
            ).squeeze(0)
        else:
            sky_mask = None
        if self.dino_features is not None:
            dino_feat = self.get_dino_feat(x, y, img_id, downscale)
            dino_feat = dino_feat.reshape(downscaled_height, downscaled_width, -1)
        else:
            dino_feat = None
        # generate rays
        c2w = self.cam_to_worlds[img_id]
        # note that the 1's will also be downscaled
        intrinsic = self.intrinsics[img_id]
        # scale the intrinsic matrix
        intrinsic = intrinsic / downscale
        intrinsic[:, 2, 2] = 1.0
        # compute the ray from the camera center to the pixel
        origins, viewdirs, direction_norm = self.get_rays(x, y, c2w, intrinsic)
        origins = origins.reshape(downscaled_height, downscaled_width, 3)
        viewdirs = viewdirs.reshape(downscaled_height, downscaled_width, 3)
        direction_norm = direction_norm.reshape(downscaled_height, downscaled_width)
        timestamp = torch.full(
            (downscaled_height, downscaled_width),
            self.timestamps[img_id[0]],
            device=self.images.device,
        )
        cam_idx = torch.full(
            (downscaled_height, downscaled_width),
            self.cam_ids[img_id[0]],
            device=self.images.device,
        )
        img_indexs = torch.full(
            (downscaled_height, downscaled_width),
            img_id.item(),
            device=self.images.device,
        )
        if self.dynamic_masks is not None:
            dynamic_mask = self.dynamic_masks[img_id]
            dynamic_mask = (
                TF.resize(dynamic_mask, (downscaled_height, downscaled_width))
                .squeeze(0)
                .bool()
            )

        else:
            dynamic_mask = None
        data = {
            "pixels": rgb,  # [h, w, 3] or [num_rays, 3]
            "origins": origins,  # [h, w, 3] or [num_rays, 3]
            "viewdirs": viewdirs,  # [h, w, 3] or [num_rays, 3]
            "direction_norm": direction_norm,  # [h, w] or [num_rays,]
            "timestamp": timestamp,  # [h, w] or [num_rays,]
            "cam_idx": cam_idx,  # [h, w] or [num_rays,]
            "img_idx": img_indexs,  # [h, w] or [num_rays,]
            "sky_mask": sky_mask,  # [h, w] or [num_rays,]
            "dino_feat": dino_feat,  # [h, w, dino_feat_dim] (which is huge) or [num_rays, dino_feat_dim]
            "pix_coords": pix_coords,
            "dynamic_mask": dynamic_mask,
        }
        return {k: v.cuda() for k, v in data.items() if v is not None}

    def fetch_train_lidar_data(self, index):
        num_rays = self.num_lidar_rays
        lidar_indices = torch.randint(
            0,
            len(self.lidar_rays),
            size=(num_rays,),
            device=self.lidar_rays.device,
        )
        # (N, 8): (origins, directions, ranges, timestamp)
        lidar_rays = self.lidar_rays[lidar_indices]
        lidar_origins, lidar_viewdirs, lidar_ranges = (
            lidar_rays[:, :3],
            lidar_rays[:, 3:6],
            lidar_rays[:, 6],
        )
        lidar_timestamps = lidar_rays[:, 7]
        lidar_xyz = lidar_origins + lidar_viewdirs * lidar_ranges[:, None]
        data = {
            "lidar_origins": lidar_origins,  # [num_rays, 3]
            "lidar_viewdirs": lidar_viewdirs,  # [num_rays, 3]
            "lidar_ranges": lidar_ranges,  # [num_rays,]
            "lidar_timestamp": lidar_timestamps,  # [num_rays,]
            "lidar_xyz": lidar_xyz,  # [num_rays, 3]
            "lidar_indices": lidar_indices,  # [num_rays,]
        }
        return {k: v.cuda() for k, v in data.items() if v is not None}

    def fetch_render_lidar_data(
        self,
        index,  # index of the TIMESTAMP
        return_rays_on_image: bool = False,
        lidar_id: int = None,
        drop_rays_beyond_this_range: float = None,
    ):
        selector = self.lidar_time_idx == self.time_indices[index * self.num_cams]
        lidar_rays = self.lidar_rays[selector]

        if lidar_id is not None:
            lidar_rays = lidar_rays[self.lidar_ids[selector] == lidar_id]

        lidar_origins, lidar_viewdirs, lidar_ranges = (
            lidar_rays[:, :3],
            lidar_rays[:, 3:6],
            lidar_rays[:, 6],
        )
        lidar_timestamps = lidar_rays[:, 7]
        lidar_xyz = lidar_origins + lidar_viewdirs * lidar_ranges[:, None]
        if return_rays_on_image:
            # note that lidar_xyz is modified in-place
            mask = self.get_valid_lidar_ray_mask_on_image(
                lidar_xyz, [index * self.num_cams + i for i in range(self.num_cams)]
            )
            lidar_origins, lidar_viewdirs, lidar_ranges = (
                lidar_origins[mask],
                lidar_viewdirs[mask],
                lidar_ranges[mask],
            )
            lidar_xyz = lidar_xyz[mask]
            lidar_timestamps = lidar_timestamps[mask]
        if drop_rays_beyond_this_range is not None:
            mask = lidar_ranges < drop_rays_beyond_this_range
            lidar_origins, lidar_viewdirs, lidar_ranges = (
                lidar_origins[mask],
                lidar_viewdirs[mask],
                lidar_ranges[mask],
            )
            lidar_xyz = lidar_xyz[mask]
            lidar_timestamps = lidar_timestamps[mask]
        data_dict = {
            "lidar_origins": lidar_origins,  # [num_rays, 3]
            "lidar_viewdirs": lidar_viewdirs,  # [num_rays, 3]
            "lidar_ranges": lidar_ranges,  # [num_rays,]
            "lidar_timestamp": lidar_timestamps,  # [num_rays,]
            "lidar_xyz": lidar_xyz,  # [num_rays, 3]
        }
        return {k: v.cuda() for k, v in data_dict.items() if v is not None}

    def project_lidar_to_image(
        self, points: Tensor, index: Union[int, List[int]], return_mask: bool = False
    ):
        if isinstance(index, int):
            index = [index]
        index = torch.tensor(index, device=self.device).long()
        projection = (
            torch.eye(4, device=self.device)
            .float()
            .unsqueeze(0)
            .repeat(len(index), 1, 1)
        )
        projection[:, :3, :3] = self.intrinsics[index] / self.downscale
        projection[:, 2, 2] = 1.0
        projection = projection @ self.cam_to_worlds[index].inverse()
        # project lidar onto image plane:
        ref_pts, mask, depths = point_sampling(
            points.unsqueeze(0),
            projection.unsqueeze(0),
            img_height=self.downscaled_height,
            img_width=self.downscaled_width,
        )
        ref_pts, mask, depths = ref_pts.squeeze(), mask.squeeze(), depths.squeeze()
        # convert to image coordinates in (w, h) format (not (h, w))
        ref_pts[..., 0] = (ref_pts[..., 0] + 1) / 2 * (self.WIDTH - 1)
        ref_pts[..., 1] = (ref_pts[..., 1] + 1) / 2 * (self.HEIGHT - 1)
        ref_pts = ref_pts.round().long()
        # only keep the points that are visible in the image
        if not return_mask:
            return ref_pts[mask]
        else:
            return ref_pts, mask

    def get_valid_lidar_ray_mask_on_image(
        self, points: Tensor, index: Union[int, List[int]], return_pts: bool = False
    ):
        if isinstance(index, int):
            index = [index]
        index = torch.tensor(index, device=self.device).long()
        projection = (
            torch.eye(4, device=self.device)
            .float()
            .unsqueeze(0)
            .repeat(len(index), 1, 1)
        )
        projection[:, :3, :3] = self.intrinsics[index] / self.downscale
        projection[:, 2, 2] = 1.0
        projection = projection @ self.cam_to_worlds[index].inverse()
        # project lidar onto image plane:
        ref_pts, mask, depths = point_sampling(
            points.unsqueeze(0),
            projection.unsqueeze(0),
            img_height=self.downscaled_height,
            img_width=self.downscaled_width,
        )
        if not return_pts:
            return mask.squeeze().any(dim=0)
        else:
            ref_pts = (ref_pts + 1) / 2
            ref_pts[..., 0] *= self.WIDTH
            ref_pts[..., 1] *= self.HEIGHT
            return mask, ref_pts

    def ego_to_world(self, lidar_xyz_ego: Tensor, index: int):
        # make lidar points homogeneous
        lidar_xyz_ego = F.pad(lidar_xyz_ego, (0, 1), value=1.0)
        lidar_xyz_world = (self.ego_to_worlds[index] @ lidar_xyz_ego.T).T
        return lidar_xyz_world[:, :3]

    def world_to_ego(self, lidar_xyz_world: Tensor, index: int):
        if not isinstance(lidar_xyz_world, torch.Tensor):
            lidar_xyz_world = torch.tensor(
                lidar_xyz_world, dtype=torch.float32, device=self.device
            )
        # make lidar points homogeneous
        lidar_xyz_world = F.pad(lidar_xyz_world, (0, 1), value=1.0)
        lidar_xyz_ego = (self.ego_to_worlds[index].inverse() @ lidar_xyz_world.T).T
        return lidar_xyz_ego[:, :3]

    def world_to_cam(self, lidar_xyz_world: Tensor, index: int):
        if not isinstance(lidar_xyz_world, torch.Tensor):
            lidar_xyz_world = torch.tensor(
                lidar_xyz_world, dtype=torch.float32, device=self.device
            )
        # make lidar points homogeneous
        lidar_xyz_world = F.pad(lidar_xyz_world, (0, 1), value=1.0)
        lidar_xyz_cam = (self.cam_to_worlds[index].inverse() @ lidar_xyz_world.T).T
        return lidar_xyz_cam[:, :3]

    def get_gt_videos(self):
        """Get the ground truth videos for visualization"""
        gt_rgbs, lidar_depths, sky_masks, dino_colors = [], [], [], []
        for i in trange(len(self), desc="Computing ground truth videos"):
            data_dict = self.fetch_render_pixel_data(i)
            gt_rgbs.append(data_dict["pixels"].cpu().numpy())
            # if self.lidar_flow is not None:
            #     lidar_dict = self.fetch_render_lidar_data(i // self.num_cams)
            # if self.lidar_rays is not None:
            #     lidar_dict = self.fetch_render_lidar_data(i // self.num_cams)
            #     valid_pts = self.project_lidar_to_image(
            #         lidar_dict["lidar_xyz"][non_ground_mask], i
            #     )
            #     depth_maps = torch.full(
            #         (self.HEIGHT, self.WIDTH),
            #         -1,
            #         dtype=torch.float32,
            #         device=lidar_dict["lidar_xyz"].device,
            #     )
            #     depth_maps[valid_pts[:, 1], valid_pts[:, 0]] = 1.0
            #     lidar_depths.append(depth_maps.cpu().numpy())
            # if self.lidar_rays is not None:
            #     if i > 3:
            #         lidar_dict = self.fetch_render_lidar_data(i // self.num_cams)
            #         next_lidar_dict = self.fetch_render_lidar_data(
            #             i // self.num_cams - 1
            #         )
            #         lidar_dict["lidar_xyz"] = lidar_dict["lidar_xyz"][
            #             lidar_dict["lidar_ground_labels"] == 0
            #         ]
            #         next_lidar_dict["lidar_xyz"] = next_lidar_dict["lidar_xyz"][
            #             next_lidar_dict["lidar_ground_labels"] == 0
            #         ]
            #         dynamic_points = self.obtain_dynamic_points(
            #             lidar_dict["lidar_xyz"], next_lidar_dict["lidar_xyz"]
            #         )
            #         valid_pts = self.project_lidar_to_image(dynamic_points, i)
            #         depth_maps = torch.full(
            #             (self.HEIGHT, self.WIDTH),
            #             -1,
            #             dtype=torch.float32,
            #             device=dynamic_points.device,
            #         )
            #         depth_maps[valid_pts[:, 1], valid_pts[:, 0]] = 1.0
            #         lidar_depths.append(depth_maps.cpu().numpy())
            #     else:
            #         lidar_depths.append(
            #             torch.full(
            #                 (self.HEIGHT, self.WIDTH),
            #                 -1,
            #                 dtype=torch.float32,
            #                 device=self.device,
            #             )
            #             .cpu()
            #             .numpy()
            #         )

            if self.sky_masks is not None:
                sky_masks.append(data_dict["sky_mask"].squeeze().cpu().numpy())
            if self.dino_features is not None:
                # ------- the following code is for masking out the sky region when computing PCA -------
                # dino_color = torch.zeros((self.HEIGHT, self.WIDTH, 3), device=self.device)
                # non_sky_dino_feats = data_dict["dino_feat"][~data_dict["sky_mask"].squeeze()]
                # non_sky_dino_feats = non_sky_dino_feats @ self.dino_dimension_reduction_mat
                # non_sky_dino_color = (non_sky_dino_feats - non_sky_dino_feats.min()) / (non_sky_dino_feats.max() - non_sky_dino_feats.min())
                # dino_color[~data_dict["sky_mask"].squeeze()] = non_sky_dino_color
                # dino_colors.append(dino_color.cpu().numpy())

                # ------- the following code is for computing PCA on the entire image -------
                dino_color = data_dict[
                    "dino_feat"
                ] @ self.dino_dimension_reduction_mat.to(data_dict["dino_feat"])
                dino_color = (dino_color - self.color_norm_min.to(dino_color)) / (
                    self.color_norm_max - self.color_norm_min
                ).to(dino_color)
                dino_color = torch.clamp(dino_color, 0, 1)
                dino_colors.append(dino_color.cpu().numpy())

        video_dict = {}
        if len(gt_rgbs) > 0:
            video_dict["gt_rgbs"] = gt_rgbs
        if len(lidar_depths) > 0:
            video_dict["gt_lidar_depths"] = lidar_depths
        if len(sky_masks) > 0:
            video_dict["gt_sky_masks"] = sky_masks
        if len(dino_colors) > 0:
            video_dict["gt_dino_colors"] = dino_colors
        return video_dict

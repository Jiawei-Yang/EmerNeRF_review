import argparse
import itertools
import json
import logging
import os
import time
from typing import Callable, List, Optional

import imageio
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from datasets import WaymoSequenceLoader
from datasets.metrics import compute_psnr, compute_valid_depth_rmse
import loss
from radiance_fields import (
    DensityField,
    RadianceField,
    build_density_field,
    build_radiance_field_from_cfg,
)
from radiance_fields.render_utils import render_rays
from radiance_fields.video_utils import (
    render_lidars,
    render_pixels,
    save_lidar_simulation,
    save_videos,
)
from third_party.dino_extractor import extract_and_save_dino_features
from third_party.nerfacc_prop_net import PropNetEstimator, get_proposal_requires_grad_fn
from utils.logging import MetricLogger, setup_logging
from utils.misc import fix_random_seeds

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

render_keys = [
    "gt_rgbs",
    "rgbs",
    # "gt_lidar_on_images",
    # "depths_on_images",
    "depths",
    "gt_dino_feats",
    "dino_feats",
    "dynamic_rgbs",
    "dynamic_depths",
    "static_rgbs",
    "static_depths",
    "flows",
    "forward_flows",
    "backward_flows",
    "static_dino_feats",
    "dynamic_dino_feats",
    "dynamic_dino_on_static_rgbs",
    "dynamic_rgb_on_static_dinos",
    "dino_pe",
    "dino_feats_pe_free",
    # "static_pix_rgbs",
    # "dynamic_pix_rgbs",
    # "pix_rgbs",
    "shadow_reduced_static_rgbs",
    "shadow_only_static_rgbs",
    "shadows",
    "gt_sky_masks",
    # "sky_masks",
]
# render_keys = ["rgbs"]


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Train a good radiance field for driving sequences."
    )
    parser.add_argument("--config_file", help="path to config file", type=str)
    parser.add_argument(
        "--eval_only", action="store_true", help="perform evaluation only"
    )
    parser.add_argument(
        "--render_supervision_video",
        action="store_true",
        help="To have a look at the supervision",
    )
    parser.add_argument(
        "--render_supervision_video_only",
        action="store_true",
        help="Quit after rendering supervision video",
    )
    parser.add_argument(
        "--render_video_postfix",
        type=str,
        default=None,
        help="an optional postfix for video",
    )
    parser.add_argument(
        "--output_root",
        default="./work_dirs/",
        help="path to save checkpoints and logs",
        type=str,
    )
    # wandb logging part
    parser.add_argument(
        "--enable_wandb", action="store_true", help="enable wandb logging"
    )
    parser.add_argument(
        "--entity",
        default="YOUR_ENTITY_NAME",
        type=str,
        help="wandb entity name",
        required=False,
    )
    parser.add_argument(
        "--project",
        default="emernerf",
        type=str,
        help="wandb project name, also used to enhance log_dir",
        required=True,
    )
    parser.add_argument(
        "--run_name",
        default="debug",
        type=str,
        help="wandb run name, also used to enhance log_dir",
        required=True,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def setup(args):
    # ------ get config from args -------- #
    default_config = OmegaConf.create(
        OmegaConf.load("configs/hash_default_config.yaml")
    )
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_config, cfg, OmegaConf.from_cli(args.opts))
    log_dir = os.path.join(args.output_root, args.project, args.run_name)
    cfg.log_dir = log_dir
    cfg.nerf.model.num_cams = cfg.data.num_cams
    os.makedirs(log_dir, exist_ok=True)
    for folder in [
        "train_images",
        "test_images",
        "full_videos",
        "test_videos",
        "lidar_sim",
        "lidar_images",
        "lowres_videos",
        "metrics",
        "configs_bk",
        "buffer_maps",
    ]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    fix_random_seeds(cfg.optim.seed)
    # ------ setup logging -------- #
    if args.enable_wandb:
        # sometimes wandb fails to init, so we give it several (many) tries
        while (
            wandb.init(
                project=args.project,
                entity=args.entity,
                sync_tensorboard=True,
                settings=wandb.Settings(start_method="fork"),
            )
            is not wandb.run
        ):
            continue
        wandb.run.name = args.run_name
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.config.update(args)

    global logger
    # this function setup a logger
    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    # -------- write config -------- #
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    saved_cfg_path_bk = os.path.join(
        log_dir, "configs_bk", f"config_{current_time}.yaml"
    )
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    with open(saved_cfg_path_bk, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"Full config saved to {saved_cfg_path}, and {saved_cfg_path_bk}")
    return cfg


def build_estimator(
    cfg: OmegaConf,
    proposal_networks: List[DensityField],
    scheduler_milestones: List[int],
    device: torch.device,
) -> PropNetEstimator:
    # TODO: make it more general
    prop_optimizer = torch.optim.Adam(
        itertools.chain(*[p.parameters() for p in proposal_networks]),
        lr=cfg.optim.lr,
        eps=1e-15,
        weight_decay=cfg.optim.weight_decay,
        betas=(0.9, 0.99),
    )
    prop_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
                prop_optimizer,
                start_factor=0.01,
                total_iters=cfg.optim.num_iters // 10,
            ),
            torch.optim.lr_scheduler.MultiStepLR(
                prop_optimizer,
                milestones=scheduler_milestones,
                gamma=0.33,
            ),
        ]
    )
    estimator = PropNetEstimator(
        prop_optimizer,
        prop_scheduler,
        enable_anti_aliasing_loss=cfg.nerf.estimator.propnet.enable_anti_aliasing_level_loss,
        anti_aliasing_pulse_width=cfg.nerf.estimator.propnet.anti_aliasing_pulse_width,
    ).to(device)
    return estimator


def build_optimizer_from_config(
    cfg: OmegaConf, model: RadianceField
) -> torch.optim.Optimizer:
    # a very simple optimizer for now
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optim.lr,
        eps=1e-15,
        weight_decay=cfg.optim.weight_decay,
        betas=(0.9, 0.99),
    )
    return optimizer


@torch.no_grad()
def do_evaluation(
    step: int = 0,
    cfg: OmegaConf = None,
    model: RadianceField = None,
    proposal_networks: Optional[List[DensityField]] = None,
    prop_estimator: PropNetEstimator = None,
    dataset: WaymoSequenceLoader = None,
    args: argparse.Namespace = None,
):
    logger.info("Evaluating on the full set...")
    model.eval()
    if prop_estimator is not None:
        prop_estimator.eval()
    if proposal_networks is not None:
        for p in proposal_networks:
            p.eval()

    if cfg.data.load_rgb and cfg.render.render_low_res:
        logger.info("Rendering full set but in a low_resolution...")
        render_results = render_pixels(
            cfg=cfg,
            model=model,
            proposal_networks=proposal_networks,
            prop_estimator=prop_estimator,
            dataset=dataset.full_set,
            compute_metrics=True,
            custom_downscale=cfg.render.low_res_downscale,
            return_decomposition=True,
        )
        if args.render_video_postfix is None:
            video_output_pth = os.path.join(cfg.log_dir, "lowres_videos", f"{step}.mp4")
        else:
            video_output_pth = os.path.join(
                cfg.log_dir,
                "lowres_videos",
                f"{step}_{args.render_video_postfix}.mp4",
            )
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            num_timestamps=dataset.num_timestamps,
            keys=render_keys,
            save_seperate_video=cfg.logging.save_seperate_video,
            num_cams=cfg.data.num_cams,
            fps=cfg.render.fps,
            verbose=True,
        )
        if args.enable_wandb:
            for k, v in vis_frame_dict.items():
                wandb.log({f"pixel_rendering/lowres_full/{k}": wandb.Image(v)})

        del render_results, vis_frame_dict
        torch.cuda.empty_cache()

    if cfg.data.load_lidar and cfg.lidar_evaluation.eval_lidar:
        logger.info("Evaluating lidar...")
        if cfg.lidar_evaluation.eval_lidar_id is not None:
            logger.info(
                f"Only evaluating lidar {cfg.lidar_evaluation.eval_lidar_id}..."
            )
        render_results = render_lidars(
            cfg=cfg,
            model=model,
            dataset=dataset.full_set,
            prop_estimator=prop_estimator,
            proposal_networks=proposal_networks,
            compute_metrics=True,
            render_rays_on_image_only=cfg.lidar_evaluation.render_rays_on_image_only,
            render_lidar_id=cfg.lidar_evaluation.eval_lidar_id,
        )
        eval_dict = {}
        for k, v in render_results.items():
            if "avg_chamfer" in k or "avg_depth" in k:
                eval_dict[f"lidar_metrics/full/{k}"] = v
        if args.enable_wandb:
            wandb.log(eval_dict)
        test_metrics_file = os.path.join(
            cfg.log_dir,
            "metrics",
            f"lidar_{current_time}.json",
        )
        with open(test_metrics_file, "w") as f:
            json.dump(eval_dict, f)
        logger.info(f"Lidar evaluation metrics saved to {test_metrics_file}")
        if cfg.lidar_evaluation.save_lidar_simulation:
            video_output_dir = os.path.join(cfg.log_dir, "lidar_sim")
            vis_frame_dict = save_lidar_simulation(
                render_results,
                dataset.full_set,
                video_output_dir,
                fps=cfg.render.fps,
                num_cams=cfg.data.num_cams,
                verbose=True,
            )
            if args.enable_wandb:
                for k, v in vis_frame_dict.items():
                    wandb.log({f"lidar_rendering/full/{k}": wandb.Image(v)})
        logger.info("Lidar evaluation done!")
        del render_results
        torch.cuda.empty_cache()

    if cfg.data.load_rgb:
        logger.info("Evaluating Pixels...")
        if dataset.test_set is not None and cfg.render.render_test:
            logger.info("Evaluating Test Set Pixels...")
            render_results = render_pixels(
                cfg=cfg,
                model=model,
                prop_estimator=prop_estimator,
                dataset=dataset.test_set,
                proposal_networks=proposal_networks,
                compute_metrics=True,
                return_decomposition=True,
            )
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "dino_psnr",
                    "depth_rmse",
                    "masked_psnr",
                    "masked_ssim",
                    "masked_dino_psnr",
                    "masked_depth_rmse",
                ]:
                    eval_dict[f"pixel_metrics/test/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            test_metrics_file = os.path.join(
                cfg.log_dir,
                "metrics",
                f"images_test_{current_time}.json",
            )
            with open(test_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {test_metrics_file}")

            if args.render_video_postfix is None:
                video_output_pth = os.path.join(
                    cfg.log_dir, "test_videos", f"{step}.mp4"
                )
            else:
                video_output_pth = os.path.join(
                    cfg.log_dir,
                    "test_videos",
                    f"{step}_{args.render_video_postfix}.mp4",
                )
            vis_frame_dict = save_videos(
                render_results,
                video_output_pth,
                num_timestamps=dataset.test_set.num_timestamps,
                keys=render_keys,
                num_cams=cfg.data.num_cams,
                save_seperate_video=cfg.logging.save_seperate_video,
                fps=cfg.render.fps,
                verbose=True,
                # save_images=True,
            )
            if args.enable_wandb:
                for k, v in vis_frame_dict.items():
                    wandb.log({"pixel_rendering/test/" + k: wandb.Image(v)})
            del render_results, vis_frame_dict
            torch.cuda.empty_cache()
            # exit()
        if cfg.render.render_full:
            logger.info("Evaluating Full Set...")
            render_results = render_pixels(
                cfg=cfg,
                model=model,
                prop_estimator=prop_estimator,
                dataset=dataset.full_set,
                proposal_networks=proposal_networks,
                compute_metrics=True,
                return_decomposition=True,
            )
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "dino_psnr",
                    "depth_rmse",
                    "masked_psnr",
                    "masked_ssim",
                    "masked_dino_psnr",
                    "masked_depth_rmse",
                ]:
                    eval_dict[f"pixel_metrics/full/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            test_metrics_file = os.path.join(
                cfg.log_dir,
                "metrics",
                f"images_full_{current_time}.json",
            )
            with open(test_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {test_metrics_file}")

            if args.render_video_postfix is None:
                video_output_pth = os.path.join(
                    cfg.log_dir, "full_videos", f"{step}.mp4"
                )
            else:
                video_output_pth = os.path.join(
                    cfg.log_dir,
                    "full_videos",
                    f"{step}_{args.render_video_postfix}.mp4",
                )
            vis_frame_dict = save_videos(
                render_results,
                video_output_pth,
                num_timestamps=dataset.num_timestamps,
                keys=render_keys,
                num_cams=cfg.data.num_cams,
                save_seperate_video=cfg.logging.save_seperate_video,
                fps=cfg.render.fps,
                verbose=True,
            )
            if args.enable_wandb:
                for k, v in vis_frame_dict.items():
                    wandb.log({"pixel_rendering/full/" + k: wandb.Image(v)})
            del render_results, vis_frame_dict
            torch.cuda.empty_cache()
        if cfg.render.render_novel_trajectory:
            logger.info("Rendering render set...")
            render_results = render_pixels(
                cfg=cfg,
                model=model,
                prop_estimator=prop_estimator,
                dataset=dataset.render_set,
                proposal_networks=proposal_networks,
                compute_metrics=False,
                return_decomposition=True,
            )
            if args.render_video_postfix is None:
                video_output_pth = os.path.join(
                    cfg.log_dir, "full_videos", f"{step}_render.mp4"
                )
            else:
                video_output_pth = os.path.join(
                    cfg.log_dir,
                    "full_videos",
                    f"{step}_render_{args.render_video_postfix}.mp4",
                )
            vis_frame_dict = save_videos(
                render_results,
                video_output_pth,
                num_timestamps=dataset.render_set.num_timestamps,
                keys=render_keys,
                save_seperate_video=cfg.logging.save_seperate_video,
                num_cams=cfg.data.num_cams,
                fps=cfg.render.fps,
                verbose=True,
            )
            logger.info(f"Exiting...")
            if args.enable_wandb:
                for k, v in vis_frame_dict.items():
                    wandb.log({"pixel_rendering/novel_trajectory/" + k: wandb.Image(v)})
            del render_results, vis_frame_dict
            torch.cuda.empty_cache()


def main(args):
    cfg = setup(args)
    # ------ build dataset -------- #
    if cfg.data.load_dino and not cfg.data.skip_dino_model:
        img_root = f"{cfg.data.data_root}/{cfg.data.sequence_idx:03d}/images"
        img_list = sorted(
            [os.path.join(img_root, image) for image in os.listdir(img_root)]
        )
        logger.info(
            f"Extracting DINO features of {len(img_list)} images from {img_root}..."
        )
        # visualize the first 5 frames at the beginning
        pca_frames = extract_and_save_dino_features(
            img_list[: cfg.data.num_cams],
            img_shape=cfg.data.dino_extraction_size,
            stride=cfg.data.dino_stride,
            model_type=cfg.data.dino_model_type,
            return_pca=True,
            num_cams=cfg.data.num_cams,
        )
        if args.enable_wandb and len(pca_frames) > 0:
            wandb.log({"data/dino_pca_frame": wandb.Image(pca_frames)})
        extract_and_save_dino_features(
            img_list[cfg.data.num_cams :],
            img_shape=cfg.data.dino_extraction_size,
            stride=cfg.data.dino_stride,
            model_type=cfg.data.dino_model_type,
            num_cams=cfg.data.num_cams,
        )
        # TODO: add an option to delete dino features after training.
        # Dino features of a single sequence (~1000 images) can take up to 40GB of disk space
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WaymoSequenceLoader(
        root=cfg.data.data_root,
        sequence_idx=cfg.data.sequence_idx,
        start_timestamp=cfg.data.start_timestamp,
        num_timestamps=cfg.data.num_timestamps,
        num_cams=cfg.data.num_cams,
        num_rays=cfg.data.ray_batch_size,
        load_size=cfg.data.load_size,
        downscale=cfg.data.downscale,
        device=torch.device(cfg.data.preload_device),
        load_rgb=cfg.data.load_rgb,
        load_lidar=cfg.data.load_lidar,
        load_sky_mask=cfg.data.load_sky_mask,
        load_dino=cfg.data.load_dino,
        load_dynamic_mask=cfg.data.load_dynamic_mask,
        test_holdout=cfg.data.test_holdout,
        target_dino_dim=cfg.data.target_dino_dim,
        dino_model_type=cfg.data.dino_model_type,
        scene_cfg=cfg.nerf.model.scene,
        buffer_ratio=cfg.data.sampler.buffer_ratio,
        buffer_downscale=cfg.data.sampler.buffer_downscale,
        only_use_first_return=cfg.data.lidar.only_use_first_return,
        only_use_top_lidar=cfg.data.lidar.only_use_top_lidar,
        only_keep_lidar_rays_on_images=cfg.data.lidar.only_keep_rays_on_images,
    )
    if args.render_supervision_video or args.render_supervision_video_only:
        videos_dict = dataset.full_set.get_gt_videos()
        save_pth = os.path.join(cfg.log_dir, "supervision.mp4")
        save_videos(
            videos_dict,
            save_pth,
            num_timestamps=dataset.num_timestamps,
            keys=videos_dict.keys(),
            num_cams=cfg.data.num_cams,
            fps=cfg.render.fps,
            verbose=True,
        )
        logger.info(f"Supervision video saved to {save_pth}")
        if args.render_supervision_video_only:
            logger.info("Render supervision video only, exiting...")
            exit()

    # ------ build model -------- #
    cfg.nerf.model.num_train_timestamps = len(dataset.unique_train_timestamps)
    if dataset.test_set is not None:
        if cfg.nerf.model.head.enable_img_embedding:
            cfg.nerf.model.head.enable_cam_embedding = True
            cfg.nerf.model.head.enable_img_embedding = False
            logger.info(
                "Overriding enable_img_embedding to False because we have test set."
            )
    model = build_radiance_field_from_cfg(cfg.nerf.model)
    model.to(device)
    model.register_training_timestamps(dataset.unique_train_timestamps)

    # ------ build optimizer and grad scaler -------- #
    optimizer = build_optimizer_from_config(cfg, model)
    pixel_grad_scaler = torch.cuda.amp.GradScaler(2**10)
    lidar_grad_scaler = torch.cuda.amp.GradScaler(2**10)

    # ------ build scheduler -------- #
    scheduler_milestones = [
        # cfg.optim.num_iters // 4,
        cfg.optim.num_iters // 2,
        cfg.optim.num_iters * 3 // 4,
        cfg.optim.num_iters * 9 // 10,
    ]
    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            # warmup
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=cfg.optim.num_iters // 10
            ),
            # Linear decay
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=scheduler_milestones,
                gamma=0.33,
            ),
        ]
    )
    if dataset.aabb is not None and cfg.resume_from is None:
        model.set_aabb(dataset.aabb)

    if cfg.data.load_dino and cfg.nerf.model.head.enable_dino_head:
        # we cache the dino PCA reduction matrix and min/max values for visualization
        model.register_dino_feats_reduction_mat(
            dataset.dino_dimension_reduction_mat,
            dataset.color_norm_min,
            dataset.color_norm_max,
        )
    # ------ build proposal networks -------- #
    proposal_networks = [
        build_density_field(
            aabb=cfg.nerf.model.scene.aabb,
            type=cfg.nerf.estimator.propnet.xyz_encoder.type,
            n_input_dims=cfg.nerf.estimator.propnet.xyz_encoder.n_input_dims,
            n_levels=cfg.nerf.estimator.propnet.xyz_encoder.n_levels_per_prop[i],
            max_resolution=cfg.nerf.estimator.propnet.xyz_encoder.max_resolutions_per_prop[
                i
            ],
            log2_hashmap_size=cfg.nerf.estimator.propnet.xyz_encoder.lgo2_hashmap_sizes_per_prop[
                i
            ],
            n_features_per_level=cfg.nerf.estimator.propnet.xyz_encoder.n_features_per_level,
            unbounded=cfg.nerf.model.scene.unbounded,
            nerf_model_cfg=cfg.nerf.model,
        ).to(device)
        for i in range(len(cfg.nerf.estimator.propnet.xyz_encoder.n_levels_per_prop))
    ]
    if dataset.aabb is not None and cfg.resume_from is None:
        for p in proposal_networks:
            p.set_aabb(dataset.aabb)
            p.register_training_timestamps(dataset.unique_train_timestamps)
    prop_estimator = build_estimator(
        cfg, proposal_networks, scheduler_milestones, device
    )
    logger.info(f"PropNetEstimator: {proposal_networks}")
    logger.info(f"Model: {model}")

    if cfg.resume_from is not None:
        logger.info(f"Loading checkpoint from {cfg.resume_from}")
        checkpoint = torch.load(cfg.resume_from)
        msg = model.load_state_dict(checkpoint["model"])
        logger.info(f"radiance_field: {msg}")
        if proposal_networks is not None:
            for i, p in enumerate(proposal_networks):
                msg = p.load_state_dict(checkpoint["proposal_networks"][i])
                logger.info(f"proposal_networks[{i}]: {msg}")
        if prop_estimator is not None:
            msg = prop_estimator.load_state_dict(checkpoint["prop_estimator"])
            logger.info(f"prop_estimator: {msg}")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = min(checkpoint["step"], cfg.optim.num_iters)
        logger.info(f"Will resuming from step {start_step}")
    else:
        start_step = 0
        logger.info(
            f"Will start training for {cfg.optim.num_iters} iterations from scratch"
        )

    if args.eval_only:
        do_evaluation(
            step=start_step,
            cfg=cfg,
            model=model,
            proposal_networks=proposal_networks,
            prop_estimator=prop_estimator,
            dataset=dataset,
            args=args,
        )
        exit()

    # ------ build losses -------- #
    # rgb loss
    if cfg.data.load_rgb:
        rgb_loss_fn = loss.RealValueLoss(
            loss_type=cfg.supervision.rgb.loss_type,
            coef=cfg.supervision.rgb.loss_coef,
            name="rgb",
            check_nan=cfg.optim.check_nan,
        )

    # lidar related losses
    if cfg.data.load_lidar:
        depth_loss_fn = loss.DepthLoss(
            loss_type=cfg.supervision.depth.loss_type,
            coef=cfg.supervision.depth.loss_coef,
            depth_percentile=cfg.supervision.depth.depth_percentile,
            check_nan=cfg.optim.check_nan,
        )
        if cfg.supervision.depth.line_of_sight.enable:
            line_of_sight_loss_fn = loss.LineOfSightLoss(
                loss_type=cfg.supervision.depth.line_of_sight.loss_type,
                name="line_of_sight",
                depth_percentile=cfg.supervision.depth.depth_percentile,
                coef=cfg.supervision.depth.line_of_sight.loss_coef,
                check_nan=cfg.optim.check_nan,
            )
        else:
            line_of_sight_loss_fn = None
    else:
        depth_loss_fn = None
        line_of_sight_loss_fn = None

    if cfg.nerf.model.head.enable_sky_head and cfg.data.load_sky_mask:
        sky_loss_fn = loss.SkyLoss(
            loss_type=cfg.supervision.sky.loss_type,
            coef=cfg.supervision.sky.loss_coef,
            check_nan=cfg.optim.check_nan,
        )
    else:
        sky_loss_fn = None

    if cfg.data.load_dino and cfg.nerf.model.head.enable_dino_head:
        dino_loss_fn = loss.RealValueLoss(
            loss_type=cfg.supervision.dino.loss_type,
            coef=cfg.supervision.dino.loss_coef,
            name="dino",
            check_nan=cfg.optim.check_nan,
        )
    else:
        dino_loss_fn = None

    ## ------ dynamic related losses -------- #
    if cfg.nerf.model.head.enable_dynamic_branch:
        dynamic_reg_loss_fn = loss.DynamicRegularizationLoss(
            loss_type=cfg.supervision.dynamic.loss_type,
            coef=cfg.supervision.dynamic.loss_coef,
            entropy_skewness=cfg.supervision.dynamic.entropy_loss_skewness,
            check_nan=cfg.optim.check_nan,
        )
    else:
        dynamic_reg_loss_fn = None

    if cfg.nerf.model.head.enable_shadow_head:
        shadow_loss_fn = loss.DynamicRegularizationLoss(
            name="shadow",
            loss_type=cfg.supervision.shadow.loss_type,
            coef=cfg.supervision.shadow.loss_coef,
            check_nan=cfg.optim.check_nan,
        )
    else:
        shadow_loss_fn = None

    metrics_file = os.path.join(cfg.log_dir, "metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    if proposal_networks is not None:
        proposal_requires_grad_fn = get_proposal_requires_grad_fn()
    else:
        proposal_requires_grad_fn = None

    epsilon_final = cfg.supervision.depth.line_of_sight.end_epsilon
    epsilon_start = cfg.supervision.depth.line_of_sight.start_epsilon
    all_iters = np.arange(start_step, cfg.optim.num_iters + 1)
    line_of_sight_loss_decay_weight = 1.0
    for step in metric_logger.log_every(all_iters, cfg.logging.print_freq):
        model.train()
        epsilon, stats, pixel_data_dict, lidar_data_dict = None, None, None, None
        pixel_loss_dict, lidar_loss_dict = {}, {}
        if proposal_networks is not None:
            for p in proposal_networks:
                p.train()
        if prop_estimator is not None:
            prop_estimator.train()

        if cfg.nerf.sampling.enable_coarse_to_fine_ray_sampling:
            start_sampling_interval = 3
            end_sampling_interval = 1
            end_iter = cfg.optim.num_iters // 3
            current_sampling_interval = (
                start_sampling_interval
                - (start_sampling_interval - end_sampling_interval) * step / end_iter
            )
            current_sampling_interval = max(current_sampling_interval, 1)
        else:
            current_sampling_interval = 1

        if (
            step > cfg.supervision.depth.line_of_sight.start_iter
            and (step - cfg.supervision.depth.line_of_sight.start_iter)
            % cfg.supervision.depth.line_of_sight.decay_steps
            == 0
        ):
            line_of_sight_loss_decay_weight *= (
                cfg.supervision.depth.line_of_sight.decay_rate
            )
            logger.info(
                f"line_of_sight_loss_decay_weight: {line_of_sight_loss_decay_weight}"
            )

        # ------ pixel ray supervision -------- #
        if cfg.data.load_rgb:
            i = torch.randint(0, len(dataset.train_set), (1,)).item()
            pixel_data_dict = dataset.train_set.fetch_train_pixel_data(
                i, sample_interval=current_sampling_interval
            )
            if proposal_requires_grad_fn is not None:
                proposal_requires_grad = proposal_requires_grad_fn(int(step))

            # ------ pixel-wise supervision -------- #
            render_results = render_rays(
                radiance_field=model,
                prop_estimator=prop_estimator,
                proposal_networks=proposal_networks,
                data_dict=pixel_data_dict,
                cfg=cfg,
                proposal_requires_grad=proposal_requires_grad,
            )
            if prop_estimator is not None:
                prop_estimator.update_every_n_steps(
                    render_results["extras"]["trans"],
                    proposal_requires_grad,
                    loss_scaler=1024,
                )
            # compute losses
            # rgb loss
            pixel_loss_dict.update(
                rgb_loss_fn(render_results["rgb"], pixel_data_dict["pixels"])
            )
            if sky_loss_fn is not None:  # if sky loss is enabled
                if cfg.supervision.sky.loss_type == "weights_based":
                    # penalize the points' weights if they point to the sky
                    pixel_loss_dict.update(
                        sky_loss_fn(
                            render_results["extras"]["weights"],
                            pixel_data_dict["sky_mask"],
                        )
                    )
                elif cfg.supervision.sky.loss_type == "opacity_based":
                    # penalize accumulated opacity if the ray points to the sky
                    pixel_loss_dict.update(
                        sky_loss_fn(
                            render_results["opacity"], pixel_data_dict["sky_mask"]
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"sky_loss_type {cfg.supervision.sky.loss_type} not implemented"
                    )
            if dino_loss_fn is not None:
                pixel_loss_dict.update(
                    dino_loss_fn(
                        render_results["dino_feat"],
                        pixel_data_dict["dino_feat"],
                    )
                )
            if dynamic_reg_loss_fn is not None:
                pixel_loss_dict.update(
                    dynamic_reg_loss_fn(
                        dynamic_density=render_results["extras"]["dynamic_density"],
                        static_density=render_results["extras"]["static_density"],
                    )
                )
            if shadow_loss_fn is not None:
                pixel_loss_dict.update(
                    shadow_loss_fn(
                        render_results["shadow_ratio"],
                    )
                )
            if "forward_flow" in render_results["extras"]:
                cycle_loss = (
                    0.5
                    * (
                        (
                            render_results["extras"]["forward_flow"].detach()
                            + render_results["extras"]["forward_pred_backward_flow"]
                        )
                        ** 2
                        + (
                            render_results["extras"]["backward_flow"].detach()
                            + render_results["extras"]["backward_pred_forward_flow"]
                        )
                        ** 2
                    ).mean()
                )
                pixel_loss_dict.update({"cycle_loss": cycle_loss * 0.01})
                stats = {
                    "max_forward_flow_norm": (
                        render_results["extras"]["forward_flow"]
                        .detach()
                        .norm(dim=-1)
                        .max()
                    ),
                    "max_backward_flow_norm": (
                        render_results["extras"]["backward_flow"]
                        .detach()
                        .norm(dim=-1)
                        .max()
                    ),
                    "max_forward_pred_backward_flow_norm": (
                        render_results["extras"]["forward_pred_backward_flow"]
                        .norm(dim=-1)
                        .max()
                    ),
                    "max_backward_pred_forward_flow_norm": (
                        render_results["extras"]["backward_pred_forward_flow"]
                        .norm(dim=-1)
                        .max()
                    ),
                }
            total_pixel_loss = sum(loss for loss in pixel_loss_dict.values())
            optimizer.zero_grad()
            pixel_grad_scaler.scale(total_pixel_loss).backward()
            optimizer.step()
            scheduler.step()

        # ------ lidar ray supervision -------- #
        if cfg.data.load_lidar:
            if proposal_requires_grad_fn is not None:
                proposal_requires_grad = proposal_requires_grad_fn(int(step))
            i = torch.randint(0, dataset.train_set.num_timestamps, (1,)).item()
            lidar_data_dict = dataset.train_set.fetch_train_lidar_data(i)
            lidar_render_results = render_rays(
                radiance_field=model,
                prop_estimator=prop_estimator,
                proposal_networks=proposal_networks,
                data_dict=lidar_data_dict,
                cfg=cfg,
                proposal_requires_grad=proposal_requires_grad,
                prefix="lidar_",
            )
            if prop_estimator is not None:
                prop_estimator.update_every_n_steps(
                    lidar_render_results["extras"]["trans"],
                    proposal_requires_grad,
                    loss_scaler=1024,
                )
            lidar_loss_dict.update(
                depth_loss_fn(
                    lidar_render_results["depth"],
                    lidar_data_dict["lidar_ranges"],
                    name="lidar_range_loss",
                )
            )
            if (
                line_of_sight_loss_fn is not None
                and step > cfg.supervision.depth.line_of_sight.start_iter
            ):
                m = (epsilon_final - epsilon_start) / (
                    cfg.optim.num_iters - cfg.supervision.depth.line_of_sight.start_iter
                )
                b = epsilon_start - m * cfg.supervision.depth.line_of_sight.start_iter

                def epsilon_decay(step):
                    if step < cfg.supervision.depth.line_of_sight.start_iter:
                        return epsilon_start
                    elif step > cfg.optim.num_iters:
                        return epsilon_final
                    else:
                        return m * step + b

                epsilon = epsilon_decay(step)
                line_of_sight_loss_dict = line_of_sight_loss_fn(
                    pred_depth=lidar_render_results["depth"],
                    gt_depth=lidar_data_dict["lidar_ranges"],
                    weights=lidar_render_results["extras"]["weights"],
                    t_vals=lidar_render_results["extras"]["t_vals"],
                    epsilon=epsilon,
                    name="lidar_line_of_sight",
                    coef_decay=line_of_sight_loss_decay_weight,
                )
                lidar_loss_dict.update(
                    {
                        "lidar_line_of_sight": line_of_sight_loss_dict[
                            "lidar_line_of_sight"
                        ].mean()
                    }
                )

            if dynamic_reg_loss_fn is not None:
                lidar_loss_dict.update(
                    dynamic_reg_loss_fn(
                        dynamic_density=lidar_render_results["extras"][
                            "dynamic_density"
                        ],
                        static_density=lidar_render_results["extras"]["static_density"],
                        name="lidar_dynamic",
                    )
                )

            total_lidar_loss = sum(loss for loss in lidar_loss_dict.values())
            optimizer.zero_grad()
            lidar_grad_scaler.scale(total_lidar_loss).backward()
            optimizer.step()
            scheduler.step()
            total_lidar_loss = total_lidar_loss.item()
        else:
            total_lidar_loss = -1

        if pixel_data_dict is not None:
            psnr = compute_psnr(render_results["rgb"], pixel_data_dict["pixels"])
            metric_logger.update(psnr=psnr)
            metric_logger.update(
                total_pixel_loss=total_pixel_loss.item(),
            )

        if lidar_data_dict is not None:
            metric_logger.update(
                total_lidar_loss=total_lidar_loss,
            )
            range_rmse = compute_valid_depth_rmse(
                lidar_render_results["depth"], lidar_data_dict["lidar_ranges"]
            )
            metric_logger.update(range_rmse=range_rmse)

        metric_logger.update(**{k: v.item() for k, v in pixel_loss_dict.items()})
        metric_logger.update(**{k: v.item() for k, v in lidar_loss_dict.items()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if stats is not None:
            metric_logger.update(**{k: v.item() for k, v in stats.items()})
        if epsilon is not None:
            metric_logger.update(epsilon=epsilon)
        # log to wandb
        if args.enable_wandb:
            wandb.log(
                {f"train_stats/{k}": v.avg for k, v in metric_logger.meters.items()}
            )

        if (
            step > 0
            and (step % cfg.logging.saveckpt_freq == 0 and (cfg.resume_from is None))
            or (step == cfg.optim.num_iters and (cfg.resume_from is None))
        ):
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
            }
            if prop_estimator is not None:
                checkpoint["prop_estimator"] = prop_estimator.state_dict()
            if proposal_networks is not None:
                checkpoint["proposal_networks"] = [
                    p.state_dict() for p in proposal_networks
                ]
            save_path = os.path.join(cfg.log_dir, f"checkpoint_{step:05d}.pth")
            torch.save(checkpoint, save_path)
            logger.info(f"Saved a checkpoint to {save_path}")

        if step > 0 and cfg.data.load_rgb and step % cfg.optim.cache_rgb_freq == 0:
            model.eval()
            if prop_estimator is not None:
                prop_estimator.eval()
            if proposal_networks is not None:
                for p in proposal_networks:
                    p.eval()
            if cfg.data.sampler.buffer_ratio > 0:
                with torch.no_grad():
                    logger.info("cache rgb error map...")
                    render_results = render_pixels(
                        cfg=cfg,
                        model=model,
                        proposal_networks=proposal_networks,
                        prop_estimator=prop_estimator,
                        dataset=dataset.train_set,
                        compute_metrics=False,
                        custom_downscale=cfg.data.sampler.buffer_downscale,
                        return_decomposition=True,
                    )
                    dataset.train_set.update_pixel_error_maps(render_results)
                    dataset.sync_pixel_error_maps()
                    maps_video = dataset.train_set.get_pixel_sample_weights_video()
                    merged_list = []
                    for i in range(len(maps_video) // 3):
                        frames = maps_video[i * 3 : (i + 1) * 3]
                        frames = [
                            np.stack([frame, frame, frame], axis=-1) for frame in frames
                        ]
                        frames = np.concatenate(frames, axis=1)
                        merged_list.append(frames)
                    merged_video = np.stack(merged_list, axis=0)
                    merged_video -= merged_video.min()
                    merged_video /= merged_video.max()
                    merged_video = np.clip(merged_video * 255, 0, 255).astype(np.uint8)

                    imageio.mimsave(
                        os.path.join(
                            cfg.log_dir, "buffer_maps", f"buffer_maps_{step}.mp4"
                        ),
                        merged_video,
                        fps=cfg.render.fps,
                    )
                logger.info("Done caching rgb error maps")

        if step > 0 and step % cfg.logging.vis_freq == 0:
            dataset.sync_error_maps()
            model.eval()
            if prop_estimator is not None:
                prop_estimator.eval()
            if proposal_networks is not None:
                for p in proposal_networks:
                    p.eval()
            if cfg.data.load_rgb:
                logger.info("Visualizing one training pixel view...")
                vis_timestamp = np.linspace(
                    0,
                    len(dataset.unique_train_timestamps),
                    cfg.optim.num_iters // cfg.logging.vis_freq + 1,
                    endpoint=False,
                    dtype=int,
                )[step // cfg.logging.vis_freq]
                with torch.no_grad():
                    render_results = render_pixels(
                        cfg=cfg,
                        model=model,
                        proposal_networks=proposal_networks,
                        prop_estimator=prop_estimator,
                        dataset=dataset.full_set,
                        compute_metrics=True,
                        vis_indices=[
                            vis_timestamp * cfg.data.num_cams + i
                            for i in range(cfg.data.num_cams)
                        ],
                        return_decomposition=True,
                    )
                if args.enable_wandb:
                    wandb.log(
                        {
                            "pixel_metrics/train_vis/psnr": render_results["psnr"],
                            "pixel_metrics/train_vis/ssim": render_results["ssim"],
                            "pixel_metrics/train_vis/dino_psnr": render_results[
                                "dino_psnr"
                            ],
                            "pixel_metrics/train_vis/depth_rmse": render_results[
                                "depth_rmse"
                            ],
                            "pixel_metrics/train_vis/masked_psnr": render_results[
                                "masked_psnr"
                            ],
                            "pixel_metrics/train_vis/masked_ssim": render_results[
                                "masked_ssim"
                            ],
                            "pixel_metrics/train_vis/masked_dino_psnr": render_results[
                                "masked_dino_psnr"
                            ],
                            "pixel_metrics/train_vis/masked_depth_rmse": render_results[
                                "masked_depth_rmse"
                            ],
                        }
                    )
                vis_frame_dict = save_videos(
                    render_results,
                    save_pth=os.path.join(
                        cfg.log_dir, "train_images", f"step_{step}.png"
                    ),  # don't save the video
                    num_timestamps=1,
                    keys=render_keys,
                    save_seperate_video=cfg.logging.save_seperate_video,
                    num_cams=cfg.data.num_cams,
                    fps=cfg.render.fps,
                    verbose=False,
                )
                if args.enable_wandb:
                    for k, v in vis_frame_dict.items():
                        wandb.log({"pixel_rendering/train_vis/" + k: wandb.Image(v)})
                if cfg.data.sampler.buffer_ratio > 0:
                    vis_frame = dataset.train_set.visualize_pixel_sample_weights(
                        [
                            vis_timestamp * cfg.data.num_cams + i
                            for i in range(cfg.data.num_cams)
                        ]
                    )
                    imageio.imwrite(
                        os.path.join(
                            cfg.log_dir, "buffer_maps", f"buffer_map_{step}.png"
                        ),
                        vis_frame,
                    )
                    if args.enable_wandb:
                        wandb.log(
                            {
                                "pixel_rendering/train_vis/buffer_map": wandb.Image(
                                    vis_frame
                                )
                            }
                        )
                del render_results
                torch.cuda.empty_cache()
                if dataset.test_set is not None:
                    logger.info("Visualizing one testing pixel view...")
                    vis_timestamp = np.linspace(
                        0,
                        len(dataset.unique_test_timestamps),
                        cfg.optim.num_iters // cfg.logging.vis_freq + 1,
                        endpoint=False,
                        dtype=int,
                    )[step // cfg.logging.vis_freq]
                    with torch.no_grad():
                        render_results = render_pixels(
                            cfg=cfg,
                            model=model,
                            proposal_networks=proposal_networks,
                            prop_estimator=prop_estimator,
                            dataset=dataset.test_set,
                            compute_metrics=True,
                            vis_indices=[
                                vis_timestamp * cfg.data.num_cams + i
                                for i in range(cfg.data.num_cams)
                            ],
                            return_decomposition=True,
                        )
                    if args.enable_wandb:
                        wandb.log(
                            {
                                "pixel_metrics/test_vis/psnr": render_results["psnr"],
                                "pixel_metrics/test_vis/ssim": render_results["ssim"],
                                "pixel_metrics/test_vis/dino_psnr": render_results[
                                    "dino_psnr"
                                ],
                                "pixel_metrics/test_vis/depth_rmse": render_results[
                                    "depth_rmse"
                                ],
                                "pixel_metrics/test_vis/masked_psnr": render_results[
                                    "masked_psnr"
                                ],
                                "pixel_metrics/test_vis/masked_ssim": render_results[
                                    "masked_ssim"
                                ],
                                "pixel_metrics/test_vis/masked_dino_psnr": render_results[
                                    "masked_dino_psnr"
                                ],
                                "pixel_metrics/test_vis/masked_depth_rmse": render_results[
                                    "masked_depth_rmse"
                                ],
                            }
                        )
                    vis_frame_dict = save_videos(
                        render_results,
                        save_pth=os.path.join(
                            cfg.log_dir, "test_images", f"step_{step}.png"
                        ),  # don't save the video
                        num_timestamps=1,
                        keys=render_keys,
                        save_seperate_video=cfg.logging.save_seperate_video,
                        num_cams=cfg.data.num_cams,
                        fps=cfg.render.fps,
                        verbose=False,
                    )
                    if args.enable_wandb:
                        for k, v in vis_frame_dict.items():
                            wandb.log({"pixel_rendering/test_vis/" + k: wandb.Image(v)})
                    del render_results, vis_frame_dict
                    torch.cuda.empty_cache()

            if cfg.data.load_lidar:
                logger.info("Visualizing one lidar view...")
                vis_timestamp = np.linspace(
                    0,
                    dataset.num_timestamps,
                    cfg.optim.num_iters // cfg.logging.vis_freq + 1,
                    endpoint=False,
                    dtype=int,
                )[step // cfg.logging.vis_freq]
                with torch.no_grad():
                    render_results = render_lidars(
                        cfg=cfg,
                        model=model,
                        dataset=dataset.full_set,
                        prop_estimator=prop_estimator,
                        proposal_networks=proposal_networks,
                        compute_metrics=True,
                        vis_indices=[vis_timestamp],
                        render_rays_on_image_only=cfg.lidar_evaluation.render_rays_on_image_only,
                        render_lidar_id=cfg.lidar_evaluation.eval_lidar_id,
                    )
                eval_dict = {}
                for k, v in render_results.items():
                    if "avg_chamfer" in k or "avg_depth" in k:
                        eval_dict["lidar_metrics/vis/" + k] = v
                if args.enable_wandb:
                    wandb.log(eval_dict)
                save_path = os.path.join(cfg.log_dir, f"lidar_images/step_{step}/")
                os.makedirs(save_path, exist_ok=True)
                vis_frame_dict = save_lidar_simulation(
                    render_results,
                    dataset=dataset.train_set,
                    save_pth=save_path,  # don't save the video
                    fps=cfg.render.fps,
                    num_cams=cfg.data.num_cams,
                    verbose=False,
                )
                if args.enable_wandb:
                    for k, v in vis_frame_dict.items():
                        wandb.log({f"lidar_rendering/vis/{k}": wandb.Image(v)})
                del render_results
                torch.cuda.empty_cache()
    logger.info("Training done!")

    do_evaluation(
        step=step,
        cfg=cfg,
        model=model,
        proposal_networks=proposal_networks,
        prop_estimator=prop_estimator,
        dataset=dataset,
        args=args,
    )
    if args.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)

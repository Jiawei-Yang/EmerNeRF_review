
export CUDA_VISIBLE_DEVICES=0
set -x
export DISPLAY=:99.0
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 1
set +x


sequence_idx=700
start_timestamp=0
num_timestamps=-1
num_iters=5000
project=1002_baseline_new
python train.py \
    --config_file configs/hash_dynamic_default_config.yaml \
    --output_root ./work_dirs/ \
    --project $project \
    --run_name ${sequence_idx}_nvs \
    data.test_holdout=10 \
    data.downscale=1 \
    data.load_size=[160,240] \
    data.preload_device="cuda" \
    data.sequence_idx=$sequence_idx \
    data.start_timestamp=$start_timestamp \
    data.num_timestamps=$num_timestamps \
    data.sampler.buffer_downscale=8 \
    render.render_novel_trajectory=False \
    lidar_evaluation.save_lidar_simulation=False \
    lidar_evaluation.eval_lidar=False \
    nerf.model.head.enable_cam_embedding=False \
    nerf.model.head.enable_img_embedding=True \
    nerf.model.head.appearance_embedding_dim=16 \
    nerf.model.head.enable_flow_branch=True \
    nerf.model.head.enable_temporal_interpolation=True \
    logging.vis_freq=1000 \
    logging.saveckpt_freq=5000 \
    optim.num_iters=$num_iters \
    resume_from=./work_dirs/1002_baseline_new/700_nvs/checkpoint_05000.pth \

# sequence_idx=700
# start_timestamp=0
# num_timestamps=-1
# num_iters=10000
# project=0922_better_flow
# python train_good_nerf.py \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_wo_flow_wo_time_interp_low_res \
#     data.test_holdout=10 \
#     data.downscale=1 \
#     data.load_size=[160,240] \
#     data.preload_device="cuda" \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     render.render_novel_trajectory=True \
#     lidar_evaluation.save_lidar_simulation=False \
#     nerf.model.head.enable_cam_embedding=True \
#     nerf.model.head.enable_img_embedding=False \
#     nerf.model.head.appearance_embedding_dim=16 \
#     nerf.model.head.enable_flow_branch=False \
#     nerf.model.head.enable_temporal_interpolation=False \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     resume_from=./work_dirs/0922_better_flow/700_wo_flow_wo_time_interp_low_res/checkpoint_10000.pth



# sequence_idx=700
# start_timestamp=0
# num_timestamps=-1
# num_iters=25000
# project=0922_better_flow
# python train_good_nerf.py \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_debug \
#     data.test_holdout=0 \
#     data.downscale=2 \
#     data.auto_novel_view_based_on_velocity=True \
#     data.preload_device="cuda" \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     lidar_evaluation.save_lidar_simulation=False \
#     nerf.model.head.enable_cam_embedding=True \
#     nerf.model.head.enable_img_embedding=False \
#     nerf.model.head.appearance_embedding_dim=32 \
#     nerf.model.head.enable_flow_branch=True \
#     data.load_dino=True \
#     data.dino_model_type="dinov2_vitb14" \
#     data.dino_stride=7 \
#     data.dino_extraction_size=[644,966] \
#     data.target_dino_dim=64 \
#     nerf.model.head.dino_embedding_dim=64 \
#     nerf.model.head.enable_learnable_pe=True \
#     nerf.model.head.enable_dino_head=True \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters
    
# sequence_idx=700
# start_timestamp=0
# num_timestamps=-1
# num_iters=5000
# project=0922_better_flow
# python train_good_nerf.py \
#     --enable_wandb \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_with_flow_condition_on_inputs_detach_wt_dino \
#     data.test_holdout=0 \
#     data.downscale=2 \
#     data.preload_device="cuda" \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     render.fps=10 \
#     lidar_evaluation.save_lidar_simulation=False \
#     nerf.model.head.enable_cam_embedding=True \
#     nerf.model.head.enable_img_embedding=False \
#     nerf.model.head.appearance_embedding_dim=32 \
#     nerf.model.head.enable_flow_branch=True \
#     data.load_dino=True \
#     data.dino_model_type="dinov2_vitb14" \
#     data.dino_stride=7 \
#     data.dino_extraction_size=[644,966] \
#     data.target_dino_dim=64 \
#     nerf.model.head.dino_embedding_dim=64 \
#     nerf.model.head.enable_learnable_pe=True \
#     nerf.model.head.enable_dino_head=True \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters


# sequence_idx=700
# start_timestamp=0
# num_timestamps=-1
# num_iters=5000
# project=0922_better_flow
# python train_good_nerf.py \
#     --enable_wandb \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_baseline \
#     data.test_holdout=0 \
#     data.downscale=2 \
#     data.preload_device="cuda" \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     render.fps=10 \
#     lidar_evaluation.save_lidar_simulation=False \
#     nerf.model.head.enable_cam_embedding=True \
#     nerf.model.head.enable_img_embedding=False \
#     nerf.model.head.appearance_embedding_dim=32 \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters


# sequence_idx=700
# start_timestamp=0
# num_timestamps=-1
# num_iters=10000
# project=0908
# python train_good_nerf.py \
#     --config_file work_dirs/0906/700_try_to_improve_reproduced_better_error_map2/config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_render_flow \
#     data.test_holdout=0 \
#     data.load_rgb=True \
#     data.load_sky_mask=True \
#     data.load_size=[640,960] \
#     lidar_evaluation.save_lidar_simulation=False \
#     lidar_evaluation.eval_lidar_id=null \
# sequence_idx=226
# start_timestamp=0
# num_timestamps=-1
# num_iters=15000
# project=0922_pos_pattern
# python train_good_nerf.py \
#     --enable_wandb \
#     --config_file configs/hash_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_dinov2 \
#     data.test_holdout=0 \
#     data.preload_device="cuda" \
#     data.skip_dino_model=False \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     nerf.model.head.enable_cam_embedding=True \
#     nerf.model.head.enable_img_embedding=False \
#     nerf.model.head.appearance_embedding_dim=32 \
#     data.load_dino=True \
#     data.dino_model_type="dinov2_vitb14" \
#     data.dino_stride=7 \
#     data.dino_extraction_size=[644,966] \
#     data.target_dino_dim=64 \
#     nerf.model.head.dino_embedding_dim=64 \
#     nerf.model.head.enable_learnable_pe=True \
#     nerf.model.head.enable_dino_head=True \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters
# sequence_idx=382
# start_timestamp=0
# num_timestamps=-1
# num_iters=25000
# project=0918_dynamic
# python train_good_nerf.py \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/0918/ \
#     --project $project \
#     --run_name ${sequence_idx} \
#     data.test_holdout=0 \
#     data.preload_device="cuda" \
#     data.auto_novel_view_based_on_velocity=True \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     nerf.model.head.enable_cam_embedding=True \
#     nerf.model.head.enable_img_embedding=False \
#     nerf.model.head.appearance_embedding_dim=16 \
#     lidar_evaluation.eval_lidar=False \
#     logging.vis_freq=1 \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \

# python train_good_nerf.py \
#     --config_file configs/hash_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --render_supervision_video \
#     --run_name ${sequence_idx}_vis \
#     data.downscale=1 \
#     data.test_holdout=0 \
#     data.load_size=[160,240] \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     data.load_rgb=True \
#     data.load_lidar=False \
#     data.load_sky_mask=True \
#     data.load_dino=False \
#     data.dino_model_type="dinov2_vitb14" \
#     data.dino_stride=7 \
#     data.dino_extraction_size=[644,966] \
#     data.target_dino_dim=64 \
#     nerf.model.head.dino_embedding_dim=64 \
#     nerf.model.head.enable_learnable_pe=False \
#     nerf.model.head.enable_dino_head=False \
#     render.low_res_downscale=2 \
#     optim.cache_rgb_freq=1000 \
#     optim.num_iters=$num_iters \
#     optim.warmup_steps=500 \
#     logging.vis_freq=1000 \
#     resume_from=./work_dirs/0918_sup/700_vis/checkpoint_02000.pth \
    
# sequence_idx=700
# start_timestamp=0
# num_timestamps=100
# num_iters=5000
# project=0914_better_flow_using_dino2
# python train_good_nerf.py \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_wt_dino_wt_flow \
#     --enable_wandb \
#     data.downscale=1 \
#     data.test_holdout=0 \
#     data.load_size=[320,480] \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     data.load_rgb=True \
#     data.load_lidar=True \
#     data.load_sky_mask=True \
#     data.load_dino=True \
#     data.dino_model_type="dinov2_vitb14" \
#     data.dino_stride=7 \
#     data.dino_extraction_size=[644,966] \
#     data.target_dino_dim=64 \
#     nerf.model.head.dino_embedding_dim=64 \
#     nerf.model.head.enable_learnable_pe=True \
#     nerf.model.head.enable_dino_head=True \
#     nerf.model.head.enable_flow_branch=True \
#     supervision.depth.loss_coef=1.0 \
#     supervision.depth.line_of_sight.enable=True \
#     supervision.depth.line_of_sight.loss_coef=0.1 \
#     supervision.depth.line_of_sight.start_iter=500 \
#     supervision.depth.line_of_sight.start_epsilon=5.0 \
#     supervision.depth.line_of_sight.end_epsilon=0.5 \
#     optim.num_iters=$num_iters \
#     optim.warmup_steps=1000 \
#     logging.vis_freq=1000 \

# sequence_idx=700
# start_timestamp=0
# num_timestamps=-1
# num_iters=15000
# project=0912_try_to_reproduce
# python train_good_nerf.py \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_keep_rays_on_imgs_no_priority \
#     data.downscale=2 \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     data.test_holdout=0 \
#     data.load_rgb=True \
#     data.load_lidar=True \
#     data.load_sky_mask=True \
#     data.sampler.buffer_downscale=32 \
#     data.lidar.only_use_top_lidar=False \
#     data.lidar.only_use_first_return=True \
#     data.lidar.only_keep_rays_on_images=True \
#     nerf.model.head.enable_flow_branch=False \
#     supervision.sky.loss_coef=0.01 \
#     supervision.depth.loss_coef=1.0 \
#     supervision.depth.line_of_sight.enable=True \
#     supervision.depth.line_of_sight.loss_coef=0.1 \
#     supervision.depth.line_of_sight.start_iter=500 \
#     supervision.depth.line_of_sight.start_epsilon=2.0 \
#     supervision.depth.line_of_sight.end_epsilon=0.5 \
#     optim.num_iters=$num_iters \
#     optim.warmup_steps=3000 \
#     optim.cache_rgb_freq=2000 \
#     optim.check_nan=False \
#     lidar_evaluation.save_lidar_simulation=True \
#     lidar_evaluation.eval_lidar_id=null \
#     lidar_evaluation.render_rays_on_image_only=True \
#     logging.vis_freq=2000 \


# python train_good_nerf.py \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_no_direct_lidar \
#     data.test_holdout=0 \
#     data.load_size=[640,960] \
#     data.downscale=1 \
#     data.load_rgb=True \
#     data.load_sky_mask=True \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     nerf.model.head.enable_shadow_head=True \
#     supervision.depth.enable_direct_lidar_sup=False \
#     supervision.depth.line_of_sight.enable=True \
#     supervision.depth.line_of_sight.start_iter=2000 \
#     supervision.depth.line_of_sight.loss_coef=0.01 \
#     supervision.depth.line_of_sight.decay_rate=0.5 \
#     supervision.depth.line_of_sight.decay_steps=3000 \
#     supervision.depth.line_of_sight.start_epsilon=5.0 \
#     supervision.depth.line_of_sight.end_epsilon=1.0 \
#     supervision.hard_surface.enable=False \
#     supervision.hard_surface.loss_coef=0.1 \
#     supervision.crisp_entropy.enable=False \
#     supervision.depth.loss_coef=0.5 \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     optim.pretrain_wt_lidar_only.enable=False \
#     logging.vis_freq=2000 \
#     nerf.estimator.propnet.enable_temporal_propnet=False \
#     lidar_evaluation.save_lidar_simulation=True \

# sequence_idx=3
# start_timestamp=0
# num_timestamps=-1
# num_iters=4000
# project=0827_debug_depthrmse
# python train_good_nerf.py \
#     --config_file configs/hash_default_config.yaml \
#     --output_root ./work_dirs/waymo_dev_static/ \
#     --project $project \
#     --run_name ${sequence_idx}_baseline \
#     data.downscale=4 \
#     data.test_holdout=0 \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     data.num_cams=3 \
#     nerf.model.head.enable_cam_embedding=False \
#     nerf.model.head.enable_img_embedding=True \
#     nerf.model.head.appearance_embedding_dim=4 \
#     logging.vis_freq=$num_iters \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     supervision.depth.enable_direct_lidar_sup=True \
#     supervision.depth.enable_freespace_loss=True \
#     lidar_evaluation.eval_lidar=True \
#     render.render_novel_trajectory=True \


# sequence_idx=754
# start_timestamp=0
# num_timestamps=-1
# num_iters=4000
# target_dino_dim=8
# project=0828_debug_static_dino
# python train_good_nerf.py \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/dev_dynamic_dino/ \
#     --project $project \
#     --run_name ${sequence_idx} \
#     data.test_holdout=0 \
#     data.downscale=4 \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     data.load_dino=True \
#     data.target_dino_dim=$target_dino_dim \
#     nerf.model.head.enable_dino_head=True \
#     nerf.model.head.dino_mlp_layer_width=64 \
#     nerf.model.head.dino_embedding_dim=$target_dino_dim \
#     nerf.model.head.enable_learnable_pe=False \
#     lidar_evaluation.save_lidar_simulation=False \


# sequence_idx=527
# start_timestamp=0
# num_timestamps=-1
# num_iters=5000
# target_dino_dim=8
# project=0829_ray_sampling_interval
# python train_good_nerf.py \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_baseline \
#     data.test_holdout=0 \
#     data.downscale=2 \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     nerf.model.sampling.enable_coarse_to_fine_ray_sampling=False \
#     supervision.depth.enable_direct_lidar_sup=False \
#     lidar_evaluation.save_lidar_simulation=False \


# sequence_idx=527
# start_timestamp=90
# num_timestamps=-1
# num_iters=10000
# project=0829_occ_grid
# python train_good_nerf_debug_occ.py \
#     --config_file configs/hash_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_large_wt_occ_test \
#     data.test_holdout=0 \
#     data.downscale=2 \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     nerf.model.sampling.enable_coarse_to_fine_ray_sampling=False \
#     nerf.model.head.enable_sky_head=True \
#     supervision.depth.enable_direct_lidar_sup=False \
#     lidar_evaluation.save_lidar_simulation=False \
#     lidar_evaluation.eval_lidar=False \

# sequence_idx=527
# start_timestamp=90
# num_timestamps=10
# num_iters=1000
# project=0830_occ_grid
# python train_good_nerf_debug_occ.py \
#     --config_file configs/hash_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_debug \
#     data.test_holdout=0 \
#     data.downscale=1 \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     nerf.estimator.type="occgrid" \
#     nerf.model.sampling.enable_coarse_to_fine_ray_sampling=False \
#     supervision.depth.enable_direct_lidar_sup=True \
#     lidar_evaluation.save_lidar_simulation=False \


# sequence_idx=527
# start_timestamp=90
# num_timestamps=-1
# num_iters=10000
# target_dino_dim=8
# project=0830_arch_debug
# CUDA_LAUNCH_BLOCKING=1 python train_good_nerf.py \
#     --config_file configs/hash_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_occ \
#     data.test_holdout=0 \
#     data.downscale=1 \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     nerf.estimator.type="occgrid" \
#     nerf.model.sampling.enable_coarse_to_fine_ray_sampling=True \
#     supervision.depth.enable_direct_lidar_sup=True \
#     lidar_evaluation.save_lidar_simulation=False \
#     resume_from=./work_dirs/0830_arch_debug/527_occ/checkpoint_10000.pth

# sequence_idx=527
# start_timestamp=90
# num_timestamps=10
# num_iters=500
# target_dino_dim=8
# project=0901
# CUDA_LAUNCH_BLOCKING=1 python train_good_nerf.py \
#     --config_file configs/hash_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_remove_occ \
#     data.test_holdout=0 \
#     data.downscale=1 \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     nerf.estimator.type="propnet" \
#     nerf.model.sampling.enable_coarse_to_fine_ray_sampling=True \
#     supervision.depth.enable_direct_lidar_sup=False \
#     lidar_evaluation.save_lidar_simulation=False \




# sequence_idx=35
# start_timestamp=0
# num_timestamps=-1
# num_iters=10000
# project=dynamic_debug
# CUDA_LAUNCH_BLOCKING=1 python train_good_nerf.py \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/0901 \
#     --project $project \
#     --run_name ${sequence_idx}_baseline_updated_density_field \
#     data.test_holdout=0 \
#     data.downscale=2 \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     nerf.estimator.type="propnet" \
#     nerf.model.sampling.enable_coarse_to_fine_ray_sampling=True \
#     supervision.depth.enable_direct_lidar_sup=False \
#     lidar_evaluation.save_lidar_simulation=False \

# sequence_idx=754
# start_timestamp=0
# num_timestamps=-1
# num_iters=8000
# project=dynamic_debug
# python train_good_nerf.py \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/0901 \
#     --project $project \
#     --run_name ${sequence_idx}_wt_tempprop_alpha \
#     data.test_holdout=0 \
#     data.downscale=2 \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     nerf.estimator.type="propnet" \
#     nerf.estimator.propnet.enable_temporal_propnet=True \
#     nerf.model.sampling.enable_coarse_to_fine_ray_sampling=True \
#     supervision.depth.enable_direct_lidar_sup=False \
#     lidar_evaluation.save_lidar_simulation=False \

# python train_good_nerf.py \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/0901 \
#     --project $project \
#     --run_name ${sequence_idx}_wt_tempprop_density \
#     data.test_holdout=0 \
#     data.downscale=2 \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     nerf.estimator.type="propnet" \
#     nerf.estimator.propnet.enable_temporal_propnet=True \
#     nerf.model.sampling.enable_coarse_to_fine_ray_sampling=True \
#     supervision.depth.enable_direct_lidar_sup=False \
#     lidar_evaluation.save_lidar_simulation=False \
# sequence_idx=754
# start_timestamp=0
# num_timestamps=-1
# num_iters=8000
# project=flow_debug
# python train_good_nerf.py \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/0901 \
#     --project $project \
#     --run_name ${sequence_idx}_large_loss_coef \
#     data.test_holdout=0 \
#     data.downscale=2 \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     nerf.estimator.type="propnet" \
#     nerf.estimator.propnet.enable_temporal_propnet=False \
#     nerf.model.sampling.enable_coarse_to_fine_ray_sampling=True \
#     nerf.model.head.enable_flow_encoder=True \
#     supervision.dynamic.loss_coef=0.05 \
#     supervision.depth.enable_direct_lidar_sup=False \
#     lidar_evaluation.save_lidar_simulation=False \


# sequence_idx=754
# start_timestamp=0
# num_timestamps=-1
# num_iters=8000
# project=flow_debug
# python train_good_nerf.py \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/0901 \
#     --project $project \
#     --run_name ${sequence_idx} \
#     --eval_only \
#     data.test_holdout=0 \
#     data.downscale=2 \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     nerf.estimator.type="propnet" \
#     nerf.estimator.propnet.enable_temporal_propnet=False \
#     nerf.model.sampling.enable_coarse_to_fine_ray_sampling=True \
#     nerf.model.head.enable_flow_encoder=True \
#     supervision.depth.enable_direct_lidar_sup=False \
#     lidar_evaluation.save_lidar_simulation=False \
#     resume_from=./work_dirs/0901/flow_debug/754/checkpoint_08000.pth \


# sequence_idx=527
# start_timestamp=90
# num_timestamps=-1
# num_iters=10000
# target_dino_dim=8
# project=depth
# python train_good_nerf.py \
#     --config_file configs/hash_default_config.yaml \
#     --output_root ./work_dirs/0903 \
#     --project $project \
#     --run_name ${sequence_idx}_2xdown2 \
#     data.test_holdout=0 \
#     data.downscale=1.0 \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     nerf.model.sampling.enable_coarse_to_fine_ray_sampling=True \
#     supervision.depth.enable_direct_lidar_sup=False \
#     lidar_evaluation.save_lidar_simulation=False \
#     render.render_novel_trajectory=False 
    # resume_from=./work_dirs/0903/depth/527_load_small_but_no_downscale/checkpoint_20000.pth



# sequence_idx=527
# start_timestamp=150
# num_timestamps=-1
# num_iters=20000
# target_dino_dim=8
# project=depth
# CUDA_VISIBLE_DEVICES=1 python train_good_nerf.py \
#     --config_file configs/hash_default_config.yaml \
#     --output_root ./work_dirs/0903 \
#     --project $project \
#     --run_name ${sequence_idx}_load_small_but_no_downscale \
#     --render_supervision_video_only \
#     data.test_holdout=0 \
#     data.downscale=1.0 \
#     data.load_size=[640,960] \
#     render.render_low_res=True \
#     render.low_res_downscale=2 \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     nerf.estimator.type="propnet" \
#     nerf.model.sampling.enable_coarse_to_fine_ray_sampling=True \
#     supervision.depth.enable_direct_lidar_sup=False \
#     lidar_evaluation.save_lidar_simulation=False \
#     render.render_novel_trajectory=False  \

# sequence_idx=200
# start_timestamp=0
# num_timestamps=-1
# num_iters=10000
# project=depth_l2_loss_1.0
# python train_good_nerf.py \
#     --config_file configs/hash_default_config.yaml \
#     --output_root ./work_dirs/0904_depth_ablation/ \
#     --project $project \
#     --run_name ${sequence_idx} \
#     --also_save_seperate_videos \
#     data.test_holdout=0 \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     supervision.depth.loss_type="l2" \
#     supervision.depth.loss_coef=1.0 \
#     nerf.model.head.enable_cam_embedding=False \
#     nerf.model.head.enable_img_embedding=True \
#     nerf.model.head.appearance_embedding_dim=4 \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     lidar_evaluation.save_lidar_simulation=False \
#     resume_from=./work_dirs/0904_depth_ablation/depth_l2_loss_1.0/200/checkpoint_10000.pth \

# sequence_idx=524
# start_timestamp=0
# num_timestamps=-1
# num_iters=5000
# project=coarse_to_fine_hash_mask
# python train_good_nerf.py \
#     --config_file configs/hash_default_config.yaml \
#     --output_root ./work_dirs/0904_depth_ablation/ \
#     --project $project \
#     --run_name ${sequence_idx} \
#     data.test_holdout=0 \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     supervision.depth.loss_type="smooth_l1" \
#     supervision.depth.loss_coef=0.1 \
#     supervision.crisp_entropy.enable=True \
#     nerf.model.head.enable_cam_embedding=False \
#     nerf.model.head.enable_img_embedding=True \
#     nerf.model.head.appearance_embedding_dim=4 \
#     nerf.model.scene.lidar_percentile=0.3 \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     optim.progressive_hash_encoding.enable=True \
#     logging.vis_freq=500 \
#     lidar_evaluation.save_lidar_simulation=False \


# sequence_idx=700
# start_timestamp=30
# num_timestamps=100
# num_iters=5000
# project=0904_new
# python train_good_nerf.py \
#     --config_file configs/debug.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_flow \
#     data.test_holdout=0 \
#     data.downscale=1 \
#     data.load_size=[320,480] \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     data.sampler.buffer_downscale=8 \
#     supervision.crisp_entropy.enable=True \
#     nerf.model.head.enable_cam_embedding=False \
#     nerf.model.head.enable_img_embedding=True \
#     nerf.model.head.appearance_embedding_dim=4 \
#     nerf.model.head.enable_flow_encoder=False \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     logging.vis_freq=500 \
#     nerf.estimator.propnet.enable_temporal_propnet=False \
#     lidar_evaluation.save_lidar_simulation=False \
#     resume_from=./work_dirs/0904_new/700_flow/checkpoint_05000.pth \
    


# sequence_idx=700
# start_timestamp=0
# num_timestamps=-1
# num_iters=10000
# project=0905
# python train_good_nerf.py \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_baseline \
#     data.test_holdout=0 \
#     data.downscale=2 \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     data.sampler.buffer_downscale=8 \
#     nerf.model.head.enable_cam_embedding=True \
#     nerf.model.head.enable_img_embedding=False \
#     nerf.model.head.appearance_embedding_dim=16 \
#     nerf.model.head.enable_dynamic_branch=True \
#     render.render_low_res=False \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     nerf.estimator.propnet.enable_temporal_propnet=False \
#     lidar_evaluation.save_lidar_simulation=True \
#     resume_from=./work_dirs/0905/700_baseline/checkpoint_10000.pth \
    

# sequence_idx=700
# start_timestamp=0
# num_timestamps=-1
# num_iters=5000
# project=0906
# python train_good_nerf.py \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_dynamic_no_rgb_5k_los_no_far \
#     data.test_holdout=0 \
#     data.downscale=2 \
#     data.load_rgb=False \
#     data.load_sky_mask=False \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     nerf.estimator.propnet.num_samples_per_prop=[128,64] \
#     nerf.estimator.propnet.far_plane=1000.0 \
#     nerf.sampling.num_samples=64 \
#     supervision.depth.enable_direct_lidar_sup=True \
#     supervision.depth.line_of_sight.enable=True \
#     supervision.depth.line_of_sight.start_iter=200 \
#     supervision.depth.line_of_sight.loss_coef=0.01 \
#     supervision.depth.line_of_sight.start_epsilon=0.5 \
#     supervision.depth.line_of_sight.end_epsilon=0.2 \
#     supervision.hard_surface.enable=False \
#     supervision.hard_surface.loss_coef=0.1 \
#     supervision.crisp_entropy.enable=False \
#     supervision.depth.loss_coef=1.0 \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     logging.vis_freq=1000 \
#     nerf.estimator.propnet.enable_temporal_propnet=False \
#     lidar_evaluation.save_lidar_simulation=True \
#     resume_from=./work_dirs/0906/700_dynamic_no_rgb_5k_los_no_far/checkpoint_05000.pth



# sequence_idx=700
# start_timestamp=0
# num_timestamps=-1
# num_iters=15000
# project=0906
# python train_good_nerf.py \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_both \
#     data.test_holdout=0 \
#     data.downscale=2 \
#     data.load_rgb=True \
#     data.load_sky_mask=True \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     nerf.estimator.propnet.num_samples_per_prop=[128,64] \
#     nerf.estimator.propnet.far_plane=1000.0 \
#     nerf.sampling.num_samples=64 \
#     supervision.depth.enable_direct_lidar_sup=True \
#     supervision.depth.line_of_sight.enable=True \
#     supervision.depth.line_of_sight.start_iter=1000 \
#     supervision.depth.line_of_sight.loss_coef=0.01 \
#     supervision.depth.line_of_sight.end_epsilon=1.0 \
#     supervision.depth.line_of_sight.start_epsilon=3.0 \
#     supervision.hard_surface.enable=False \
#     supervision.hard_surface.loss_coef=0.1 \
#     supervision.crisp_entropy.enable=False \
#     supervision.depth.loss_coef=1.0 \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     logging.vis_freq=2000 \
#     nerf.estimator.propnet.enable_temporal_propnet=False \
#     lidar_evaluation.save_lidar_simulation=True \
    

# sequence_idx=700
# start_timestamp=0
# num_timestamps=-1
# num_iters=10000
# project=0906
# python train_good_nerf.py \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_dynamic_10k \
#     data.test_holdout=0 \
#     data.load_size=[320,480] \
#     data.downscale=1 \
#     data.load_rgb=True \
#     data.load_sky_mask=True \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     nerf.estimator.propnet.num_samples_per_prop=[128,64] \
#     nerf.estimator.propnet.far_plane=1000.0 \
#     nerf.sampling.num_samples=64 \
#     supervision.depth.enable_direct_lidar_sup=True \
#     supervision.depth.line_of_sight.enable=True \
#     supervision.depth.line_of_sight.start_iter=1000 \
#     supervision.depth.line_of_sight.loss_coef=0.001 \
#     supervision.depth.line_of_sight.start_epsilon=2.0 \
#     supervision.depth.line_of_sight.end_epsilon=1.0 \
#     supervision.hard_surface.enable=False \
#     supervision.hard_surface.loss_coef=0.1 \
#     supervision.crisp_entropy.enable=False \
#     supervision.depth.loss_coef=0.5 \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters \
#     optim.pretrain_wt_lidar_only.enable=True \
#     logging.vis_freq=2000 \
#     nerf.estimator.propnet.enable_temporal_propnet=False \
#     lidar_evaluation.save_lidar_simulation=False \
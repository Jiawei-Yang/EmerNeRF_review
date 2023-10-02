
export CUDA_VISIBLE_DEVICES=1
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
    --run_name ${sequence_idx}_dino \
    data.test_holdout=0 \
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
    nerf.model.head.enable_temporal_interpolation=False \
    data.load_dino=True \
    data.dino_model_type="dinov2_vitb14" \
    data.dino_stride=7 \
    data.dino_extraction_size=[644,966] \
    data.target_dino_dim=64 \
    nerf.model.head.dino_embedding_dim=64 \
    nerf.model.head.enable_learnable_pe=True \
    nerf.model.head.enable_dino_head=True \
    logging.vis_freq=1000 \
    logging.saveckpt_freq=100000 \
    optim.num_iters=$num_iters

# sequence_idx=700
# start_timestamp=0
# num_timestamps=-1
# num_iters=5000
# project=1002_baseline_new
# python train.py \
#     --config_file configs/hash_dynamic_default_config.yaml \
#     --output_root ./work_dirs/ \
#     --project $project \
#     --run_name ${sequence_idx}_baseline_no_flow_new_lr \
#     data.test_holdout=0 \
#     data.downscale=1 \
#     data.load_size=[160,240] \
#     data.preload_device="cuda" \
#     data.sequence_idx=$sequence_idx \
#     data.start_timestamp=$start_timestamp \
#     data.num_timestamps=$num_timestamps \
#     data.sampler.buffer_downscale=8 \
#     render.render_novel_trajectory=False \
#     lidar_evaluation.save_lidar_simulation=False \
#     nerf.model.head.enable_cam_embedding=False \
#     nerf.model.head.enable_img_embedding=True \
#     nerf.model.head.appearance_embedding_dim=16 \
#     nerf.model.head.enable_flow_branch=False \
#     nerf.model.head.enable_temporal_interpolation=False \
#     logging.vis_freq=1000 \
#     logging.saveckpt_freq=$num_iters \
#     optim.num_iters=$num_iters

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



# EmerNeRF_review

#### Installation

```
conda create -n emernerf_local python=3.9
pip install -r requirements.txt
```

Manually install nerfacc and tiny-cuda-nn. Note that these installations may take more than 30 minutes.

```
pip install git+https://github.com/KAIR-BAIR/nerfacc.git
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

If you encounter an error `nvcc fatal : Unsupported gpu architecture 'compute_89` during the installation of tiny-cuda-nn, try the following command:

```
TCNN_CUDA_ARCHITECTURES=86 pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

#### Dataset preparation

Sample code:

```
python datasets/download_waymo.py --target_dir ./data/waymo_raw --scene_ids 700
```

Run:

```
cd $PROJECT_ROOT
sequence_idx=700
start_timestamp=0
num_timestamps=-1

num_iters=25000
project=0925_diverse_img_dinov1_noflow_pos
python train_good_nerf.py \
    --enable_wandb \
    --config_file configs/hash_dynamic_default_config.yaml \
    --output_root ./work_dirs/0925/diverse/ \
    --project $project \
    --run_name ${sequence_idx} \
    data.test_holdout=0 \
    data.preload_device="cuda" \
    data.skip_dino_model=True \
    data.sequence_idx=$sequence_idx \
    data.start_timestamp=$start_timestamp \
    data.num_timestamps=$num_timestamps \
    nerf.model.head.enable_cam_embedding=False \
    nerf.model.head.enable_img_embedding=True \
    nerf.model.head.enable_time_interpolation=False \
    nerf.model.head.camera_embedding_dim=16 \
    data.load_dino=True \
    data.dino_model_type="dino_vitb8" \
    data.dino_stride=8 \
    data.dino_extraction_size=[640,960] \
    data.target_dino_dim=64 \
    nerf.model.head.dino_embedding_dim=64 \
    nerf.model.head.enable_dino_pix_field=True \
    nerf.model.head.enable_dino_head=True \
    logging.saveckpt_freq=$num_iters \
    optim.num_iters=$num_iters
```

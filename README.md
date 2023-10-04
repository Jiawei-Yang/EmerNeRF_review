# EmerNeRF_review

### Dataset split

For Waymo Open Dataset, we first sort the scenes' string names in alphabetical order, store them at `data/waymo_train_list.txt`, and then use the line number - 1 as the scene index. The split of NOTR is as follows:

Static-32: 3, 19, 36, 69, 81, 126, 139, 140, 146, 148, 157, 181, 200, 204, 226, 232, 237, 241, 245, 246, 271, 297, 302, 312, 314, 362, 482, 495, 524, 527, 753, 780

Dynamic-32: 16, 21, 22, 25, 31, 34, 35, 49, 53, 80, 84, 86, 89, 94, 96, 102, 111, 222, 323, 323, 382, 382, 402, 402, 427, 427, 438, 438, 546, 581, 592, 620, 640, 700, 754, 795, 796

Diverse-56

- Ego-static: 1, 23, 24, 37, 66, 108, 114, 115
- Dusk/Dawn: 124, 147, 206, 213, 574, 680, 696, 737
- Gloomy: 47, 205, 220, 284, 333, 537, 699, 749
- Exposure mismatch: 58, 93, 143, 505, 545, 585, 765, 766
- Nighttime: 7, 15, 30, 51, 130, 133, 159, 770
- Rainy: 44, 56, 244, 449, 688, 690, 736, 738
- High-speed: 2, 41, 46, 62, 71, 73, 82, 83

### Installation

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

### Dataset preparation

Once the installation is success

Download raw data per your need. For example, to download the 114th and 700th scene of Waymo Open Dataset, run:

```
python datasets/download_waymo.py --target_dir ./data/waymo_raw --scene_ids 114 700
```

After downloading the raw data, we need to preprocess the data to extract the camera poses and the images. To do so, run:

```
python preprocess.py --data_root data/waymo_raw/ --target_dir data/waymo_processed --split training --workers 2 --scene_ids 114 700
```

### Sky mask

We use ViT-adapater to extract sky masks. We refer readers to [their repo](https://github.com/czczup/ViT-Adapter/tree/main/segmentation) for more details. The full sky mask will be released soon. We provide the sky mask for the 114th and 700th scenes here (anonymized): <https://drive.google.com/drive/folders/1EHNjhEiifxyx3ndIbU-Q8ibslxeVpbqN?usp=sharing>

### Train the model

We cover the usage of different arguments in `configs/hash_default_config.yaml`. Please refer to it for more details.

A sample code to run the training is as follows:

```
cd $PROJECT_ROOT # your project root
sequence_idx=700
start_timestamp=0
num_timestamps=-1
num_iters=25000

project=emerged_flow
python train.py \
    --enable_wandb \
    --config_file configs/hash_dynamic_default_config.yaml \
    --output_root ./work_dirs/dynamic/ \
    --project $project \
    --run_name ${sequence_idx} \
    data.test_holdout=0 \
    data.preload_device="cuda" \
    data.sequence_idx=$sequence_idx \
    data.start_timestamp=$start_timestamp \
    data.num_timestamps=$num_timestamps \
    nerf.model.head.enable_cam_embedding=False \
    nerf.model.head.enable_img_embedding=True \
    nerf.model.head.appearance_embedding_dim=16 \
    nerf.model.head.enable_flow_branch=True \
    logging.saveckpt_freq=$num_iters \
    optim.num_iters=$num_iters
```

To run with DINOv2 features:

```
sequence_idx=700
start_timestamp=0
num_timestamps=-1
num_iters=25000
project=emernerf_dinov2_pe_free
python train.py \
    --enable_wandb \
    --config_file configs/hash_dynamic_default_config.yaml \
    --output_root ./work_dirs/dynamic/ \
    --project $project \
    --run_name ${sequence_idx} \
    data.test_holdout=0 \
    data.preload_device="cuda" \
    data.sequence_idx=$sequence_idx \
    data.start_timestamp=$start_timestamp \
    data.num_timestamps=$num_timestamps \
    nerf.model.head.enable_cam_embedding=False \
    nerf.model.head.enable_img_embedding=True \
    nerf.model.head.appearance_embedding_dim=16 \
    nerf.model.head.enable_flow_branch=True \
    data.load_dino=True \
    data.dino_model_type="dinov2_vitb14" \
    data.dino_stride=7 \
    data.dino_extraction_size=[644,966] \
    data.target_dino_dim=64 \
    nerf.model.head.dino_embedding_dim=64 \
    nerf.model.head.enable_learnable_pe=True \ 
    nerf.model.head.enable_dino_head=True \
    logging.saveckpt_freq=$num_iters \
    optim.num_iters=$num_iters
```

Set `nerf.model.head.enable_learnable_pe=False` to disable the PE decomposition.

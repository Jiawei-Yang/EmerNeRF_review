import argparse
from datasets.waymo_preprocess import WaymoProcessor
import numpy as np

if __name__ == "__main__":
    # python preprocess.py --data_root data/raw/training --target_dir data/waymo_processed --split training --workers 32 --process_keys lidar --start_idx 0 --num_sequences 32
    # python preprocess.py --data_root data/waymo_raw/ --target_dir data/waymo_processed --split training --workers 1 --scene_ids 700
    parser = argparse.ArgumentParser(description="Data converter arg parser")
    parser.add_argument(
        "--data_root", type=str, required=True, help="root path of waymo dataset"
    )
    parser.add_argument("--split", type=str, default="training", help="split name")
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="output directory of processed data",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="number of threads to be used"
    )
    parser.add_argument(
        "--scene_ids",
        default=None,
        type=int,
        nargs="+",
        help="scene ids to be processed, a list of integers separated by space. Range: [0, 798] for training, [0, 202] for validation",
    )

    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="If no scene id is given, use start_idx and num_sequences to generate scene_ids_list",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=200,
        help="number of sequences to be processed",
    )
    parser.add_argument(
        "--process_keys",
        nargs="+",
        default=[
            "images",
            "lidar",
            "calib",
            "pose",
            "velocity",
            "frame_info",
        ],
    )
    args = parser.parse_args()
    if args.scene_ids is not None:
        scene_ids_list = args.scene_ids
    else:
        scene_ids_list = np.arange(args.start_idx, args.start_idx + args.num_sequences)
    if "flow" in args.process_keys:
        # To be released
        from datasets.waymo_preprocess_flow import WaymoProcessor

        args.data_root = args.data_root.replace("raw", "raw_flow")
    else:
        from datasets.waymo_preprocess import WaymoProcessor
    waymo_processor = WaymoProcessor(
        load_dir=args.data_root,
        save_dir=args.target_dir,
        prefix=args.split,
        process_keys=args.process_keys,
        process_id_list=scene_ids_list,
        workers=args.workers,
    )
    if args.scene_ids is not None and len(scene_ids_list) == 1:
        waymo_processor.convert_one(args.scene_ids[0])
    else:
        waymo_processor.convert()

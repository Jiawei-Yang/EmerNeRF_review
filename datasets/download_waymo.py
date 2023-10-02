import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List

# Download the Waymo dataset from Google Cloud Storage
# note: `gcloud auth login` is required before running this script


def download_file(filename, target_dir, source):
    print(f"Dowloading {filename} ...")
    subprocess.run(
        [
            "gsutil",
            "cp",
            "-n",
            f"{source}/{filename}.tfrecord",
            target_dir,
        ]
    )


def download_files(
    file_names: List[str],
    target_dir: str,
    source: str = "gs://waymo_open_dataset_v_1_4_2/individual_files/training",
) -> None:
    """
    Downloads a list of files from a given source to a target directory using multiple threads.

    Args:
        file_names (List[str]): A list of file names to download.
        target_dir (str): The target directory to save the downloaded files.
        source (str, optional): The source directory to download the files from. Defaults to "gs://waymo_open_dataset_v_1_4_2/individual_files/training".
    """
    # Get the total number of file_names
    total_files = len(file_names)

    # Use ThreadPoolExecutor to manage concurrent downloads
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(download_file, filename, target_dir, source)
            for filename in file_names
        ]

        for counter, future in enumerate(futures, start=1):
            # Wait for the download to complete and handle any exceptions
            try:
                future.result()
                print(f"[{counter}/{total_files}] Downloaded successfully!")
            except Exception as e:
                print(f"[{counter}/{total_files}] Failed to download. Error: {e}")


if __name__ == "__main__":
    # Sample code:
    #   python datasets/download_waymo.py --target_dir ./data/waymo_raw --scene_ids 700
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_dir",
        type=str,
        default="data/waymo_raw",
        help="Path to the target directory",
    )
    parser.add_argument(
        "--scene_ids", type=int, nargs="+", help="Scene IDs to download"
    )
    parser.add_argument(
        "--download_flow",
        action="store_true",
        help="Whether to download the flow dataset",
    )
    args = parser.parse_args()

    if args.download_flow:
        args.target_dir = f"{args.target_dir}_flow"
        source = "gs://waymo_open_dataset_scene_flow/train"
    else:
        source = "gs://waymo_open_dataset_v_1_4_2/individual_files/training"
    os.makedirs(args.target_dir, exist_ok=True)
    total_list = open("data/waymo_train_list.txt", "r").readlines()
    file_names = [total_list[i].strip() for i in args.scene_ids]
    download_files(file_names, args.target_dir)

""" THis module creates an initial dataset from manualy selected images.
"""
import random
import argparse
from typing import List
from pathlib import Path
import pandas as pd



def parse_args():
    """ Return arguments for dataset generation when the module is run as main.

    Returns:
        arguments (argparse.Namespace): Program arguments
    """
    parser = argparse.ArgumentParser(description="Parse arguments for dataset creation.")

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/task1_dataset",
        help="Path to the dataset directory."
    )
    parser.add_argument(
        "--train_metadata",
        type=str,
        default="data/signverOD/train.csv",
        help="Path to the train metadata from original dataset."
    )
    parser.add_argument(
        "--test_metadata",
        type=str,
        default="data/signverOD/test.csv",
        help="Path to the test metadata from original dataset."
    )
    parser.add_argument(
        "--id2img",
        type=str,
        default="data/signverOD/image_ids.csv",
        help="Path to the mappings between image name and image ID."
    )
    return parser.parse_args()


def get_img_names(dataset_dir: str) -> List[str]:
    """ Get all the .png files from 'dataset_dir/images/' directory
    and return list of the file names.

    Args:
        dataset_dir (str): Path to the dataset

    Returns:
        List[str]: List of image file names.
    """
    img_path = Path(dataset_dir) / "images"
    # Get a list of all .png files in the directory
    png_files = list(img_path.glob("*.png"))
    png_files = [name.name for name in png_files]
    return png_files




def main(args):
    """ Loads all necessary metadata to create one metadata file for manually
    selected images.

    Args:
        args (argparse.Namespace): Program arguments.
    """

    # 1. Load metadata
    train_data = pd.read_csv(args.train_metadata)
    test_data = pd.read_csv(args.test_metadata)
    metadata = pd.concat([train_data, test_data])

    id2img = pd.read_csv(args.id2img)

    # 2. Select images for our dataset

    # !!! this has duplicate ids
    # get an img id based on img name
    selected_imgs = get_img_names(args.dataset_dir)
    selected_id2img = id2img[id2img["file_name"].isin(
        selected_imgs)][["id", "file_name"]].reset_index(drop=True)

    # select only bbox data for selected images
    selected_md = metadata[metadata["image_id"].isin(
        selected_id2img["id"].to_list())].reset_index(drop=True)

    # select only signature bounding boxes
    selected_md = selected_md[selected_md["category_id"] == 1]

    # add image file names to the bbox metadata
    bbox_data = pd.merge(selected_md, selected_id2img,
                         how='left', left_on="image_id",
                         right_on="id", suffixes=("_bbox", "_img"))

    # 3. save corresponding metadata
    bbox_data_path = Path(args.dataset_dir) / "data.csv"
    bbox_data.to_csv(bbox_data_path, encoding='utf-8', index=False)


if __name__ == "__main__":
    random.seed(42)
    args = parse_args()
    main(args)

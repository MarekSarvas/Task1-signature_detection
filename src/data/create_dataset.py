""" THis module creates an initial dataset from manualy selected images.
"""
import random
import argparse
from typing import List, Dict
from pathlib import Path
import ast

import cv2
import pandas as pd
import numpy as np
from dataclasses import dataclass
import tqdm


from create_signature_class_imgs import PERSONA_LIST


@dataclass
class Bbox:
    x: int
    y: int
    w: int
    h: int


def parse_args():
    """ Return arguments for dataset generation when the module is run as main.

    Returns:
        arguments (argparse.Namespace): Program arguments
    """
    parser = argparse.ArgumentParser(description="Parse arguments for dataset creation.")

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/FINAL_dataset",
        help="Path to the dataset directory."
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="data/signverOD/images",
        help="Path to the dataset directory."
    )
    parser.add_argument(
        "--train_metadata",
        type=str,
        default="data/signverOD_fixed/train.csv",
        help="Path to the train metadata from original dataset."
    )
    parser.add_argument(
        "--test_metadata",
        type=str,
        default="data/signverOD_fixed/test.csv",
        help="Path to the test metadata from original dataset."
    )
    parser.add_argument(
        "--id2img",
        type=str,
        default="data/signverOD_fixed/image_ids.csv",
        help="Path to the mappings between image name and image ID."
    )
    parser.add_argument(
        "--signature_dir",
        type=str,
        default="data/persona_signatures",
        help=""
    )
    parser.add_argument(
        "--n_images",
        type=int,
        default=400,
        help=""
    )
    return parser.parse_args()


def get_img_names(dataset_dir: str) -> List[str]:
    """ Get all the .png files from 'dataset_dir/images/' directory
    and return list of the file names.

    Args:
        dataset_dir (str): Path to the dataset.

    Returns:
        List[str]: List of image file names.
    """
    img_path = Path(dataset_dir) / "images"
    # Get a list of all .png files in the directory
    png_files = list(img_path.glob("*.png"))
    png_files = [name.name for name in png_files]
    return png_files


def filter_bbox_category(df: pd.DataFrame) -> pd.DataFrame:
    """Removes all bounding boxes that are not signatures

    Args:
        df (pd.DataFrame): Dataset.

    Returns:
        pd.DataFrame: Dataset containing only signature bounding boxes.
    """
    df = df[df["category_id"] == 1]
    return df


def relative_to_pixel(bbox: List, img_width: int, img_height: int) -> Bbox:
    """
    Args:
        bbox (Bbox): Bounding box with relative coordinates.
        img_width (int):
        img_height (int):

    Returns:
        Bbox: Bounding box with pixel value coordinates.
    """
    x = int(bbox[0] * img_width)
    y = int(bbox[1] * img_height)
    rect_width = int(bbox[2] * img_width)
    rect_height = int(bbox[3] * img_height)

    return Bbox(x, y, rect_width, rect_height)


def insert_signature(image, bbox, signature_file):
    # 1. recalculate relative bbox coords into pixels
    img_width = image.shape[1]
    img_height = image.shape[0]
    # load bbox
    bbox = ast.literal_eval(bbox)
    # x, y, height, length
    orig_bbox = np.array(bbox)
    bbox = relative_to_pixel(orig_bbox, img_width=img_width, img_height=img_height)

    signature = cv2.imread(signature_file, cv2.IMREAD_GRAYSCALE)
    signature = cv2.resize(signature, (bbox.w, bbox.h))

    image[bbox.y:bbox.y+bbox.h, bbox.x:bbox.x+bbox.w] = signature

    # adjust bbox coords for yolo
    orig_bbox[0] = orig_bbox[0] + (orig_bbox[2]/2)
    orig_bbox[1] = orig_bbox[1] + (orig_bbox[3]/2)

    return image, orig_bbox


def sample_personas(n_personas: int) -> Dict:
    """
    Args:
        n_personas (int):

    Raises:
        Exception: If we want to sample more than 2 people which should not happen.

    Returns:
        Dict: Dictionary where keys are persona_id and value is persona name, e.g. "ee": "employee".
    """
    persona_id = random.randint(0, len(PERSONA_LIST)-1)

    if n_personas == 1:
        persona_type = random.choice(list(PERSONA_LIST[persona_id].keys()))
        return  {persona_type:PERSONA_LIST[persona_id][persona_type]}
    if n_personas == 2:
        return PERSONA_LIST[persona_id]
    # should not happen
    raise Exception("Cannot sample more than 2 people signature for the document.")


def sample_signature(sign_dir: Path, suffix: str) -> str:
    """
    Args:
        sign_dir (Path): Path to dir containing all signatures.
        suffix (str): Sample only signatures with this file name suffix.

    Returns:
        str: Path to the signature image.
    """
    sign_dir = Path(sign_dir)
    signature_file = random.choice(list(sign_dir.glob(suffix)))
    return signature_file


def store_yolo_labels(image_data: List[Dict], img_stem: str, out_path: Path):
    """Stores bbox coordinates and bbox label in format for yolo11

    Args:
        image_data (List[Dict]):
        img_stem (str): Name of the image.
        out_path (Path): Root for path where the labels will be stored.
    """
    df = pd.DataFrame(image_data)
    df["is_signed"] = df["is_signed"].apply(lambda x: 1 if x == "signed" else 0)

    label_path = out_path.parent/ "labels" / f"{img_stem}.txt"
    df.to_csv(label_path, index=False, header=False, sep=" ")


def create_signed_documents(image_path: Path, signature_split_dir: Path, bbox:pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """From each document in dataset creates signed, unsigned and 
    "half-signed" (if there are 2 people)  versions.
    Stores the documents and all additional metadata.

    Args:
        image_path (Path):
        signature_split_dir (Path): Path to dir with signature images for current split.
        bbox (pd.DataFrame): DataFrame containing all bboxes in current image.
        out_path (Path):

    Returns:
        pd.DataFrame: Metadata for all created documents.
    """
    documents_metadata = []
    # 1. Create signed documents
    data_for_yolo = []
    personas = sample_personas(len(bbox))
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_img_path = out_path / f"{image_path.stem}_signed.png"
    for i, (persona_type, persona_name) in enumerate(personas.items()):

        sample_from = signature_split_dir.joinpath(f"{persona_type}_signed")
        suffix = f"*{persona_name}.png"
        signature_file = sample_signature(sample_from, suffix)

        # bbox is x,y,w,h
        image, new_bbox = insert_signature(image, bbox.iloc[i], signature_file)
        documents_metadata.append({"file_name": new_img_path.stem,
                                   "bbox": new_bbox,
                                   "is_signed": "signed",
                                   "persona_name": persona_name,
                                   "persona_type": persona_type}) 
        data_for_yolo.append({"is_signed": "signed",
                                "x": new_bbox[0], 
                                "y": new_bbox[1], 
                                "w": new_bbox[2], 
                                "h": new_bbox[3]} 
                                )
    store_yolo_labels(data_for_yolo, new_img_path.stem, out_path)

    # Save image and corresponding metadata
    cv2.imwrite(new_img_path, image)

    # 2. Create unsigned documents
    data_for_yolo = []
    personas = sample_personas(len(bbox))
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_img_path = out_path / f"{image_path.stem}_unsigned.png"
    for i, (persona_type, persona_name) in enumerate(personas.items()):

        sample_from = signature_split_dir.joinpath(f"{persona_type}_unsigned")
        suffix = f"*{persona_name}.png"
        signature_file = sample_signature(sample_from, suffix)

        image, new_bbox = insert_signature(image, bbox.iloc[i], signature_file)
        documents_metadata.append({"file_name": new_img_path.stem,
                                   "bbox": new_bbox,
                                   "is_signed": "unsigned",
                                   "persona_name": persona_name,
                                   "persona_type": persona_type}) 

        data_for_yolo.append({"is_signed": "unsigned",
                                "x": new_bbox[0], 
                                "y": new_bbox[1], 
                                "w": new_bbox[2], 
                                "h": new_bbox[3]} 
                                )
    store_yolo_labels(data_for_yolo, new_img_path.stem, out_path)
    # Save image and corresponding metadata
    cv2.imwrite(new_img_path, image)


    # 3. Create half signed documents
    data_for_yolo = []
    if len(bbox) == 2:
        # load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        new_img_path = out_path / f"{image_path.stem}_halfsigned.png"

        # 3.1 Insert signed persona
        # sample persona
        persona_type, persona_name = next(iter(personas.items()))
        # sample signature
        sample_from = signature_split_dir.joinpath(f"{persona_type}_signed")
        suffix = f"*{persona_name}.png"
        signature_file = sample_signature(sample_from, suffix)

        # add signature to the image
        image, new_bbox = insert_signature(image, bbox.iloc[0], signature_file)
        documents_metadata.append({"file_name": new_img_path.stem,
                                   "bbox": new_bbox,
                                   "is_signed": "signed",
                                   "persona_name": persona_name,
                                   "persona_type": persona_type}) 
        data_for_yolo.append({"is_signed": "signed",
                                "x": new_bbox[0], 
                                "y": new_bbox[1], 
                                "w": new_bbox[2], 
                                "h": new_bbox[3]})


        # 3.2. Insert unsigned persona
        # sample persona
        personas = sample_personas(1)
        persona_type, persona_name = next(iter(personas.items()))

        # sample signature
        sample_from = signature_split_dir.joinpath(f"{persona_type}_unsigned")
        suffix = f"*{persona_name}.png"
        signature_file = sample_signature(sample_from, suffix)

        # add signature to the image
        image, new_bbox = insert_signature(image, bbox.iloc[1], signature_file)
        documents_metadata.append({"file_name": new_img_path.stem,
                                   "bbox": new_bbox,
                                   "is_signed": "unsigned",
                                   "persona_name": persona_name,
                                   "persona_type": persona_type}) 
            # Save image and corresponding metadata
        cv2.imwrite(new_img_path, image)
        data_for_yolo.append({"is_signed": "unsigned",
                                "x": new_bbox[0], 
                                "y": new_bbox[1], 
                                "w": new_bbox[2], 
                                "h": new_bbox[3]} 
                                )
        store_yolo_labels(data_for_yolo, new_img_path.stem, out_path)

    return pd.DataFrame(documents_metadata)


def create_data_yaml(dataset_dir: Path):
    """
    """
    data_yaml = f"""path: /home/marek/Personal/mama/mamaai_task1/{dataset_dir}
train: train/images
val: valid/images

nc: 2
names: ['unsigned', 'signed']
    """
    with open(Path(dataset_dir) / "data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml)

    test_data_yaml = f"""path: /home/marek/Personal/mama/mamaai_task1/{dataset_dir}
train: train/images
val: test/images

nc: 2
names: ['unsigned', 'signed']
    """
    with open(Path(dataset_dir) / "test_data.yaml", "w", encoding="utf-8") as f:
        f.write(test_data_yaml)




def main(args):
    """ Loads all necessary metadata to create one metadata file for manually
    selected images.

    Args:
        args (argparse.Namespace): Program arguments.
    """
    # 1. Load metadata
    train_data = pd.read_csv(args.train_metadata)
    test_data = pd.read_csv(args.test_metadata)
    id2img = pd.read_csv(args.id2img)

    # select only signature bounding boxes
    train_data = filter_bbox_category(train_data)
    test_data = filter_bbox_category(test_data)

    # 2. add file names into the train/test metadata
    # add image file names to the bbox metadata
    train_data_names = pd.merge(train_data, id2img,
                         how='left', left_on="image_id_new",
                         right_on="id_new", suffixes=("_bbox", "_img"))
    train_data_names = train_data_names.drop(["id_bbox", "id_img"], axis=1)


    test_data_names = pd.merge(test_data, id2img,
                         how='left', left_on="image_id_new",
                         right_on="id_new", suffixes=("_bbox", "_img"))
    test_data_names = test_data_names.drop(["id_bbox", "id_img"], axis=1)

    # 3. Filter out documents with more than 2 signatures
    # train split
    bbox_counts = train_data_names["image_id_new"].value_counts()
    bbox_2 = bbox_counts[bbox_counts <= 2].index
    train_data_filtered = train_data_names[train_data_names["image_id_new"].isin(bbox_2)]

    # test split
    bbox_counts = test_data_names["image_id_new"].value_counts()
    bbox_2 = bbox_counts[bbox_counts <= 2].index
    test_data_filtered = test_data_names[test_data_names["image_id_new"].isin(bbox_2)]

    # Create signed, unsigned and half-signed documents
    all_files = train_data_filtered["file_name"].unique()
    random.shuffle(all_files)
    train_files = all_files[:args.n_images]
    val_files = all_files[args.n_images:args.n_images+100]

    # 4. Create train documents
    train_metadata = []
    signature_split_dir = Path(args.signature_dir) / "train"
    out_path = Path(args.dataset_dir) / "train" / "images"
    labels_out_path = Path(args.dataset_dir) / "train" / "labels"
    # check for paths
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)
    if not labels_out_path.exists():
        labels_out_path.mkdir(parents=True, exist_ok=True)


    for file in tqdm.tqdm(train_files, total=len(train_files)):
        image_path = Path(args.img_dir) / file

        bbox = train_data_filtered[train_data_filtered["file_name"] == file]["bbox"]

        img_metadata = create_signed_documents(image_path, signature_split_dir, bbox, out_path)
        train_metadata.append(img_metadata)

    df = pd.concat(train_metadata)
    df.to_csv(Path(args.dataset_dir) / "train_clean.csv", encoding='utf-8', index=False)

    # 5. Create valid documents
    train_metadata = []
    signature_split_dir = Path(args.signature_dir) / "train"
    out_path = Path(args.dataset_dir) / "valid" / "images"
    labels_out_path = Path(args.dataset_dir) / "valid" / "labels"
    # check for paths
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)
    if not labels_out_path.exists():
        labels_out_path.mkdir(parents=True, exist_ok=True)

    # Create signed, unsigned and half-signed documents
    for file in tqdm.tqdm(val_files, total=len(val_files)):
        image_path = Path(args.img_dir) / file

        bbox = train_data_filtered[train_data_filtered["file_name"] == file]["bbox"]

        img_metadata = create_signed_documents(image_path, signature_split_dir, bbox, out_path)
        train_metadata.append(img_metadata)

    df = pd.concat(train_metadata)
    df.to_csv(Path(args.dataset_dir) / "train_clean.csv", encoding='utf-8', index=False)


    # 5. Create test documents
    test_metadata = []
    signature_split_dir = Path(args.signature_dir) / "test"
    out_path = Path(args.dataset_dir) / "test" / "images"
    labels_out_path = Path(args.dataset_dir) / "test" / "labels"

    # check for paths
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)
    if not labels_out_path.exists():
        labels_out_path.mkdir(parents=True, exist_ok=True)

    # Create signed, unsigned and half-signed documents
    all_files = test_data_filtered["file_name"].unique()[:100]
    for file in tqdm.tqdm(all_files, total=len(all_files)):
        image_path = Path(args.img_dir) / file

        bbox = test_data_filtered[test_data_filtered["file_name"] == file]["bbox"]

        img_metadata = create_signed_documents(image_path, signature_split_dir , bbox, out_path)
        test_metadata.append(img_metadata)

    df = pd.concat(test_metadata)
    df.to_csv(Path(args.dataset_dir) / "test_clean.csv", encoding='utf-8', index=False)

    # 7. create data.yaml
    create_data_yaml(args)


if __name__ == "__main__":
    random.seed(42)
    args = parse_args()
    main(args)

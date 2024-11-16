import argparse
from pathlib import Path

import cv2


def parse_args():
    """ Returns arguments for signature cleaning.

    Returns:
        arguments (argparse.Namespace): Program arguments
    """
    parser = argparse.ArgumentParser(description="Parse arguments for signature data creation.")

    parser.add_argument(
        "--signature_dataset",
        type=str,
        default="data/signatures/sign_data",
        help="Path to the dataset directory."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/signatures_clean",
        help="Path to the dataset directory."
    )
    return parser.parse_args()


def signature_to_black_on_white(image_path: Path, output_path: Path):
    """ Read signatures from downloaded dataset and adjust the contrast 

    Args:
        image_path (Path): Path to original signature image.
        output_path (Path): Path to where the new image will be stored.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply binary thresholding to isolate the signature
    _, binary_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Invert the mask to make the signature black and the background white
    inverted_signature = cv2.bitwise_not(binary_mask)
    cv2.imwrite(output_path, inverted_signature)


def main(args):
    signatures_train =  Path(args.signature_dataset)/"train"
    signatures_test =  Path(args.signature_dataset)/"test"
    signatures_clean_train = Path(args.out_dir)/"train"
    signatures_clean_test = Path(args.out_dir)/"test"

    # check for paths
    if not signatures_clean_train.exists():
        signatures_clean_train.mkdir(parents=True, exist_ok=True)
    if not signatures_clean_test.exists():
        signatures_clean_test.mkdir(parents=True, exist_ok=True)

    # Fix all signatures to black and white to match the documents and
    # save them into args.out_dir train and test set
    for file_path in signatures_train.rglob("*.png"):
        file_name = Path(file_path).name
        signature_to_black_on_white(file_path, signatures_clean_train/file_name)

    for file_path in signatures_test.rglob("*.png"):
        file_name = Path(file_path).name
        signature_to_black_on_white(file_path, signatures_clean_test/file_name)

if __name__ == "__main__":
    args = parse_args()
    main(args)

import sys
import argparse
from pathlib import Path
from typing import Dict

import requests
import cv2
import matplotlib.pyplot as plt


def parse_args():
    """ Returns example inference arguments.

    Returns:
        arguments (argparse.Namespace):
    """
    parser = argparse.ArgumentParser(description="Parse arguments for object detection.")

    parser.add_argument(
        "--img_path",
        type=str,
        default="data/example_data/gsa_LAK07318-Lease-01_signed.png",
        help="Path to image file that will be sent to the model."
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="out_file.png",
        help="Where to store the classified file with bboxes in it."
    )
    return parser.parse_args()


def process_response(response_data: Dict, orig_img, out_path: str, img_path: Path, verbose: bool = False):
    doc_status = response_data.get("document_class", "N/A")
    print(f"Document {img_path} is {doc_status}.")
    predictions = response_data.get("predictions", [])

    # Draw the bounding boxes on the image
    for prediction in predictions:
        # Print bounding box and corresponding class
        print(prediction)

        x1 = prediction.get("x1", 0)
        y1 = prediction.get("y1", 0)
        x2 = prediction.get("x2", 0)
        y2 = prediction.get("y2", 0)
        confidence = prediction.get("confidence", 0)
        class_id = prediction.get("class_id", 0)

        # Draw bbox
        rectangle_color = (0, 0, 255)
        cv2.rectangle(orig_img, (int(x1), int(y1)),
                    (int(x2), int(y2)), rectangle_color, 2)
        label = f"ID: {class_id}, Conf: {confidence:.2f}"

        # Draw bbox class info
        font_scale = 0.8
        font_thickness = 2
        text_color = (0, 0, 0)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        cv2.rectangle(orig_img, (x1, y1 - 2*text_size[1]), (x1 + text_size[0], y1), rectangle_color, -1)
        cv2.putText(orig_img, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)


    # Convert BGR to RGB for displaying in matplotlib
    image_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # Display and save
    if verbose:
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.show()

    cv2.imwrite(out_path, orig_img)
    print(f"Response image saved to {out_path}.")



def main(args):
    # Docker app endpoint
    endpoint_url = "http://localhost:8000/predict"

    # Load the image
    img_path = Path(args.img_path)
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image {img_path} not found.")

    # Optionaly resize the image (not necessary with ultralytics)
    # image = cv2.resize(image, (640, 640))

    # Convert the image to bytes for uploading
    _, image_bytes = cv2.imencode(img_path.suffix, image)
    image_data = image_bytes.tobytes()

    # Send the image to the endpoint
    response = requests.post(
        endpoint_url,
        files={"file": (img_path.name, image_data, "image/png")},
        timeout=10
    )

    # Check if the response is successful
    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        sys.exit()

    response_data = response.json()
    process_response(response_data, image, args.out_file, img_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)

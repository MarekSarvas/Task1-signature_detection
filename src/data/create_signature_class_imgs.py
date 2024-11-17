import random
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PERSONA_LIST = [
        {"ee": "Employee", "er": "Employer"},
        {"ee": "Tenant", "er": "Lessor"}
]

def parse_args():
    """ Returns arguments for named signature creation.

    Returns:
        arguments (argparse.Namespace): Program arguments
    """
    parser = argparse.ArgumentParser(description="Parse arguments for signature data creation.")

    parser.add_argument(
        "--signature_dataset",
        type=str,
        default="data/signatures_clean",
        help="Path to the dataset directory."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/persona_signatures",
        help="Path to the dataset directory."
    )
    return parser.parse_args()



def create_persona_signatures(signature_path: Path, output_dir_path: Path, persona_name: str, persona_type: str):
    """Loads a signature image and adds line with the person who should sign the contract underneath the signature.
    Create white empty image for with line and person for unsigned documents.

    Args:
        signature_path (Path): Path to image containing signature
        output_dir_path (Path): Where the created signatures will be stored
        persona_name (str): E.g. employee or employer
        persona_type (str): Type of the person on contract, e.g.: ee (for employee or tenant), er (for employer, lessor)
    """
    # Open the original image
    img = Image.open(signature_path)
    img_width, img_height = img.size

    img_empty = Image.new('RGB', (img_width, img_height), color='white')

    # Set up font and text size
    font_size = 50
    try:
        # Adjust font path if needed
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Fallback to default font if arial.ttf is unavailable
        font = ImageFont.load_default(size=font_size)

    # Calculate text size using getbbox
    text_bbox = font.getbbox(persona_name)  # (left, top, right, bottom)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Additional height and width for the dividing line
    line_height = 5
    line_padding = 40  # Padding for the dividing line to make it wider than the image

    # Create a new image with extra space for the text and the dividing line
    new_img_height = img_height + text_height + \
        line_height + 20  # 20 pixels for padding
    new_img_width = img_width + (2 * line_padding)  # Extend width for the line

    img_x = line_padding  # Center image with respect to the new width


    for sign_img, sign_type in zip([img, img_empty], ["signed", "unsigned"]):
        new_img = Image.new("RGB", (new_img_width, new_img_height), "white")
     
        # 1. add signature or no_signature into new empty image,
        # centered horizontaly
        new_img.paste(sign_img, (img_x, 0))

        # 2. add persona under the signature
        draw = ImageDraw.Draw(new_img)
        # Draw the black line
        line_y = img_height  # + 5  # 5 pixels below the original image
        # Make line cover the entire new width
        draw.line(
            [(0, line_y), (new_img_width, line_y)],
            fill="black",
            width=line_height,
        )

        # Draw the class name text below the dividing line
        text_x = (new_img_width - text_width) // 2
        text_y = line_y + line_height
        draw.text((text_x, text_y), persona_name, fill="black", font=font)

        output_path = (output_dir_path / f"{persona_type}_{sign_type}") / f"{signature_path.stem}_{persona_name}.png"
        new_img.save(output_path)


def create_signature_dirs(default_dir: Path):
    """Creates all the necessary directories to store signatures with person
    asigned.

    Args:
        default_dir (Path): Path to dir where the subdirs will be created.
    """
    sub_dirs = ["ee_signed", "ee_unsigned","er_signed", "er_unsigned"]
    for sub_dir in sub_dirs:
        sub_dir_path = default_dir / sub_dir
        sub_dir_path.mkdir(parents=True, exist_ok=True)


def main(args):
    personas_train = Path(args.out_dir)/"train"
    personas_test = Path(args.out_dir)/"test"

    create_signature_dirs(personas_train)
    create_signature_dirs(personas_test)

    # check for paths
    if not personas_train.exists():
        personas_train.mkdir(parents=True, exist_ok=True)
    if not personas_test.exists():
        personas_test.mkdir(parents=True, exist_ok=True)

    # 1. Create train split signatures
    print("Creating train split...")
    signatures = Path(args.signature_dataset) / "train"
    for signature_path in signatures.rglob("*.png"):
        persona_id = random.randint(0, len(PERSONA_LIST)-1)
        # 1.a. create for "employee"
        create_persona_signatures(signature_path,
                                  personas_train,
                                  PERSONA_LIST[persona_id]["ee"],
                                  "ee")
        # 1.b. create for "employer"
        create_persona_signatures(signature_path,
                                  personas_train,
                                  PERSONA_LIST[persona_id]["er"],
                                  "er")


    # 2. Create test split signatures
    print("Creating test split...")
    signatures = Path(args.signature_dataset) / "test"
    for signature_path in signatures.rglob("*.png"):
        persona_id = random.randint(0, len(PERSONA_LIST)-1)
        # 2.a. create for "employee"
        create_persona_signatures(signature_path,
                                  personas_test,
                                  PERSONA_LIST[persona_id]["ee"],
                                  "ee")
        # 2.b. create for "employer"
        create_persona_signatures(signature_path,
                                  personas_test,
                                  PERSONA_LIST[persona_id]["er"],
                                  "er")


if __name__ == "__main__":
    random.seed(42)
    args = parse_args()
    main(args)

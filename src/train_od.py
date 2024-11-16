import random
import argparse
from pathlib import Path

from torch.cuda import is_available as cuda_is_available
from ultralytics import YOLO


def parse_args():
    """ Returns arguments for object detection model training.

    Returns:
        arguments (argparse.Namespace): Program arguments
    """
    parser = argparse.ArgumentParser(description="Parse arguments for object detection training.")

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m.pt",
        help="Pre-trained model."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/persona_signatures",
        help="Where to store the trained models."
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/FINAL_dataset/data.yaml",
        help="Path to yaml file with yolo style training data config."
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/FINAL_dataset/data_test.yaml",
        help="Path to yaml file with yolo style test data config."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="yolo11m_100e_400img",
        help="How the checkpoint will be saved."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help=""
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=""
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help=""
    )

    return parser.parse_args()


def main(args):
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    # Load a model
    model = YOLO(args.model)

    if cuda_is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Train the model
    train_results = model.train(
        data=args.train_data,
        epochs=args.epochs,
        imgsz=640,
        device=device,
        batch=8,
        patience=5,
    )

    # Evaluate model performance on the validation set
    model.save(f"models/{args.exp_name}.pt")

    print("Evaluating the model on test set")
    metrics = model.val(data=args.test_data)

if __name__ == "__main__":
    random.seed(42)
    args = parse_args()
    main(args)

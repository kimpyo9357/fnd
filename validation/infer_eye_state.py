"""Run the shared ResNet18 eye classifier on the preserved test images."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "test")
    parser.add_argument("--weights", type=Path, default=Path(__file__).resolve().parents[1] / "weights" / "eye_closure_resnet18.pth")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "frame_predictions.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--height", type=int, default=32, help="Model input height (default: 32)")
    parser.add_argument("--width", type=int, default=32, help="Model input width (default: 32)")
    args = parser.parse_args()
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.height < 1 or args.width < 1:
        parser.error("height and width must be positive")
    if not args.weights.is_file():
        parser.error(f"Weights not found: {args.weights}")
    if not args.data.is_dir():
        parser.error(f"Test data not found: {args.data}")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(str(args.data), transform=transform)
    if dataset.class_to_idx != {"0": 0, "1": 1}:
        raise ValueError(f"Expected 0=closed, 1=open; got {dataset.class_to_idx}")
    if not dataset.samples:
        raise ValueError("No test images")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    try:
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "video_id", "frame_id", "y_true", "y_pred"])
        writer.writeheader()
        cursor = 0
        with torch.inference_mode():
            for images, labels in loader:
                preds = model(images.to(device)).argmax(1).cpu().tolist()
                for label, pred in zip(labels.tolist(), preds):
                    path, file_label = dataset.samples[cursor]
                    if label != file_label:
                        raise RuntimeError("ImageFolder order mismatch")
                    stem = Path(path).stem
                    if "_" not in stem:
                        raise ValueError(f"Cannot parse video ID from {path}")
                    video_id, frame_id = stem.rsplit("_", 1)
                    if not frame_id.isdigit():
                        raise ValueError(f"Cannot parse frame number from {path}")
                    writer.writerow({"image_path": Path(path).relative_to(args.data).as_posix(), "video_id": video_id, "frame_id": frame_id, "y_true": label, "y_pred": pred})
                    cursor += 1
    digest = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    print(json.dumps({"output": str(args.output), "n_frames": cursor, "class_mapping": dataset.class_to_idx, "weights_sha256": digest, "input_size_height_width": [args.height, args.width], "device": device}, indent=2))


if __name__ == "__main__":
    main()

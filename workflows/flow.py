"""Connect the three archived FND paths without changing the archived scripts.

Only the CSV/result-summary paths use the Python standard library. Video and
model paths use the Docker environment and a CUDA GPU for HRNet and Swin.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
LANDMARK_COLUMNS = [f"original_{i}_{axis}" for i in range(9) for axis in ("x", "y")]


def require_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_rows(path: Path) -> list[dict[str, str]]:
    with require_file(path).open(newline="") as file:
        reader = csv.DictReader(file)
        needed = {"image_name", "scale", "center_w", "center_h", *LANDMARK_COLUMNS}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError(f"Landmark CSV columns missing: {sorted(needed - set(reader.fieldnames or []))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty landmark CSV: {path}")
    return rows


def points(row: dict[str, str], *, integer: bool = True):
    result = [(float(row[f"original_{i}_x"]), float(row[f"original_{i}_y"])) for i in range(9)]
    return [(int(x), int(y)) for x, y in result] if integer else result


def require_consecutive_frames(rows):
    """A 60/300-row window must represent 60/300 adjacent video frames."""
    names = [Path(row["image_name"]).stem for row in rows]
    if all(name.isdigit() for name in names):
        numbers = [int(name) for name in names]
        for previous, current in zip(numbers, numbers[1:]):
            if current != previous + 1:
                raise ValueError(
                    f"Landmark CSV skips frames ({previous} -> {current}); "
                    "event clips require consecutive frames")


def nme(current, previous) -> float:
    eye = math.dist(previous[1], previous[2]) or 1.0
    return sum(math.dist(a, b) for a, b in zip(current, previous)) / (9 * eye)


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(path)


def manual_grades(method: str):
    """Read V/E video GT, allowing both formats present in test.csv."""
    grades = defaultdict(set)
    for source in (ROOT / "reference/output_videos.csv", ROOT / "reference/test.csv"):
        with source.open(newline="") as file:
            for columns in csv.reader(file):
                if len(columns) < 4:
                    continue
                name = columns[0] if Path(columns[0]).suffix.lower() in {".mp4", ".mov"} else columns[1]
                value = columns[3] if method == "blink" else columns[2]
                if method == "blink":
                    grade = 0 if value == "X" else int(value) if value.isdigit() else None
                else:
                    grade = int(value) - 1 if value.isdigit() else None
                if grade is not None:
                    grades[Path(name).stem].add(grade)
    return {name: next(iter(values)) for name, values in grades.items() if len(values) == 1}


def group_results(path: Path, method: str):
    data = json.loads(require_file(path).read_text())
    grades = manual_grades(method)
    grouped = defaultdict(list)
    for row in data:
        video_id = row["video_id"]
        base, sep, suffix = video_id.rpartition("_")
        origin = base if sep and suffix.isdigit() else video_id
        grouped[origin].append(row)
    results = []
    for origin, clips in sorted(grouped.items()):
        labels = {row.get("label") for row in clips}
        if len(labels) > 1:
            raise ValueError(f"Conflicting GT labels for {origin}: {labels}")
        vote = Counter(row["pred_top1"] for row in clips).most_common(1)[0][0]
        stored_gt = next(iter(labels))
        manual_gt = grades.get(origin)
        if manual_gt is not None and stored_gt is not None and manual_gt != stored_gt:
            raise ValueError(f"Stored label differs from manual GT for {origin}: {stored_gt} vs {manual_gt}")
        gt = manual_gt if manual_gt is not None else stored_gt
        results.append({"video_id": origin, "clips": len(clips), "gt": gt,
                        "gt_source": "manual_table" if manual_gt is not None else "results_json",
                        "pred": vote, "difference": abs(gt - vote) if gt is not None else None})
    return results


def print_summary(rows):
    for row in rows:
        print(f'{row["video_id"]}: GT={row["gt"]}, Pred={row["pred"]}, clips={row["clips"]}')
    with_gt = [row for row in rows if row["gt"] is not None]
    if with_gt:
        exact = sum(row["difference"] == 0 for row in with_gt)
        within_one = sum(row["difference"] <= 1 for row in with_gt)
        print(f"{len(with_gt)} videos with GT: {exact} exact, {within_one} within one grade")


def nose_angle(rows):
    selected = []
    for index, row in enumerate(rows):
        p = points(row)
        if index and nme(p, points(rows[index - 1])) > 0.01:
            continue
        def slope(a, b):
            return math.inf if a[0] == b[0] else (b[1] - a[1]) / (b[0] - a[0])
        if abs(slope(p[1], p[2])) >= 0.1 or abs(slope(p[0], p[3])) >= 0.1:
            continue
        center = ((p[0][0] + p[3][0]) / 2, (p[0][1] + p[3][1]) / 2)
        a = (p[0][0] - center[0], p[0][1] - center[1])
        b = (p[4][0] - center[0], p[4][1] - center[1])
        denominator = math.hypot(*a) * math.hypot(*b)
        if denominator == 0:
            continue
        cosine = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / denominator))
        selected.append((row["image_name"], math.degrees(math.acos(cosine))))
    if not selected:
        raise ValueError("No frames pass the nose filters")
    values = sorted(value for _, value in selected)
    def percentile(percent):
        location = (len(values) - 1) * percent
        low = math.floor(location)
        high = math.ceil(location)
        return values[low] + (values[high] - values[low]) * (location - low)
    q1, q3 = percentile(0.25), percentile(0.75)
    iqr = q3 - q1
    kept = [(name, angle) for name, angle in selected
            if q1 - 1.5 * iqr <= angle <= q3 + 1.5 * iqr]
    raw_mean = sum(value for _, value in kept) / len(kept)
    return {"source_frames": len(rows), "selected_frames": len(selected),
            "after_iqr": len(kept), "raw_mean_degrees": raw_mean,
            "deviation_from_90_after_mean": abs(raw_mean - 90),
            "selected": [{"frame": name, "angle_degrees": angle} for name, angle in kept]}


def extract_and_landmark(video: Path, work: Path) -> tuple[Path, Path]:
    """The extraction/YOLO/HRNet stage shared by all three analyses."""
    try:
        import cv2
        import pandas as pd
        import torch
    except ImportError as exc:
        raise RuntimeError("Video preprocessing needs OpenCV, pandas and PyTorch") from exc

    work.mkdir(parents=True, exist_ok=True)
    frames = work / "data"
    frames.mkdir(exist_ok=True)
    if any(frames.iterdir()):
        raise FileExistsError(f"Frame directory must be empty: {frames}")
    video = require_file(video)
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or fps <= 0:
        raise ValueError(f"Cannot read video/FPS: {video}")
    if abs(fps - 60) > 0.5:
        capture.release()
        if not shutil.which("ffmpeg"):
            raise RuntimeError(f"Input is {fps:.2f} fps; FFmpeg is required for 60 fps conversion")
        converted = work / "input_60fps.mp4"
        subprocess.run(["ffmpeg", "-i", str(video), "-r", "60", "-y", str(converted)], check=True)
        capture = cv2.VideoCapture(str(converted))
        if not capture.isOpened():
            raise ValueError(f"Cannot read converted video: {converted}")
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        if width < height:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if frame.shape[1] != 1280 or frame.shape[0] != 720:
            frame = cv2.resize(frame, (1280, 720))
        if not cv2.imwrite(str(frames / f"{count}.jpg"), frame):
            raise OSError(f"Could not write frame {count}")
        count += 1
    capture.release()
    if not count:
        raise ValueError(f"No frames extracted: {video}")
    print(f"Extracted {count} frames")

    detector = torch.hub.load(str(ROOT / "common/yolov5"), "custom",
                              path=str(WEIGHTS / "yolov5_face.pt"), source="local")
    detections = []
    names = [f"{i}.jpg" for i in range(count)]
    for start in range(0, count, 32):
        batch_names = names[start:start + 32]
        images = [cv2.cvtColor(cv2.imread(str(frames / name)), cv2.COLOR_BGR2RGB)
                  for name in batch_names]
        for name, table in zip(batch_names, detector(images, size=640).pandas().xyxy):
            if table.empty:
                continue
            box = table.iloc[0]
            xmin, ymin, xmax, ymax = [float(box[column]) for column in ("xmin", "ymin", "xmax", "ymax")]
            detections.append({"image_name": name, "scale": max(math.ceil(xmax) - math.floor(xmin),
                                                                    math.ceil(ymax) - math.floor(ymin)) / 200,
                               "center_w": (math.ceil(xmax) + math.floor(xmin)) / 2,
                               "center_h": (math.ceil(ymax) + math.floor(ymin)) / 2,
                               **{column: None for column in LANDMARK_COLUMNS}})
    if not detections:
        raise ValueError("YOLO detected no faces")
    pd.DataFrame(detections).to_csv(work / "inference.csv", index=False)

    # HRNet's original config uses ./data and ./inference.csv from the current directory.
    # Its argument parser must see no workflow CLI options.
    sys.path.insert(0, str(ROOT / "common/HRNet"))
    from tools import test as hrnet_test
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        import os
        old_cwd = Path.cwd()
        os.chdir(work)
        try:
            inferred = hrnet_test.alignment(
                cfg=str(ROOT / "common/HRNet/experiments/animal/inference.yaml"),
                model_file=str(WEIGHTS / "hrnet_landmarks.pth"))
        finally:
            os.chdir(old_cwd)
    finally:
        sys.argv = old_argv
    if len(inferred) != len(detections):
        raise ValueError("HRNet output count differs from YOLO input count")
    for row, landmark_values in zip(detections, inferred):
        row.update(zip(LANDMARK_COLUMNS, map(float, landmark_values)))
    output_csv = work / "processing_inference.csv"
    pd.DataFrame(detections).to_csv(output_csv, index=False)
    print(f"YOLO/HRNet: {len(detections)} landmark rows")
    return output_csv, frames


def prepare(args):
    if args.video:
        video = require_file(Path(args.video))
        work = Path(args.output).expanduser().resolve() if args.output else ROOT / "runs" / args.method / video.stem
        csv_path, frames = extract_and_landmark(video, work)
        return work, csv_path, frames, video.stem.removesuffix("_60fps")
    if args.landmarks_csv:
        csv_path = require_file(Path(args.landmarks_csv))
        work = Path(args.output).expanduser().resolve() if args.output else ROOT / "runs" / args.method / csv_path.stem
        frames = Path(args.frames_dir).expanduser().resolve() if args.frames_dir else None
        return work, csv_path, frames, args.video_id or csv_path.stem
    raise ValueError("Supply --video, --landmarks-csv, or --results-json")


def make_clips(clips, work: Path, base: str, size: tuple[int, int], gt: int | None):
    """Save model-ready JPG sequences and metadata. GT is optional for new videos."""
    import cv2
    dataset = work / "dataset"
    records = []
    for number, frames in enumerate(clips):
        clip_id = f"{base}_{number}"
        directory = dataset / "frames" / clip_id
        directory.mkdir(parents=True, exist_ok=True)
        count = 0
        for index, frame in enumerate(frames):
            path = directory / f"{clip_id}_img_{index:04d}.jpg"
            if not cv2.imwrite(str(path), cv2.resize(frame, size)):
                raise OSError(path)
            count += 1
        expected = 60 if size == (64, 32) else 300
        if count != expected:
            raise ValueError(f"{clip_id} has {count} frames; expected {expected}")
        records.append({"video_id": clip_id, "class_id": gt if gt is not None else 0,
                        "class_name": str(gt) if gt is not None else "unlabeled",
                        "num_frames": count, "frames_dir": str(directory)})
    save_json(dataset / "test.json", records)
    return records


def predict_clips(method: str, work: Path, records, gt: int | None):
    if not records:
        return []
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, str(ROOT / "ResNet_Trans"))
    from model import ResNetTransformerImproved
    from video_dataset import CustomVideoDataset
    if method == "blink":
        config_name = "test_custom.yaml"
        model_path = WEIGHTS / "blink_video_resnet_transformer.pth"
        image_size, frame_count = (64, 32), 60
    else:
        config_name = "ResNet-Transformer-Whisker.yaml"
        model_path = WEIGHTS / "whisker_video_resnet_transformer.pth"
        image_size, frame_count = (224, 224), 300
    config = OmegaConf.load(ROOT / "ResNet_Trans/configs" / config_name)
    config.IMG_PATH = str(work / "dataset/frames")
    config.ANNO_PATH = str(work / "dataset")
    config.MODEL.BACKBONE.PRETRAINED_DIR = str(WEIGHTS)
    require_file(WEIGHTS / "resnet18_imagenet.pth")
    require_file(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNetTransformerImproved(config).to(device)
    checkpoint = torch.load(require_file(model_path), map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = checkpoint.to(device)
    model.eval()
    dataset = CustomVideoDataset(config, mode="test", is_train=False,
                                 img_size=image_size, max_frames=frame_count)
    predictions = []
    with torch.no_grad():
        for item in dataset:
            output = model(item["frames"].unsqueeze(0).to(device))
            top = output.topk(min(3, config.NUM_CLASSES), 1).indices[0].tolist()
            predictions.append({"video_id": item["video_id"], "label": gt,
                                "pred_top1": top[0], "pred_top3": top})
    save_json(work / "predictions.json", predictions)
    return predictions


def blink_clips(rows, frames: Path, work: Path):
    import cv2
    import numpy as np
    import eye_ops as eye
    model = eye.load_model(str(WEIGHTS / "eye_closure_resnet18.pth"))
    images, closed = [], []
    for row in rows:
        frame = cv2.imread(str(frames / row["image_name"]))
        if frame is None:
            raise FileNotFoundError(frames / row["image_name"])
        p = points(row)
        try:
            left = eye.eye_crop(frame.copy(), p[0], p[1], -1)
            right = eye.eye_crop(frame.copy(), p[2], p[3], 1)
            image = np.hstack((eye.color_slicing(left), eye.color_slicing(right)))
            state = 1 - eye.open_predict(model, image)
        except (ValueError, cv2.error):
            image = images[-1] if images else np.zeros((32, 64, 3), dtype=np.uint8)
            state = 0
        images.append(image)
        closed.append(state)
    merged = eye.merge_close(np.array(closed), threshold=10)
    midpoints = eye.extract_midpoints_from_ones(merged)
    segments = eye.extract_around_midpoints(merged, midpoints, window=30)
    save_json(work / "blink_events.json", {"closed_frames": int(sum(closed)),
              "midpoints": midpoints, "segments": segments})
    return [images[start:end + 1] for start, end in segments if end - start + 1 == 60]


def whisker_clips(rows, frames: Path, work: Path, threshold: float):
    import cv2
    from mmdet.apis import init_detector
    import whisker_ops as whisker
    model = init_detector(str(ROOT / "workflows/htc_swin_fpn_dyhead_whisker.py"),
                          str(WEIGHTS / "whisker_swin_htc.pth"), device="cuda:0")
    masks, left_angles, right_angles = [], [], []
    mask_dir = work / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    previous_points = None
    for row in rows:
        p = points(row)
        valid = previous_points is None or nme(p, previous_points) <= 0.1
        previous_points = p
        if valid:
            frame = cv2.imread(str(frames / row["image_name"]))
            if frame is None:
                raise FileNotFoundError(frames / row["image_name"])
            eye_center = ((p[1][0] + p[2][0]) // 2, (p[1][1] + p[2][1]) // 2)
            nose = p[4]
            angle = whisker.angle_between_points(*eye_center, *nose)
            height, width = frame.shape[:2]
            aligned, transformed = whisker.transform_mask_and_points(
                frame, angle - 90, width // 2 - nose[0], height // 2 - nose[1],
                points=[eye_center, nose])
            mask, candidates, _ = whisker.segment(aligned, model,
                                                   (transformed[0][1], transformed[1][1]))
            left = [item["PCA"][0] for item in candidates
                    if (item["bbox"][0] + item["bbox"][2]) / 2 <= width / 2]
            right = [item["PCA"][0] for item in candidates
                     if (item["bbox"][0] + item["bbox"][2]) / 2 > width / 2]
            mask_path = mask_dir / f"{len(masks):06d}.png"
            if not cv2.imwrite(str(mask_path), mask):
                raise OSError(mask_path)
            masks.append(mask_path)
            left_angles.append(sum(left) / len(left) if left else (left_angles[-1] if left_angles else 0))
            right_angles.append(sum(right) / len(right) if right else (right_angles[-1] if right_angles else 0))
        else:
            masks.append(masks[-1])
            left_angles.append(left_angles[-1])
            right_angles.append(right_angles[-1])
    changes = [max(abs(left_angles[i] - left_angles[i - 1]),
                   abs(right_angles[i] - right_angles[i - 1])) if i else 0
               for i in range(len(rows))]
    active = [i for i, value in enumerate(changes) if value > threshold]
    groups = []
    for index in active:
        if groups and index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    peaks = [max(group, key=lambda i: changes[i]) for group in groups]
    windows = [(peak - 150, peak + 150) for peak in peaks
               if peak >= 150 and peak + 150 <= len(rows)]
    save_json(work / "whisker_events.json", {"threshold_radians": threshold,
              "active_frames": active, "peaks": peaks, "windows": windows})
    def clip_frames(start, end):
        for mask_path in masks[start:end]:
            frame = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                raise FileNotFoundError(mask_path)
            yield cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return (clip_frames(start, end) for start, end in windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("nose", "blink", "whisker"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="60 fps input video; extracts frames and landmarks")
    source.add_argument("--landmarks-csv", help="Reuse an existing HRNet landmark CSV")
    source.add_argument("--results-json", help="Summarize archived clip GT/Pred JSON (blink/whisker)")
    parser.add_argument("--frames-dir", help="Frame directory matching --landmarks-csv image_name")
    parser.add_argument("--video-id", help="Original video ID when using a CSV")
    parser.add_argument("--output", help="New output directory; defaults to runs/<method>/<input>")
    parser.add_argument("--gt", help="Optional video GT: EX/E1-E5 for blink, V1-V5 for whisker")
    parser.add_argument("--threshold", type=float, default=0.2, help="Whisker angular change in radians")
    args = parser.parse_args()
    if args.results_json:
        if args.method == "nose":
            parser.error("Nose uses --landmarks-csv, not --results-json")
        grouped = group_results(Path(args.results_json), args.method)
        print_summary(grouped)
        if args.output:
            save_json(Path(args.output) / "video_comparison.json", grouped)
        return
    work, csv_path, frames, base = prepare(args)
    rows = load_rows(csv_path)
    save_json(work / "source.json", {"method": args.method,
              "input_video": str(Path(args.video).expanduser().resolve()) if args.video else None,
              "landmarks_csv": str(csv_path),
              "frames_dir": str(frames) if frames else None,
              "video_id": base, "gt_argument": args.gt})
    if args.method == "nose":
        result = nose_angle(rows)
        save_json(work / "nose_result.json", result)
        print(f'Raw mean: {result["raw_mean_degrees"]:.6f}°; |mean - 90°|: {result["deviation_from_90_after_mean"]:.6f}°')
        return
    if frames is None or not frames.is_dir():
        parser.error("Blink/whisker needs --frames-dir with the landmark CSV, or --video")
    require_consecutive_frames(rows)
    manual_gt = manual_grades(args.method).get(base)
    if args.method == "blink":
        if args.gt is not None and args.gt not in {"EX", *(f"E{i}" for i in range(1, 6))}:
            parser.error("Blink GT must be EX or E1-E5")
        gt = manual_gt if args.gt is None else (0 if args.gt == "EX" else int(args.gt[1:]))
        clips = blink_clips(rows, frames, work)
        size = (64, 32)
    else:
        if args.gt is not None and args.gt not in {f"V{i}" for i in range(1, 6)}:
            parser.error("Whisker GT must be V1-V5")
        gt = manual_gt if args.gt is None else int(args.gt[1:]) - 1
        clips = whisker_clips(rows, frames, work, args.threshold)
        size = (128, 72)
    records = make_clips(clips, work, base, size, gt)
    if not records:
        print("No complete event clips; no grade prediction produced")
        return
    predictions = predict_clips(args.method, work, records, gt)
    comparison = group_results(work / "predictions.json", args.method)
    save_json(work / "video_comparison.json", comparison)
    print_summary(comparison)


if __name__ == "__main__":
    main()

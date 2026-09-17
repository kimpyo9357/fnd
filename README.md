# FND Video Analysis

Analyzes **nose deviation**, **eye blinking**, and **whisker movement** in rat face videos.

| Analysis | Camera view | Script |
| --- | --- | --- |
| Nose deviation | Top view | `run_nose.sh` |
| Eye blinking | Front view | `run_blink.sh` |
| Whisker movement | Top view | `run_whisker.sh` |

## Requirements

- Linux, Docker Engine, an NVIDIA driver, and NVIDIA Container Toolkit.
- A CUDA-capable NVIDIA GPU. The Docker build compiles MMCV operators for compute capability **8.6** by default.
- About **27 GB** of storage for the Docker image, plus space for extracted frames in `runs/`.
- MP4 input videos and seven model weights, both distributed separately from this repository.

## 1. Get the code and weights

Clone this repository and run every command below from its root.

Download the [model weights from Google Drive](https://drive.google.com/file/d/1RjJhWc36SdN0lTQBVfK-3u3wppauZLYj/view?usp=drive_link) and place the seven files in `weights/` under these exact names. If Google Drive asks for access, request it from the repository owner.

```text
weights/
├── yolov5_face.pt
├── hrnet_landmarks.pth
├── eye_closure_resnet18.pth
├── whisker_swin_htc.pth
├── resnet18_imagenet.pth
├── blink_video_resnet_transformer.pth
└── whisker_video_resnet_transformer.pth
```

Verify them. All seven entries should report `OK`.

```bash
sha256sum -c weights/SHA256SUMS
```

## 2. Get the Docker image

```bash
export FND_DOCKER_IMAGE=ghcr.io/kimpyo9357/fnd:release
docker pull "$FND_DOCKER_IMAGE"
./docker.sh nvidia-smi
```

Keep `FND_DOCKER_IMAGE` set for the commands below. `docker.sh` mounts this repository at `/workspace` inside the container.

To build locally instead, leave the variable unset. `docker.sh` then defaults to the local `fnd_flow:release` image.

```bash
unset FND_DOCKER_IMAGE
docker build -f Dockerfile.release -t fnd_flow:release .
./docker.sh nvidia-smi
```

For another GPU architecture, pass its compute capability when building. For example, for 8.9:

```bash
docker build -f Dockerfile.release --build-arg CUDA_ARCH_LIST=8.9 -t fnd_flow:release .
```

## 3. Run the analyses

Put the videos in one directory and set `FND_SOURCE_ROOT` to its **absolute path**. That directory is mounted read-only at `/source`. If the variable is unset, no video directory is mounted and `/source` paths will not resolve.

```bash
export FND_SOURCE_ROOT="$HOME/fnd-videos"

# Front view
./docker.sh ./run_blink.sh --video /source/front.mp4

# Top view
./docker.sh ./run_nose.sh --video /source/top.mp4
./docker.sh ./run_whisker.sh --video /source/top.mp4
```

Inputs that are not 60 fps are converted with FFmpeg before frame extraction. The whisker angle-change threshold defaults to **0.2 rad**; change it with `--threshold`.

If you run the nose analysis first, the whisker analysis can reuse its frames and landmarks. Use this **instead of** the `run_whisker.sh --video` command above:

```bash
./docker.sh ./run_whisker.sh \
  --landmarks-csv runs/nose/top/processing_inference.csv \
  --frames-dir runs/nose/top/data \
  --video-id top \
  --output runs/whisker/top_reused
```

To supply a ground-truth grade, add `--gt` with `EX` or `E1`–`E5` for eyes, or `V1`–`V5` for whiskers. Predictions also work without it, and a GT matching the input filename in `reference/` is read automatically.

```bash
./docker.sh ./run_blink.sh --video /source/front.mp4 --gt E3
./docker.sh ./run_whisker.sh --video /source/top.mp4 --gt V5
```

## Outputs

Results go to `runs/<analysis>/<video filename>/`. To rerun the same video, pass `--output runs/new_name` with a fresh directory — preprocessing will not restart in a directory that already holds extracted frames.

| File | Contents |
| --- | --- |
| `processing_inference.csv` | Face detections and landmark coordinates |
| `nose_result.json` | Nose angles for selected frames and their mean |
| `blink_events.json` / `whisker_events.json` | Event intervals |
| `dataset/frames/`, `dataset/test.json` | Event clips and metadata for the classifier |
| `predictions.json` | Predicted grade for each clip |
| `video_comparison.json` | Video-level Pred and GT; GT is `null` when unavailable |

Eye and whisker analysis need detections and landmarks on consecutive frames. When no complete event interval is found, no grade prediction is produced.

## License

Distributed under the **GNU Affero General Public License v3.0**; see [`LICENSE`](LICENSE). The AGPL applies because this project bundles YOLOv5, which Ultralytics releases under AGPL-3.0.

Bundled third-party components keep their original license files, which must not be removed:

| Component | License | File |
| --- | --- | --- |
| YOLOv5 (Ultralytics) | AGPL-3.0 | [`common/yolov5/LICENSE`](common/yolov5/LICENSE) |
| HRNet (Microsoft) | MIT | [`common/HRNet/LICENCE`](common/HRNet/LICENCE) |

## Acknowledgment

Codex and Claude were used to organize this code and to write this documentation.

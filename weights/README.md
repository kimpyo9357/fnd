# Model weights

This directory holds the seven weights needed by the three analysis pipelines. `.gitignore` excludes the binary files from the GitHub repository. Download them from [Google Drive](https://drive.google.com/file/d/1RjJhWc36SdN0lTQBVfK-3u3wppauZLYj/view?usp=drive_link), place them here under the exact filenames below, and verify them from the repository root with `sha256sum -c weights/SHA256SUMS`. If Google Drive asks for access, request it from the repository owner.

| File | Purpose |
| --- | --- |
| `yolov5_face.pt` | Shared face detection |
| `hrnet_landmarks.pth` | Shared nine-point landmark estimation |
| `eye_closure_resnet18.pth` | Frame-level eye closure detection |
| `whisker_swin_htc.pth` | Swin-based whisker segmentation |
| `resnet18_imagenet.pth` | Backbone initialization for the eye and whisker video classifiers |
| `blink_video_resnet_transformer.pth` | Grade prediction from 60-frame eye clips |
| `whisker_video_resnet_transformer.pth` | Grade prediction from 300-frame whisker clips |

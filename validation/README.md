# Additional validation and confidence intervals

This folder is separate from the three video-analysis workflows. It reproduces
the additional confidence-interval calculations from fixed predictions. It can
also rerun eye-state inference when the prepared test images and the shared eye
weight are available.

## Inputs and provenance

Download [Extended Data 1 validation data.zip](https://drive.google.com/file/d/1aIvjEN2CbzxYFWCtTTlAiFxKoBJBdISI/view?usp=sharing)
separately and extract it beside `Extended Data 1.zip`, so the files land
under `fnd/validation/data/`.

Data archive SHA-256:
`d21ca8c6f16a6b3688cf4793645ea3c047474e0cc7eb6c4335e50529acc5f167`

The data archive contains:

- `data/test/0/` and `data/test/1/`: 4,489 previously prepared bilateral-eye
  test images from two source videos. Class `0` is closed and class `1` is open.
- `data/test_manifest.csv`: image path, source video, frame number, and manual
  eye-state label. The two videos contribute 2,056 and 2,433 images.
- `data/predictions/longtail_predictions.csv`: saved predictions from the
  shared eye-state classifier, allowing CI reproduction without PyTorch.
- `data/previous_ci/`: saved blink and whisker clip predictions and the nose
  raw-angle comparison workbook, used by `analyze_other_ci.py`.

The saved images are already cropped and concatenated eye regions. In the
original processing path, video frames were passed through face detection and
landmark localization; the two eyes were aligned/cropped, color adjusted, and
placed side by side. Inference loads the 64-by-32-pixel saved images as RGB,
resizes them to 32 by 32, applies ImageNet normalization, and uses a ResNet18
binary classifier. The original videos and the complete manual labeling and
crop provenance are not in this release. Consequently, the exact 4,489-image
test set cannot be regenerated from raw videos using this package alone; the
separate data archive preserves the actual evaluation inputs.

The primary model is the existing `weights/eye_closure_resnet18.pth`, distributed
with the seven main workflow weights. Its SHA-256 is
`d3ec5050ca4dc1abb97fa8f29d09c1abef261172842779b553991ce4a0996335`.
No second copy of this weight is included here. The earlier balance-model
comparison is outside this reproducibility package because its weight is not
one of the seven main workflow weights.

## Reproduce the calculations

Run these commands from the extracted `fnd/` directory. Python 3.9 or newer
is required. The CI script uses only the standard library; the other-results
script additionally needs `openpyxl`.

```bash
# Recalculate eye-state metrics and the video-cluster bootstrap from saved predictions.
python validation/analyze_eye_state.py

# Recalculate blink/whisker Wilson intervals and nose animal-cluster intervals.
python validation/analyze_other_ci.py
```

Both scripts default to 2,000 bootstrap resamples with random seed 20261001.
The eye-state calculation resamples the two source videos with replacement and
retains all frames belonging to each selected video. The nose calculation
resamples four animals with replacement and retains their paired angle
measurements. Blink and whisker grade agreement use Wilson intervals, not a
bootstrap. Outputs are written under `validation/results/`; the originally
saved JSONs are under `validation/expected/` for comparison. The JSON `source`
path may differ between machines, but numerical results should match.

To regenerate the prediction CSV from the preserved test images, install a
compatible `torch`, `torchvision`, and `Pillow`, and place the seven main
weights in `weights/` as described in the project README. Then run:

```bash
python validation/infer_eye_state.py --output validation/results/recomputed_predictions.csv
python validation/analyze_eye_state.py \
  --predictions validation/results/recomputed_predictions.csv \
  --output validation/results/recomputed_eye_state_ci.json
```

`infer_eye_state.py` uses `weights/eye_closure_resnet18.pth` by default. It
prints that file's SHA-256 and the image count. The eye-state CI is exploratory
because it resamples only two source videos; the nose CI likewise has only
four animal clusters. These intervals do not establish performance on new
animals.

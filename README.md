# Snack Launcher

Webcam mouth detector prototype for a desk robot that aims a snack
launcher.

## Setup

```bash
uv sync
```

## Run the mouth detector

Show the live camera overlay:

```bash
uv run snack-mouth-detector
```

The first run downloads the OpenCV LBF 68-point face-landmark model into
`~/.cache/snack-launcher/`.

Stream one JSON payload per frame:

```bash
uv run snack-mouth-detector --json --no-window
```

Useful options:

- `--camera 1` selects a different webcam index.
- `--no-mirror` disables the default selfie-style mirror.
- `--max-frames 30` exits after a fixed number of frames for smoke testing.
- `--model /path/to/lbfmodel.yaml` uses an explicit model file.
- `--mouth-wide-open-ratio 0.7` requires a larger opening before the mouth is
  considered wide open (default: `0.6`).
- `--assumed-ipd-mm 63` sets the adult interpupillary-distance assumption used
  for distance estimation.
- `--camera-preset macbook-pro-13-2022` selects the current default webcam
  calibration. It approximates the 720p FaceTime HD camera as a 60° horizontal
  field of view (about 1,109 px focal length at 1280-pixel width).
- `--camera-horizontal-fov-deg 70` overrides a preset for another webcam.

Each detection includes:

- mouth center in pixels
- mouth openness ratio
- normalized `aim_offset` from the center of the frame
- `mouth_wide_open`, using the configured wide-open ratio threshold
- `distance_estimate_m`, a coarse single-camera estimate of camera distance

`distance_estimate_m` is based on an assumed adult interpupillary distance, eye
center positions (a pupil-position proxy), and the camera's configured field of
view. It is affected by individual anatomy, head rotation, lens distortion, and
field-of-view accuracy; do not use it as a precision or safety measurement.

The OpenCV detector measures mouth height between averaged outer-lip landmarks.
This avoids the LBF model's inner-lip points collapsing toward the teeth on a
very wide-open mouth.

## Safety notes

Treat this like a robotics project, not a toy. Use soft, low-mass projectiles,
keep launch energy very low, add a physical arming switch, and test against a
paper target before aiming anywhere near a person.

## Development

```bash
uv run pytest
uv build
```

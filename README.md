# Snack Launcher

Webcam mouth detector prototype for a desk robot that aims a snack
launcher.

## Setup

```bash
uv sync
```

The project intentionally pins MediaPipe `0.10.18`, Python below `3.13`, NumPy
below `2`, and OpenCV Contrib `4.10.0.84`. This is the newest tested stack in
this project that provides prebuilt wheels for both macOS and Raspberry Pi
Linux ARM64; upgrading MediaPipe without checking ARM64 wheel availability can
make installation on the Pi fail.

## Run the mouth detector

Show the live camera overlay:

```bash
uv run snack-mouth-detector
```

The first run downloads the MediaPipe `face_landmarker.task` model bundle into
`~/.cache/snack-launcher/`.

Stream one JSON payload per frame:

```bash
uv run snack-mouth-detector --json --no-window
```

Useful options:

- `--camera 1` selects a different webcam index.
- `--no-mirror` disables the default selfie-style mirror.
- `--max-frames 30` exits after a fixed number of frames for smoke testing.
- `--model /path/to/face_landmarker.task` uses an explicit model file.
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

`distance_estimate_m` is based on an assumed adult interpupillary distance and
the camera's configured field of view. It is affected by individual anatomy,
head rotation, lens distortion, and field-of-view accuracy; do not use it as a
precision or safety measurement.

## Safety notes

Treat this like a robotics project, not a toy. Use soft, low-mass projectiles,
keep launch energy very low, add a physical arming switch, and test against a
paper target before aiming anywhere near a person.

## Development

```bash
uv run pytest
uv build
```

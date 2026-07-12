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
- `--mouth-wide-open-ratio 0.2` requires a larger opening before the mouth is
  considered wide open (default: `0.15`).

Each detection includes:

- mouth center in pixels
- mouth openness ratio
- normalized `aim_offset` from the center of the frame
- `mouth_wide_open`, using the configured wide-open ratio threshold

## Safety notes

Treat this like a robotics project, not a toy. Use soft, low-mass projectiles,
keep launch energy very low, add a physical arming switch, and test against a
paper target before aiming anywhere near a person.

## Development

```bash
uv run pytest
uv build
```

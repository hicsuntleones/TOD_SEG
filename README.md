# FAITH – SAM2 Turntable Segmentation Pipeline

Automatic and interactive batch segmentation pipeline built on top of
[SAM2](https://github.com/facebookresearch/sam2) (Segment Anything Model 2),
developed for turntable object photography (72 views, 5° steps) in the
context of the Horizon Europe **FAITH** project (GA #101135932), WP5 –
Transportation Large Scale Pilot.

Given a batch of turntable captures, the pipeline computes (or lets you pick
interactively) the positive/negative points and bounding box needed to
prompt SAM2's video predictor, then propagates the segmentation mask across
all views of the sequence, with quality-control reporting (area
deviation/jump outliers, loop-closure IoU at the midpoint).

## Scripts

- **`auto_prompt.py`** – computes the automatic point prompts for an object.
  Uses a three-phase hybrid strategy: a small reserve of unseeded
  farthest-point-sampling (FPS) on the raw foreground mask (to catch thin
  protruding features such as straps/cables), zone-color points (k-means in
  Lab color space + connected components) on the cleaned object mask, and a
  final FPS completion pass seeded by all previously chosen points.
- **`auto_segment.py`** – orchestrates fully automatic segmentation of a
  single view or an entire batch (recursively finds `color` frame
  directories), computing prompts via `auto_prompt.py` and propagating them
  with `sam2_video_propagate.py`. Produces per-object overlays and a batch
  QC summary (JSON/CSV).
- **`manual_segment.py`** – same batch/single-view orchestration as
  `auto_segment.py`, but lets you pick the points and box interactively for
  each object (via the logic in `pick_points.py`) before launching SAM2's
  video predictor with those exact prompts.
- **`pick_points.py`** – standalone interactive point/box picker: click to
  add positive points, right-click (or similar) for negative points, and
  hold Shift while dragging to draw the bounding box live. Prints the
  resulting `--point` / `--neg_point` / `--box` flags on a single line.
- **`sam2_video_propagate.py`** – thin wrapper around
  `SAM2VideoPredictor`: takes the prompts (points + labels + optional box),
  runs `propagate_in_video()`, and returns/writes a QC report (frame areas,
  relative deviation from the mean, frame-to-frame jumps, outlier frames,
  loop-closure IoU).

## Setup

See [`GUIDA_setup_sam2_batch.md`](GUIDA_setup_sam2_batch.md) (in Italian) for
the full Windows setup walkthrough (Python venv, PyTorch/CUDA build, SAM2
install, checkpoints). In short:

```
python -m venv sam2-venv
sam2-venv\Scripts\activate
pip install torch ...   # see guide for the right CUDA build
pip install -e <path-to-facebookresearch/sam2-checkout>
pip install opencv-python matplotlib pillow scikit-learn
```

Then, for a whole batch:

```
python auto_segment.py --batch_root <path> --out_root <path> --checkpoint <ckpt> --config <cfg>
```

or interactively:

```
python manual_segment.py --batch_root <path> --out_root <path> --checkpoint <ckpt> --config <cfg>
```

## License / provenance

These scripts are original work built on top of SAM2 (Apache 2.0,
Meta/FAIR); they do not vendor SAM2's own source — install it separately
from the [official repository](https://github.com/facebookresearch/sam2).

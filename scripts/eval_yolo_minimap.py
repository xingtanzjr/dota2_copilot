r"""Validate the trained YOLO minimap detector on real minimap crops.

Runs :class:`YoloMinimapDetector` (the same inference path the app uses for
``display_mode: yolo``) over one or more real minimap images, draws the
detected boxes + ``hero__team`` labels, and writes annotated PNGs.

Run on a machine with the ``yolo`` extra installed (ultralytics + torch)::

    pip install -e ".[yolo]"

    # default: annotate the two bundled real frames
    python scripts/eval_yolo_minimap.py

    # custom images / folder, lower threshold, CPU
    python scripts/eval_yolo_minimap.py --images path\to\crop.png --conf 0.2 --device cpu

Outputs go to ``--out-dir`` (default ``runs/eval_minimap``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Make ``src`` importable when run from the repo root without an install.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dota2_copilot.capture.yolo_detect import YoloMinimapDetector  # noqa: E402
from dota2_copilot.config import YoloDetectConfig  # noqa: E402
from dota2_copilot.types import Team  # noqa: E402

# BGR colors: ally green, enemy red (matches the app's team convention).
_ALLY_BGR = (0, 200, 0)
_ENEMY_BGR = (0, 0, 230)

# Default real frames bundled in the repo (contain actual heroes).
_DEFAULT_IMAGES = [
    REPO_ROOT / "snapshots" / "debug_1781330566" / "minimap_raw.png",
    REPO_ROOT / "assets" / "minimap_bg" / "minimap_sample.png",
]


def _collect_images(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if path.is_dir():
            out.extend(sorted(q for q in path.iterdir() if q.suffix.lower() in {".png", ".jpg", ".jpeg"}))
        elif path.exists():
            out.append(path)
        else:
            print(f"[eval] WARNING: image not found: {path}")
    return out


def _annotate(img: np.ndarray, enemies, allies) -> np.ndarray:
    canvas = img.copy()
    for blob in list(allies) + list(enemies):
        x, y, w, h = blob.bbox
        color = _ALLY_BGR if blob.team == Team.ALLY else _ENEMY_BGR
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
        label = f"{blob.hero_id or '?'} {blob.score:.2f}"
        ty = y - 3 if y - 3 > 6 else y + h + 10
        cv2.putText(canvas, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="models/minimap_yolo.pt", help="path to trained .pt")
    ap.add_argument("--images", nargs="*", default=None, help="image files or folders (default: bundled real frames)")
    ap.add_argument("--conf", type=float, default=0.25, help="min confidence")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="inference image size")
    ap.add_argument("--device", default="", help='"" auto | "cpu" | "0" for cuda:0')
    ap.add_argument("--max-det", type=int, default=30, help="max detections per frame")
    ap.add_argument("--out-dir", default="runs/eval_minimap", help="where to write annotated PNGs")
    args = ap.parse_args()

    image_args = args.images if args.images else [str(p) for p in _DEFAULT_IMAGES]
    images = _collect_images(image_args)
    if not images:
        print("[eval] no images to process")
        sys.exit(1)

    cfg = YoloDetectConfig(
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
    )
    detector = YoloMinimapDetector(cfg)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[eval] WARNING: cannot read {path}")
            continue
        enemies, allies = detector.detect(img)
        annotated = _annotate(img, enemies, allies)
        out_path = out_dir / f"{path.stem}_pred.png"
        cv2.imwrite(str(out_path), annotated)

        print(f"\n=== {path.name}  ({img.shape[1]}x{img.shape[0]}) ===")
        print(f"  allies ({len(allies)}): " + ", ".join(f"{b.hero_id}[{b.score:.2f}]" for b in allies))
        print(f"  enemies({len(enemies)}): " + ", ".join(f"{b.hero_id}[{b.score:.2f}]" for b in enemies))
        print(f"  -> {out_path}")

    print(f"\n[eval] done. Annotated images in {out_dir}")


if __name__ == "__main__":
    main()

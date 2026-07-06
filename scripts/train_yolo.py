r"""Train the YOLO minimap hero detector (254 classes = 127 heroes x 2 teams).

This is a thin, reproducible wrapper around Ultralytics YOLOv8. It is meant to
be run **on the Windows GPU machine** after copying the whole project there
(the dataset lives in ``datasets/minimap_yolo/``).

What it does
------------
1. Rewrites ``<dataset>/data.yaml`` so its ``path:`` points at the dataset on
   *this* machine (the committed data.yaml has a Linux path -- this fixes it on
   Windows automatically).
2. Trains a YOLOv8 detector with augmentation tuned for a single, centred
   minimap (mosaic/rotation off -- inference always sees one upright minimap).
3. Copies the best checkpoint to ``models/minimap_yolo.pt`` for the app to load.

Quick start (Windows PowerShell)
--------------------------------
    # 0) create + activate a venv with Python 3.11+
    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1

    # 1) install a CUDA-enabled PyTorch FIRST (pick the CUDA that matches your
    #    driver; cu121 works for most recent NVIDIA drivers)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    # 2) install ultralytics (the project's optional 'yolo' extra)
    pip install ultralytics

    # 3) sanity-check the GPU is visible
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

    # 4) train (defaults: yolov8s, 100 epochs, imgsz 640, auto batch, GPU 0)
    python scripts/train_yolo.py

    # ...or a heavier/lighter model, more epochs, explicit batch:
    python scripts/train_yolo.py --model yolov8m.pt --epochs 150 --batch 32

Outputs
-------
* ``runs/minimap/<name>/weights/best.pt``  (Ultralytics run dir)
* ``models/minimap_yolo.pt``               (copied best weights for the app)

Monitor training in ``runs/minimap/<name>/`` (results.png, confusion_matrix.png,
val batch previews). Resume an interrupted run with ``--resume``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dota2_copilot.training.classes import write_data_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "datasets" / "minimap_yolo",
                    help="dataset root (contains images/ labels/ data.yaml)")
    ap.add_argument("--model", type=str, default="yolov8s.pt",
                    help="base model/weights: yolov8n/s/m/l .pt (downloaded on first use)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="training image size; upsamples the ~375px minimap so small icons are learnable")
    ap.add_argument("--batch", type=int, default=-1,
                    help="batch size; -1 lets Ultralytics auto-pick from GPU memory")
    ap.add_argument("--device", type=str, default="0",
                    help="CUDA device index (e.g. 0), '0,1' for multi-GPU, or 'cpu'")
    ap.add_argument("--workers", type=int, default=8, help="dataloader workers")
    ap.add_argument("--cache", type=str, default="ram", choices=["ram", "disk", "none"],
                    help="cache images for faster epochs; 'ram' fits this small dataset (~4GB)")
    ap.add_argument("--project", type=Path, default=REPO_ROOT / "runs" / "minimap")
    ap.add_argument("--name", type=str, default="yolov8s_254")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=30,
                    help="early-stop patience (epochs without val improvement)")
    ap.add_argument("--resume", action="store_true", help="resume the last run of --name")
    # augmentation knobs (defaults tuned for a single, upright, centred minimap)
    ap.add_argument("--mosaic", type=float, default=0.0,
                    help="mosaic prob; 0 = off (inference is always ONE minimap, not a 2x2 collage)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "minimap_yolo.pt",
                    help="where to copy the best checkpoint after training")
    ap.add_argument("--export-onnx", action="store_true",
                    help="also export best.pt to ONNX next to it")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Windows spawns DataLoader workers as fresh processes; they cannot share the
    # main process's in-RAM image cache, which deadlocks at the first epoch. Force
    # single-process loading in that combo (RAM-cached small images stay fast).
    import platform
    if platform.system() == "Windows" and args.cache == "ram" and args.workers != 0:
        print(f"[train_yolo] Windows + cache='ram': forcing --workers 0 "
              f"(was {args.workers}) to avoid a DataLoader spawn deadlock.")
        args.workers = 0

    data_root = args.data.resolve()
    if not (data_root / "images" / "train").is_dir():
        raise SystemExit(
            f"Dataset not found at {data_root}. Copy datasets/minimap_yolo/ here, "
            f"or pass --data <path>."
        )

    # Make data.yaml point at THIS machine's dataset location (fixes the Linux
    # path baked in when the dataset was generated).
    data_yaml = write_data_yaml(data_root)
    print(f"[train_yolo] data.yaml -> {data_yaml} (path: {data_root})")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "ultralytics is not installed. Install it (and a CUDA torch) first:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
            "  pip install ultralytics"
        ) from e

    model = YOLO(args.model)
    print(f"[train_yolo] training {args.model} on 254 classes "
          f"(epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, device={args.device})")

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=(False if args.cache == "none" else args.cache),
        project=str(args.project),
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        resume=args.resume,
        # --- augmentation: keep the minimap upright & singular ---
        mosaic=args.mosaic,   # 0.0: no 2x2 collage (unlike any real frame)
        mixup=0.0,
        degrees=0.0,          # never rotate (bases stay in their real corners)
        fliplr=0.5,           # safe: team is by arc COLOUR, not orientation
        flipud=0.0,
        translate=0.05,
        scale=0.2,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.4,
    )

    # Locate best.pt from the finished run and copy it for the app.
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if best.is_file():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, args.out)
        print(f"[train_yolo] best weights -> {args.out}")
        if args.export_onnx:
            YOLO(str(best)).export(format="onnx")
            print("[train_yolo] exported ONNX next to best.pt")
    else:
        print(f"[train_yolo] WARNING: best.pt not found under {save_dir/'weights'}")

    print("[train_yolo] done. Run dir:", save_dir)


if __name__ == "__main__":
    main()

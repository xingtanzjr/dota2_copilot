"""Generate a synthetic YOLO dataset for minimap hero detection.

Composites hero portraits (assets/minimap/*_32.png) with team borders onto real
minimap backgrounds, adds unlabeled distractors (buildings/creeps/couriers), and
writes an Ultralytics-format dataset:

    datasets/minimap_yolo/
        images/train/*.png   labels/train/*.txt
        images/val/*.png     labels/val/*.txt
        data.yaml

Examples
--------
    python scripts/gen_synth_dataset.py --train 4000 --val 400
    python scripts/gen_synth_dataset.py --backgrounds path/to/bg_dir --seed 7
    python scripts/gen_synth_dataset.py --preview           # also dump a sample grid

Backgrounds
-----------
By default we harvest any real minimap crops we can find:
    snapshots/**/minimap_raw.png
    recordings/**/frames/*.png
Point --backgrounds at a folder of *clean* minimap crops (ideally without heroes)
for best quality; the more varied the terrain lighting, the better.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dota2_copilot.training.classes import load_hero_shorts, write_data_yaml  # noqa: E402
from dota2_copilot.training.synth import (  # noqa: E402
    PortraitBank,
    SynthConfig,
    load_backgrounds,
    synth_frame,
)


def _is_clean_bg(p: Path) -> bool:
    # *sample* crops contain real (unlabelled) heroes -> never use as background.
    return "sample" not in p.stem.lower()


def collect_default_backgrounds() -> list[Path]:
    """Prefer clean, unit-free backgrounds in assets/minimap_bg/.

    Only if that folder is empty do we fall back to raw captures (which contain
    real heroes -> unlabeled positives -> label noise), with a warning.
    """
    clean = sorted(p for p in (REPO_ROOT / "assets" / "minimap_bg").glob("*.png")
                   if _is_clean_bg(p))
    if clean:
        return clean
    print("WARNING: no clean backgrounds in assets/minimap_bg/; falling back to "
          "raw captures which may contain unlabeled heroes.")
    paths: list[Path] = []
    paths += sorted((REPO_ROOT / "snapshots").glob("**/minimap_raw.png"))
    paths += sorted((REPO_ROOT / "recordings").glob("**/frames/*.png"))
    return paths


def _write_split(
    split: str,
    count: int,
    out_root: Path,
    bank: PortraitBank,
    backgrounds: list,
    shorts: list[str],
    cfg: SynthConfig,
    rng: random.Random,
) -> None:
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        img, labels = synth_frame(bank, backgrounds, shorts, rng, cfg)
        stem = f"{split}_{i:06d}"
        cv2.imwrite(str(img_dir / f"{stem}.png"), img)
        lines = [
            f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            for (cid, cx, cy, w, h) in labels
        ]
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if (i + 1) % 500 == 0:
            print(f"  [{split}] {i + 1}/{count}")


def _preview_grid(out_root: Path, bank, backgrounds, shorts, cfg, rng, n: int = 9) -> Path:
    import numpy as np

    tiles = []
    for _ in range(n):
        img, labels = synth_frame(bank, backgrounds, shorts, rng, cfg)
        H, W = img.shape[:2]
        vis = img.copy()
        for cid, cx, cy, w, h in labels:
            x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
            team_enemy = cid % 2 == 1
            col = (0, 0, 255) if team_enemy else (0, 220, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 1)
        vis = cv2.resize(vis, (360, 360))
        tiles.append(vis)
    rows = [np.hstack(tiles[r * 3:r * 3 + 3]) for r in range(3)]
    grid = np.vstack(rows)
    out = out_root / "preview_grid.png"
    cv2.imwrite(str(out), grid)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=4000, help="number of training frames")
    ap.add_argument("--val", type=int, default=400, help="number of validation frames")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "datasets" / "minimap_yolo",
                    help="output dataset root")
    ap.add_argument("--backgrounds", type=Path, default=None,
                    help="directory of background minimap crops (recurses for images)")
    ap.add_argument("--template-dir", type=Path, default=REPO_ROOT / "assets" / "minimap")
    ap.add_argument("--base-size", type=int, default=32, help="portrait template size to load")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--team-size", type=int, default=5, help="heroes per team (5v5)")
    ap.add_argument("--ally-visible-p", type=float, default=0.9,
                    help="P(an ally shows on the minimap); <1 models dead/respawning heroes")
    ap.add_argument("--enemy-visible-p", type=float, default=0.5,
                    help="P(an enemy is visible); low value models fog of war")
    ap.add_argument("--distractors", action="store_true",
                    help="draw synthetic buildings/creeps (OFF by default: the real "
                         "backgrounds already contain authentic buildings/camps)")
    ap.add_argument("--preview", action="store_true", help="also write a 3x3 labelled preview grid")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # seed numpy too so the per-icon pixel noise is reproducible across machines
    import numpy as np
    np.random.seed(args.seed)

    shorts = load_hero_shorts()
    bank = PortraitBank(template_dir=args.template_dir, base_size=args.base_size).load(shorts)
    print(f"portraits loaded: {len(bank.available())}/{len(shorts)} heroes @ {args.base_size}px")

    if args.backgrounds is not None:
        bg_paths = [p for p in sorted(args.backgrounds.rglob("*"))
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
                    and _is_clean_bg(p)]
    else:
        bg_paths = collect_default_backgrounds()
    backgrounds = load_backgrounds(bg_paths)
    if not backgrounds:
        raise SystemExit(
            "No backgrounds found. Pass --backgrounds DIR with real minimap crops, "
            "or record some first (dota2-copilot dump / record)."
        )
    print(f"backgrounds: {len(backgrounds)} image(s) from {len(bg_paths)} path(s)")

    cfg = SynthConfig(
        team_size=args.team_size,
        ally_visible_p=args.ally_visible_p,
        enemy_visible_p=args.enemy_visible_p,
        draw_distractors=args.distractors,
    )

    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"generating {args.train} train + {args.val} val -> {out_root}")
    _write_split("train", args.train, out_root, bank, backgrounds, shorts, cfg, rng)
    _write_split("val", args.val, out_root, bank, backgrounds, shorts, cfg, rng)

    data_yaml = write_data_yaml(out_root, shorts)
    print(f"wrote {data_yaml}")

    if args.preview:
        grid = _preview_grid(out_root, bank, backgrounds, shorts, cfg, rng)
        print(f"wrote preview grid {grid}")

    print("done.")


if __name__ == "__main__":
    main()

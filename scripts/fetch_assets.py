"""Download Dota 2 hero assets from Valve's Steam CDN.

Produces:

    assets/
        heroes.json                 — id ↔ short_name ↔ localized_name table
        card/<short>.png            — Valve's small 32x32 hex hero card icon
                                       (sourced from .../heroes/icons/<short>.png);
                                       same content the minimap renders
        topbar/<short>.png          — Valve's 256x144 wide hero portrait
                                       (sourced from .../heroes/<short>.png);
                                       same content the in-game top hero bar shows
        minimap/<short>_<size>.png  — square downsamples of card/, used as
                                       templates for minimap icon matching

Sources
-------
* Hero list & short-name mapping: https://api.opendota.com/api/constants/heroes
  (mirrors Valve's dotaconstants — free, no API key required)
* Images: https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/...
  (Valve's official Steam CDN; the same URLs used by Dota 2's own web pages)

Run
---
    python scripts/fetch_assets.py
    python scripts/fetch_assets.py --sizes 28 30 32 --workers 16
    python scripts/fetch_assets.py --force        # overwrite existing files

The script is idempotent: cached files are skipped unless --force is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"

CONSTANTS_URL = "https://api.opendota.com/api/constants/heroes"
# Valve's public datafeed — used to pull simplified-Chinese hero names.
# language can be: schinese, tchinese, english, russian, koreana, japanese, …
DATAFEED_URL = "https://www.dota2.com/datafeed/herolist?language=schinese"
CDN_BASE = "https://cdn.cloudflare.steamstatic.com"
CDN_BASE_FALLBACK = "https://cdn.akamai.steamstatic.com"

USER_AGENT = "dota2-copilot/0.1 (+https://github.com/zjr/dota2_copilot)"
DEFAULT_SIZES = (28, 30, 32)
DEFAULT_WORKERS = 12
HTTP_TIMEOUT = 20.0


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _http_get(url: str) -> bytes:
    """GET bytes with a User-Agent and a CDN fallback for image URLs."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code in (403, 404) and CDN_BASE in url:
            alt = url.replace(CDN_BASE, CDN_BASE_FALLBACK)
            with urllib.request.urlopen(
                urllib.request.Request(alt, headers={"User-Agent": USER_AGENT}),
                timeout=HTTP_TIMEOUT,
            ) as r:
                return r.read()
        raise


# ---------------------------------------------------------------------------
# Step 1: hero index
# ---------------------------------------------------------------------------


def _fetch_zh_names() -> dict[int, str]:
    """Return ``{hero_id: simplified_chinese_name}`` from Valve's datafeed.

    Returns an empty dict if the request fails (Chinese names are optional).
    """
    try:
        print(f"[fetch] GET {DATAFEED_URL}")
        raw = _http_get(DATAFEED_URL)
        data = json.loads(raw.decode("utf-8"))
        heroes = data["result"]["data"]["heroes"]
        return {int(h["id"]): h["name_loc"] for h in heroes if h.get("name_loc")}
    except Exception as e:  # noqa: BLE001
        print(f"[fetch] WARN: failed to load Chinese names: {e!r}", file=sys.stderr)
        return {}


def fetch_hero_index() -> list[dict]:
    """Return a list of ``{id, short, localized_name, name_zh, card_url, topbar_url}``.

    ``short`` is the Valve internal short name (``npc_dota_hero_<short>``) used
    in image URLs — e.g. ``"nevermore"`` for Shadow Fiend.
    """
    print(f"[fetch] GET {CONSTANTS_URL}")
    raw = _http_get(CONSTANTS_URL)
    data = json.loads(raw.decode("utf-8"))

    zh_names = _fetch_zh_names()

    heroes: list[dict] = []
    for _id_str, h in data.items():
        npc_name = h["name"]  # e.g. "npc_dota_hero_antimage"
        if not npc_name.startswith("npc_dota_hero_"):
            continue
        short = npc_name[len("npc_dota_hero_") :]
        hero_id = int(h["id"])
        heroes.append(
            {
                "id": hero_id,
                "short": short,
                "localized_name": h["localized_name"],
                "name_zh": zh_names.get(hero_id, ""),
                # h["img"]/h["icon"] are like "/apps/dota2/images/.../xxx.png?" —
                # we strip the trailing "?" cache buster and prepend the CDN host.
                # h["icon"]   -> small 32x32 hex card (same as minimap render)
                # h["img"]    -> wide 256x144 portrait (same as top hero bar)
                "card_url": CDN_BASE + h["icon"].rstrip("?"),
                "topbar_url": CDN_BASE + h["img"].rstrip("?"),
            }
        )
    heroes.sort(key=lambda x: x["id"])
    missing_zh = [h["short"] for h in heroes if not h["name_zh"]]
    print(f"[fetch] got {len(heroes)} heroes ({len(heroes) - len(missing_zh)} with zh names)")
    if missing_zh:
        print(f"[fetch]   no zh name for: {', '.join(missing_zh)}", file=sys.stderr)
    return heroes


# ---------------------------------------------------------------------------
# Step 2: download topbar + portrait images
# ---------------------------------------------------------------------------


def _download_one(url: str, out: Path, force: bool) -> tuple[Path, str]:
    if out.exists() and not force:
        return out, "cached"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = _http_get(url)
    tmp = out.with_suffix(out.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(out)
    return out, "downloaded"


def download_images(
    heroes: list[dict],
    *,
    workers: int,
    force: bool,
) -> None:
    card_dir = ASSETS_DIR / "card"
    topbar_dir = ASSETS_DIR / "topbar"
    card_dir.mkdir(parents=True, exist_ok=True)
    topbar_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path]] = []
    for h in heroes:
        jobs.append((h["card_url"], card_dir / f"{h['short']}.png"))
        jobs.append((h["topbar_url"], topbar_dir / f"{h['short']}.png"))

    total = len(jobs)
    print(f"[fetch] downloading {total} images with {workers} workers")

    done = 0
    failed: list[tuple[str, Path, str]] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_download_one, url, out, force): (url, out) for url, out in jobs}
        for fut in as_completed(futures):
            url, out = futures[fut]
            try:
                _, status = fut.result()
                done += 1
                if done % 25 == 0 or done == total:
                    elapsed = time.time() - started
                    print(f"[fetch]   {done}/{total} ({elapsed:.1f}s)")
            except Exception as e:  # noqa: BLE001
                failed.append((url, out, repr(e)))

    if failed:
        print(f"[fetch] {len(failed)} downloads failed:", file=sys.stderr)
        for url, out, err in failed:
            print(f"  - {out.name}: {err}  ({url})", file=sys.stderr)


# ---------------------------------------------------------------------------
# Step 3: generate minimap templates (square downsamples)
# ---------------------------------------------------------------------------


def _square_crop(img: np.ndarray) -> np.ndarray:
    """Center-crop an image to a square."""
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0 : y0 + side, x0 : x0 + side]


def generate_minimap_templates(
    heroes: list[dict],
    sizes: tuple[int, ...],
    *,
    force: bool,
) -> None:
    src_dir = ASSETS_DIR / "card"
    out_dir = ASSETS_DIR / "minimap"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gen ] minimap templates at sizes={sizes}")
    skipped = 0
    written = 0
    for h in heroes:
        src = src_dir / f"{h['short']}.png"
        if not src.exists():
            skipped += 1
            continue
        # Read with alpha so we can drop transparent borders later if needed.
        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            skipped += 1
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            # Composite alpha onto black so resize doesn't bleed.
            bgr = img[:, :, :3]
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            img = (bgr.astype(np.float32) * alpha).astype(np.uint8)

        sq = _square_crop(img)
        for s in sizes:
            out = out_dir / f"{h['short']}_{s}.png"
            if out.exists() and not force:
                continue
            resized = cv2.resize(sq, (s, s), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out), resized)
            written += 1

    print(f"[gen ] wrote {written} files, skipped {skipped} heroes with missing source")


# ---------------------------------------------------------------------------
# Step 4: write heroes.json index
# ---------------------------------------------------------------------------


def write_index(heroes: list[dict], sizes: tuple[int, ...]) -> Path:
    out = ASSETS_DIR / "heroes.json"
    payload = {
        "source": {
            "constants": CONSTANTS_URL,
            "cdn": CDN_BASE,
        },
        "minimap_sizes": list(sizes),
        "heroes": [
            {
                "id": h["id"],
                "short": h["short"],
                "localized_name": h["localized_name"],
                "name_zh": h.get("name_zh", ""),
                "card": f"card/{h['short']}.png",
                "topbar": f"topbar/{h['short']}.png",
                "minimap": [f"minimap/{h['short']}_{s}.png" for s in sizes],
            }
            for h in heroes
        ],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[idx ] wrote {out}  ({len(heroes)} heroes)")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Dota 2 hero assets.")
    ap.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="Square sizes (px) to generate for minimap templates.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel download workers.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-download / re-render even if files already exist.",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Only regenerate minimap templates + heroes.json from existing card/.",
    )
    args = ap.parse_args()

    heroes = fetch_hero_index()
    if not args.skip_download:
        download_images(heroes, workers=args.workers, force=args.force)
    generate_minimap_templates(heroes, tuple(args.sizes), force=args.force)
    write_index(heroes, tuple(args.sizes))

    print("[done] assets ready under assets/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

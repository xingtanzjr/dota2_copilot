"""Canonical class scheme for the YOLO minimap detector.

We detect 254 classes: every hero (from ``assets/heroes.json``) in two team
flavours. The team is the *player-perspective* border colour Dota renders on the
minimap:

* ``ally``  — green border  (your own team, whether you are Radiant or Dire)
* ``enemy`` — red border    (the opposing team)

Class-id layout (stable, derived from the hero order in ``heroes.json``)::

    class_id = hero_index * 2 + team_bit        # team_bit: 0 = ally, 1 = enemy
    hero_index = class_id // 2
    team_bit   = class_id %  2

Keeping ally/enemy adjacent means the two flavours of one hero are always
``2k`` and ``2k+1`` -- handy for grouping and for a possible team-agnostic
collapse later.

This module deliberately avoids importing the main app config (pydantic) so it
stays usable in a minimal synth/training environment.
"""

from __future__ import annotations

import json
from pathlib import Path

# training/ -> dota2_copilot/ -> src/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
HEROES_JSON = REPO_ROOT / "assets" / "heroes.json"

TEAMS: tuple[str, str] = ("ally", "enemy")  # index == team_bit


def load_hero_shorts(heroes_json: Path | None = None) -> list[str]:
    """Return hero ``short`` ids in the canonical (file) order."""
    path = heroes_json or HEROES_JSON
    data = json.loads(path.read_text(encoding="utf-8"))
    shorts = [h["short"] for h in data.get("heroes", []) if h.get("short")]
    if not shorts:
        raise ValueError(f"No heroes found in {path}")
    return shorts


def build_class_names(shorts: list[str] | None = None) -> list[str]:
    """Return the 2*len(shorts) class names, ordered by class id.

    Names look like ``"pudge__ally"`` / ``"pudge__enemy"``.
    """
    shorts = shorts if shorts is not None else load_hero_shorts()
    names: list[str] = []
    for short in shorts:
        for team in TEAMS:
            names.append(f"{short}__{team}")
    return names


def class_id(hero_index: int, team: str) -> int:
    """``(hero_index, "ally"|"enemy") -> class_id``."""
    try:
        team_bit = TEAMS.index(team)
    except ValueError as e:  # pragma: no cover - programmer error
        raise ValueError(f"team must be one of {TEAMS}, got {team!r}") from e
    return hero_index * 2 + team_bit


def decode_class(cid: int, shorts: list[str] | None = None) -> tuple[str, str]:
    """``class_id -> (hero_short, "ally"|"enemy")``."""
    shorts = shorts if shorts is not None else load_hero_shorts()
    hero_index, team_bit = divmod(cid, 2)
    if hero_index < 0 or hero_index >= len(shorts):
        raise ValueError(f"class id {cid} out of range for {len(shorts)} heroes")
    return shorts[hero_index], TEAMS[team_bit]


def write_data_yaml(
    dataset_root: Path,
    shorts: list[str] | None = None,
) -> Path:
    """Write an Ultralytics ``data.yaml`` for ``dataset_root`` and return its path.

    Expects the standard layout under ``dataset_root``::

        images/train  images/val
        labels/train  labels/val
    """
    import yaml

    shorts = shorts if shorts is not None else load_hero_shorts()
    names = build_class_names(shorts)
    dataset_root = dataset_root.resolve()
    doc = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": {i: n for i, n in enumerate(names)},
    }
    out = dataset_root / "data.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    return out

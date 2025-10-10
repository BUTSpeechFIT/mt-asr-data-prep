#!/usr/bin/env python3
"""
Batch convert all Lhotse SupervisionSets in a directory tree to STM format.

Example:
    python generate_all_stm.py /path/to/root-dir
"""

from pathlib import Path
import re
import argparse
import logging
from lhotse import SupervisionSet
from typing import Optional


# ------------------------- Logging setup -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------- Helpers -------------------------
def sanitize_text(text: str | None) -> str:
    """Clean transcript for STM format (remove tabs/newlines and normalize spaces)."""
    if not text:
        return ""
    return " ".join(text.split())


def to_iterable_channels(ch):
    """Normalize channel field (may be int, list, None)."""
    if ch is None:
        return [1]
    if isinstance(ch, (list, tuple, set)):
        return list(ch) if ch else [1]
    return [ch]


def supervision_set_to_stm(sups: SupervisionSet):
    """Convert one SupervisionSet to STM lines."""
    rows = []
    for sup in sups:
        rec_id = sup.recording_id or sup.id
        speaker = sup.speaker or "unknown"
        start = float(sup.start or 0.0)
        end = start + float(sup.duration or 0.0)
        text = sanitize_text(sup.text)
        for ch in to_iterable_channels(sup.channel):
            rows.append((rec_id, ch, start, f"{rec_id} {ch} {speaker} {start:.3f} {end:.3f} {text}"))
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    return [r[-1] for r in rows]


# ------------------------- Main processing -------------------------
def convert_all(root_dir: Path, output_dir: Optional[Path] = None):
    """
    Convert all files matching '*_supervisions_*.jsonl.gz' under `root_dir/manifests`
    to STM format.
    """
    manifests_dir = root_dir / "manifests"
    stm_root = output_dir or (root_dir / "stms")
    stm_root.mkdir(parents=True, exist_ok=True)

    manifest_files = sorted(manifests_dir.glob("*/*supervisions*.jsonl.gz"))
    if not manifest_files:
        logger.error("No supervision manifests found.")
        return

    for manifest_path in manifest_files:
        match = re.match(
            r"^(?P<prefix>.+?)_supervisions(?:_(?P<suffix>.+))?\.jsonl\.gz$",
            manifest_path.name,
        )
        if not match:
            logger.warning(f"Skipping file with unexpected name: {manifest_path.name}")
            continue

        prefix = match.group("prefix")   # e.g. "ami-sdm"
        suffix = match.group("suffix")   # e.g. "train", "train_L"

        dataset_dir = manifest_path.parent.name  # e.g. "ami"

        stm_dir = stm_root / dataset_dir
        stm_dir.mkdir(parents=True, exist_ok=True)

        if suffix:
            stm_filename = f"{prefix}_{suffix}.stm"
        else:
            stm_filename = f"{prefix}.stm"

        stm_path = stm_dir / stm_filename

        logger.info(f"Processing {dataset_dir}/{manifest_path.name} -> {stm_path}")

        sups = SupervisionSet.from_file(manifest_path)
        stm_lines = supervision_set_to_stm(sups)

        with open(stm_path, "w", encoding="utf-8") as f:
            f.write("\n".join(stm_lines) + "\n")

        logger.info(f"Wrote {len(stm_lines)} lines to {stm_path}")

    logger.info("All STM files successfully generated.")


# ------------------------- CLI entry -------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert all Lhotse supervision manifests to STM format.")
    parser.add_argument("root_dir", type=Path, help="Root directory containing 'manifests/' subfolder.")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Optional output root directory for STM files (default: <root_dir>/stms)."
    )

    args = parser.parse_args()
    convert_all(args.root_dir, args.output)


if __name__ == "__main__":
    main()

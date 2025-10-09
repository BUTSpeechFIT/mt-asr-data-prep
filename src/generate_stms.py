#!/usr/bin/env python3
"""
Batch convert all Lhotse SupervisionSets in a directory tree to STM format.

Example:
    python generate_all_stm.py root-dir
"""

from pathlib import Path
import re
import typer
from lhotse import SupervisionSet
from typing import Optional


app = typer.Typer(add_completion=False, help="Convert all Lhotse supervision manifests to STM format.")


def sanitize_text(text: Optional[str]) -> str:
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


def supervision_to_stm_lines(sups: SupervisionSet):
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


@app.command()
def convert_all(
    root_dir: Path = typer.Argument(..., help="Root directory containing 'manifests/' subfolder."),
    output_dir: Optional[Path] = typer.Option(None, "-o", "--output", help="Output root directory for STM files."),
):
    """
    Convert all files matching '*_supervisions_*.jsonl.gz' under `root_dir/manifests`
    to STM format.

    For example:
      manifests/ami/ami-sdm_supervisions_train.jsonl.gz
      → stms/ami/ami-sdm_train.stm
    """

    manifests_dir = root_dir / "manifests"
    stm_root = output_dir or (root_dir / "stms")
    stm_root.mkdir(parents=True, exist_ok=True)

    manifest_files = sorted(manifests_dir.glob("*/*_supervisions_*.jsonl.gz"))
    if not manifest_files:
        typer.echo("No supervision manifests found.")
        raise typer.Exit(1)

    for manifest_path in manifest_files:
        match = re.match(r"^(?P<prefix>.+?)_supervisions_(?P<suffix>.+)\.jsonl\.gz$", manifest_path.name)
        if not match:
            typer.echo(f"Skipping file with unexpected name: {manifest_path.name}")
            continue

        prefix = match.group("prefix")   # e.g. "ami-sdm"
        suffix = match.group("suffix")   # e.g. "train", "train_L"

        dataset_dir = manifest_path.parent.name  # e.g. "ami"
        stm_dir = stm_root / dataset_dir
        stm_dir.mkdir(parents=True, exist_ok=True)

        stm_path = stm_dir / f"{prefix}_{suffix}.stm"

        typer.echo(f"Processing {dataset_dir}/{manifest_path.name} -> {stm_path}")

        sups = SupervisionSet.from_file(manifest_path)
        stm_lines = supervision_to_stm_lines(sups)

        with open(stm_path, "w", encoding="utf-8") as f:
            f.write("\n".join(stm_lines) + "\n")

        typer.echo(f"Wrote {len(stm_lines)} lines to {stm_path}")

    typer.echo("All STM files successfully generated.")


if __name__ == "__main__":
    app()


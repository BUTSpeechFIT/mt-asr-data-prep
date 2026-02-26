#!/usr/bin/env python3
import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import textgrid  # pip install textgrid

from lhotse import CutSet
from lhotse.supervision import AlignmentItem


def align_cut(cut, textgrid_dir):
    """
    Process a single cut. Returns the aligned cut if successful, or None if skipped.
    This must be defined at the top level so it can be pickled for multiprocessing.
    """
    tg_path = os.path.join(textgrid_dir, f"{cut.id}.TextGrid")

    if not os.path.exists(tg_path):
        logging.warning(f"Skipped '{cut.id}': TextGrid file not found ({tg_path}).")
        return None

    try:
        tg = textgrid.TextGrid.fromFile(tg_path)
    except Exception as e:
        logging.warning(f"Skipped '{cut.id}': Failed to parse TextGrid. Error: {e}")
        return None

    words = []
    try:
        words_tier = tg.getFirst("words")

        for interval in words_tier.intervals:
            if interval.mark and interval.mark not in ["", "spn", "sil", "<eps>"]:
                words.append(
                    AlignmentItem(
                        symbol=interval.mark,
                        start=round(interval.minTime, 6) + cut.start,
                        duration=round(interval.maxTime - interval.minTime, 6),
                    )
                )

        if words:
            for sup in cut.supervisions:
                # Ensure the alignment dictionary exists before adding to it
                if sup.alignment is None:
                    sup.alignment = {}
                sup.alignment["word"] = words
            return cut
        else:
            logging.warning(f"Skipped '{cut.id}': No valid words found in the tier.")
            return None

    except ValueError:
        logging.warning(f"Skipped '{cut.id}': 'words' tier not found in TextGrid.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Merge MFA TextGrids into Lhotse cuts natively."
    )
    parser.add_argument(
        "--cuts_file", required=True, help="Path to input cuts (JSONL.GZ)."
    )
    parser.add_argument(
        "--textgrid_dir", required=True, help="Directory containing MFA TextGrids."
    )
    parser.add_argument(
        "--out_cuts", required=True, help="Path to save aligned cuts (JSONL.GZ)."
    )
    parser.add_argument(
        "-j",
        "--num_jobs",
        type=int,
        default=1,
        help="Number of background workers for parallel processing.",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    print(f"Loading cuts from {args.cuts_file}...")
    cuts = CutSet.from_file(args.cuts_file)
    total_count = len(cuts)

    # Bind the textgrid directory to the function so it only requires the 'cut' argument
    align_fn = partial(align_cut, textgrid_dir=args.textgrid_dir)

    print(f"Aligning cuts using {args.num_jobs} worker(s)...")

    if args.num_jobs > 1:
        # Multiprocessing Map: Distributes the workload across CPU cores
        with ProcessPoolExecutor(max_workers=args.num_jobs) as executor:
            mapped_cuts = executor.map(align_fn, cuts)

        # Load the successfully aligned cuts back into a CutSet natively (dropping None values)
        aligned_cuts = CutSet.from_items(c for c in mapped_cuts if c is not None)
    else:
        # Standard Lhotse Sequential Map + Filter
        aligned_cuts = cuts.map(align_fn).filter(lambda c: c is not None)

    aligned_cuts.to_file(args.out_cuts)

    # Note: If memory becomes an issue on massive datasets, you can wrap aligned_cuts
    # in a list() to force eager evaluation before printing the length.
    print(f"Wrote aligned cuts to {args.out_cuts}")
    print(f"Matched TextGrids: {len(aligned_cuts)}/{total_count}")


if __name__ == "__main__":
    main()

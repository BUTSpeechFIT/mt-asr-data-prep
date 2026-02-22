#!/usr/bin/env python3
import argparse
import os

import textgrid  # pip install textgrid

from lhotse import CutSet
from lhotse.supervision import AlignmentItem


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
    args = parser.parse_args()

    print(f"Loading cuts from {args.cuts_file}...")
    cuts = CutSet.from_file(args.cuts_file)

    matched_count = 0
    total_count = len(cuts)

    def process_cuts():
        nonlocal matched_count
        for cut in cuts:
            # Assuming your cut.id perfectly matches the MFA TextGrid filename
            tg_path = os.path.join(args.textgrid_dir, f"{cut.id}.TextGrid")

            if os.path.exists(tg_path):
                tg = textgrid.TextGrid.fromFile(tg_path)
                words = []

                try:
                    # MFA typically names the word-level tier "words"
                    words_tier = tg.getFirst("words")

                    for interval in words_tier.intervals:
                        # Filter out silence/noise markers
                        if interval.mark and interval.mark not in [
                            "",
                            "spn",
                            "sil",
                            "<eps>",
                        ]:
                            words.append(
                                AlignmentItem(
                                    symbol=interval.mark,
                                    start=round(interval.minTime, 6),
                                    duration=round(
                                        interval.maxTime - interval.minTime, 6
                                    ),
                                )
                            )

                    # Assign the AlignmentItem lists to Lhotse's native alignment dictionary
                    if words:
                        for sup in cut.supervisions:
                            sup.alignment = {"word": words}
                        matched_count += 1
                except ValueError:
                    pass  # "words" tier not found in this TextGrid

            yield cut

    # Convert generator back to a CutSet and write to disk
    aligned_cuts = CutSet.from_cuts(process_cuts())
    aligned_cuts.to_file(args.out_cuts)

    print(
        f"Wrote aligned cuts to {args.out_cuts} (Matched TextGrids: {matched_count}/{total_count})"
    )


if __name__ == "__main__":
    main()

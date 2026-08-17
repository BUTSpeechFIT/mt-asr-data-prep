"""Cut long recordings into ~fixed-length windows whose boundaries fall in silence.

`pre_segment_using_alignments.py` bounds a cut by the supervisions it groups, so `--max_len`
is an upper bound on speech rather than a window length. This instead targets a window size and
places each boundary at the nearest point where *nobody* is speaking, which means no supervision
is ever split -- in particular none is cut inside overlapped speech, where truncation would
leave one speaker's audio clipped against another's.

Boundaries are chosen from the silence gaps between merged speech intervals: for a target end
`start + window`, the gap midpoint closest to that target within `--tolerance` wins. A window is
therefore approximately, not exactly, `--window` seconds long. `--stochastic` adds copies of
each recording windowed from a different random starting offset, which gives the model different
groupings of the same audio.

Usage:
    python src/window_at_silence.py \
        --input  data/manifests/ami/ami-sdm_cutset_train.jsonl.gz \
        --output data/manifests/ami/ami-sdm_cutset_train_600s.jsonl.gz \
        --window 600 --stochastic --num_stochastic_copies 2 --seed 1234
"""

import argparse
import logging
import random
from typing import List, Optional, Tuple

from lhotse import CutSet, load_manifest


def merged_speech_intervals(cut) -> List[Tuple[float, float]]:
    """Speech spans of `cut`, with overlapping supervisions merged into one interval."""
    spans = sorted((s.start, s.end) for s in cut.supervisions)
    merged: List[Tuple[float, float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def silence_boundaries(cut, min_gap: float) -> List[float]:
    """Candidate cut points: the midpoint of every silence gap of at least `min_gap` seconds.

    A point inside such a gap has no supervision active, so truncating there splits nothing --
    which is the whole objective. The recording's own end is included as a boundary.
    """
    speech = merged_speech_intervals(cut)
    boundaries = []
    previous_end = 0.0
    for start, end in speech:
        if start - previous_end >= min_gap:
            boundaries.append((previous_end + start) / 2.0)
        previous_end = max(previous_end, end)
    if cut.duration - previous_end >= min_gap:
        boundaries.append((previous_end + cut.duration) / 2.0)
    boundaries.append(cut.duration)
    return boundaries


def window_cut(cut, window: float, min_gap: float, start: float = 0.0) -> Tuple[List, int]:
    """Window one cut from `start`, snapping every boundary to a silence gap.

    `window` is a hard maximum, so each boundary is the *latest* silence at or before
    `start + window`: that keeps windows as close to the cap as the silence allows, which is
    what holds the duration distribution up against it. Returns the windows and how many times
    no silence existed in range, forcing a cut through speech.
    """
    boundaries = silence_boundaries(cut, min_gap)
    windows = []
    forced = 0
    index = 0

    while start < cut.duration - min_gap:
        if cut.duration - start <= window:
            end = cut.duration          # the tail already fits under the cap
        else:
            candidates = [b for b in boundaries if start + min_gap < b <= start + window]
            if candidates:
                end = max(candidates)
            else:
                # 600 s of unbroken speech: nothing can be cut cleanly, so cut at the cap and
                # let the straddling supervisions go. Vanishingly rare; counted and reported.
                end = start + window
                forced += 1

        piece = cut.truncate(offset=start, duration=end - start, keep_excessive_supervisions=False)
        if piece.supervisions:
            windows.append(piece.with_id(f"{cut.id}-w{index}"))
            index += 1
        start = end

    return windows, forced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Full-session CutSet")
    parser.add_argument("--output", required=True, help="Windowed CutSet to write")
    parser.add_argument("--window", type=float, default=600.0,
                        help="Maximum window length (s). Never exceeded: boundaries snap to the "
                             "latest silence at or before it")
    parser.add_argument("--min_gap", type=float, default=0.2,
                        help="Shortest silence usable as a boundary (s)")
    parser.add_argument("--min_duration", type=float, default=60.0,
                        help="Drop windows shorter than this. Each recording and each stochastic "
                             "copy leaves a leftover tail, which without this can be seconds long")
    parser.add_argument("--stochastic", action="store_true",
                        help="Also emit copies windowed from random starting offsets")
    parser.add_argument("--num_stochastic_copies", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = random.Random(args.seed)

    cuts = load_manifest(args.input)
    windows = []
    forced_total = 0
    for cut in cuts:
        starts = [0.0]
        if args.stochastic:
            starts += [rng.uniform(0.0, args.window) for _ in range(args.num_stochastic_copies)]
        for copy_index, start in enumerate(starts):
            pieces, forced = window_cut(cut, args.window, args.min_gap, start=start)
            forced_total += forced
            for piece in pieces:
                windows.append(piece.with_id(f"{piece.id}-c{copy_index}") if copy_index else piece)
    if forced_total:
        logging.info("%d boundaries had no silence within the window and cut through speech",
                     forced_total)

    kept = [c for c in windows if c.duration >= args.min_duration]
    if len(kept) != len(windows):
        logging.info("Dropped %d/%d windows shorter than %.0f s",
                     len(windows) - len(kept), len(windows), args.min_duration)

    result = CutSet.from_cuts(kept)
    result.to_file(args.output)

    durations = sorted(c.duration for c in result)
    speakers = [len({s.speaker for s in c.supervisions}) for c in result]
    total = sum(durations)
    logging.info(
        "%d windows from %d recordings, %.1f h\n"
        "  duration  min %.0f s  p10 %.0f s  median %.0f s  p90 %.0f s  max %.0f s\n"
        "  speakers  median %d  max %d\n"
        "  items (window, speaker) %d",
        len(durations), len(cuts), total / 3600,
        durations[0], durations[len(durations) // 10], durations[len(durations) // 2],
        durations[9 * len(durations) // 10], durations[-1],
        sorted(speakers)[len(speakers) // 2], max(speakers), sum(speakers),
    )


if __name__ == "__main__":
    main()

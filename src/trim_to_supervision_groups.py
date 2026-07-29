import argparse
import random
from typing import Optional

from lhotse import CutSet, fastcopy, load_manifest


def _apply_stochastic_offset(
    cut,
    offset_window: float,
    rng: random.Random,
    max_offset_gap: float = 1.0,
    id_suffix: str = "",
):
    """
    Randomly shift where a long recording's grouping/windowing will start, analogous
    to `pre_segment_using_alignments._apply_stochastic_offset` but for corpora without
    word-level alignments: snaps to the nearest SUPERVISION start (instead of a word
    start) at or after a randomly sampled point in ``[0, offset_window)``, then backs
    off a further random ``[0, max_offset_gap]`` seconds of leading silence, capped so
    it never overlaps the end of a preceding (non-overlapping) supervision.

    Since there's no word-level alignment to trim a straddling supervision precisely,
    any supervision that overlaps the chosen offset (e.g. cross-speaker overlap in
    meeting audio) is dropped rather than split — this mirrors how the rest of this
    no-alignment pipeline drops (via filter_by_length.py) whatever it can't split
    precisely, instead of guessing.

    Returns None if the cut is already shorter than `offset_window`, if no supervision
    starts at or after the sampled point, or if nothing survives the truncation.
    """
    if cut.duration <= offset_window:
        return None

    sup_starts = sorted(s.start for s in cut.supervisions)
    target = rng.uniform(0, offset_window)
    candidates = [t for t in sup_starts if t >= target]
    if not candidates:
        return None
    sup_offset = candidates[0]

    prev_ends = [s.end for s in cut.supervisions if s.end <= sup_offset]
    floor = max(prev_ends) if prev_ends else 0.0
    gap = rng.uniform(0, min(max_offset_gap, sup_offset - floor))
    offset = sup_offset - gap

    truncated = cut.truncate(
        offset=offset, keep_excessive_supervisions=False, preserve_id=True
    )
    if not truncated.supervisions:
        return None

    return fastcopy(
        truncated,
        id=f"{cut.id}{id_suffix}",
        supervisions=[
            fastcopy(s, id=f"{s.id}{id_suffix}") for s in truncated.supervisions
        ],
    )


def main(
    input_manifest: str,
    output_manifest: str,
    max_pause: float,
    num_jobs: int,
    stochastic: bool = False,
    num_stochastic_copies: int = 2,
    offset_window: float = 30.0,
    max_offset_gap: float = 1.0,
    seed: Optional[int] = None,
):
    cuts = load_manifest(input_manifest).to_eager()

    if stochastic and num_stochastic_copies > 0:
        rng = random.Random(seed)
        all_cuts = list(cuts)
        for variant in range(num_stochastic_copies):
            id_suffix = f"-off{variant}"
            variant_cuts = [
                _apply_stochastic_offset(
                    c, offset_window, rng, max_offset_gap=max_offset_gap, id_suffix=id_suffix
                )
                for c in cuts
            ]
            all_cuts.extend(c for c in variant_cuts if c is not None)
        cuts = CutSet.from_cuts(all_cuts)

    cuts = cuts.trim_to_supervision_groups(max_pause=max_pause, num_jobs=num_jobs).to_eager()
    cuts.to_file(output_manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trim cuts to supervision groups (utterance groups), i.e. sets of "
        "supervisions with no gaps longer than --max_pause between them. Used as a "
        "no-alignment substitute for pre_segment_using_alignments.py when the corpus "
        "has no word-level alignments to split long recordings precisely."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input cutset manifest"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output cutset manifest"
    )
    parser.add_argument(
        "--max_pause",
        type=float,
        default=2.0,
        help="Max gap (seconds) between supervisions to keep them in the same group",
    )
    parser.add_argument(
        "--num_jobs", type=int, default=8, help="Number of parallel jobs"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="In addition to the base (offset=0) grouping, generate "
        "--num_stochastic_copies extra copies of each long recording with a random, "
        "supervision-boundary-aligned starting offset (plus a random leading silence "
        "gap). Multiplies the amount of output data.",
    )
    parser.add_argument(
        "--num_stochastic_copies",
        type=int,
        default=2,
        help="Number of extra randomly-offset copies per long recording (only used "
        "with --stochastic)",
    )
    parser.add_argument(
        "--offset_window",
        type=float,
        default=30.0,
        help="Range [0, offset_window) to sample the random offset target from; "
        "should generally match the downstream filter_by_length.py --max_len "
        "(only used with --stochastic)",
    )
    parser.add_argument(
        "--max_offset_gap",
        type=float,
        default=1.0,
        help="Max random leading silence in seconds before the snapped supervision "
        "boundary (only used with --stochastic)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for --stochastic offsets"
    )

    args = parser.parse_args()

    main(
        args.input,
        args.output,
        args.max_pause,
        args.num_jobs,
        stochastic=args.stochastic,
        num_stochastic_copies=args.num_stochastic_copies,
        offset_window=args.offset_window,
        max_offset_gap=args.max_offset_gap,
        seed=args.seed,
    )

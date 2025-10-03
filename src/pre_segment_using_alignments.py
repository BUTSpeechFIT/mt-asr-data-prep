import argparse
import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lhotse
from lhotse import CutSet, fastcopy, load_manifest
from lhotse.lazy import LazyFlattener, LazyMapper


PROCESSING_REGEX_STR = "!\"#$%&()*+,./:;<=>?@[\\]^_`'{|}~"
ALIGNMENT_WORD_MAP = {  # Normalization map between the words and the alignment (works for AMI and NOTSOFAR).
    'mm-hmm': 'mmm',
}

# Precomputed constants for token normalization and timing precision.
TRANSLATOR = str.maketrans('', '', PROCESSING_REGEX_STR)
EPS = 1e-5
WORD_ALIGNMENT_KEY = 'word'

def _normalize_word(word: str) -> str:
    """
    Lowercase a token and strip punctuation/special characters defined in
    `PROCESSING_REGEX_STR`.

    Args:
        word (str): The input word to normalize.

    Returns:
        str: The normalized word with punctuation removed and lowercased.
    """
    return word.translate(TRANSLATOR).lower()


def filter_punctuation_alignments(cut: lhotse.cut.Cut) -> lhotse.cut.Cut:
    """
    Filter out alignment entries that are only punctuation/special characters.

    Leaves the supervision text as-is; only prunes alignment tokens whose normalized
    symbol is empty after stripping punctuation.

    Args:
        cut (lhotse.cut.Cut): The input cut whose word alignments should be filtered.

    Returns:
        lhotse.cut.Cut: The same cut with punctuation-only alignment entries removed
        from each supervision's `word` alignment.
    """
    new_sups: List[Any] = []
    for sup in cut.supervisions:
        new_alignments = [
            word
            for word in sup.alignment[WORD_ALIGNMENT_KEY]
            if len(_normalize_word(word.symbol)) > 0
        ]
        sup.alignment[WORD_ALIGNMENT_KEY] = new_alignments
        new_sups.append(sup)
    cut.supervisions = new_sups
    return cut


def _prepare_segmented_data(
        cuts: CutSet,
        output_path: str,
        max_segment_duration: float = 30.0,
        num_jobs: int = 1,
) -> lhotse.CutSet:
    """
    Trim, normalize, and split a `CutSet` into segments not exceeding the
    specified maximum duration.

    Args:
        cuts (CutSet): Input cuts to process.
        output_path (str): Path where the resulting manifest will be written.
        max_segment_duration (float, optional): Maximum duration (seconds) per
            segment. Defaults to 30.0.
        num_jobs (int, optional): Number of parallel jobs for processing. Defaults to 1.

    Returns:
        lhotse.CutSet: The processed and segmented cut set.
    """
    output_path = Path(output_path) if output_path else None
    logging.info("Trimming to groups with max_pause=2s")
    cuts = cuts.trim_to_supervision_groups(max_pause=2, num_jobs=num_jobs).to_eager()
    for c in cuts:
        for s in c.supervisions:
            if WORD_ALIGNMENT_KEY in s.alignment:
                s.alignment[WORD_ALIGNMENT_KEY] = [
                    a.with_offset(-c.start) for a in s.alignment[WORD_ALIGNMENT_KEY]
                ]
    logging.info(f"Windowing the overlapping segments to max {int(max_segment_duration)}s")

    cuts = cuts.map(filter_punctuation_alignments).to_eager()
    cuts = split_overlapping_segments(
        cuts,
        max_segment_duration=max_segment_duration,
        num_jobs=num_jobs,
    ).to_eager()

    logging.info("Saving the output")
    cuts.to_file(output_path)
    return cuts


def split_overlapping_segments(cutset: CutSet, max_segment_duration: float = 30, num_jobs: int = 1) -> CutSet:
    """
    Split a CutSet containing overlapping segments into smaller chunks while preserving speaker overlap information.

    This function processes a CutSet by splitting it into smaller segments based on speaker overlap patterns
    and a maximum duration constraint. It can operate in either single-threaded or multi-threaded mode.

    Args:
        cutset (CutSet): The input CutSet containing potentially overlapping segments.
        max_segment_duration (float, optional): Maximum duration in seconds for each resulting segment.
            Defaults to 30 seconds.
        num_jobs (int, optional): Number of parallel jobs to use for processing. If <= 1, runs in
            single-threaded mode. Defaults to 1.

    Returns:
        CutSet: A new CutSet containing the split segments, where each segment respects the
            max_segment_duration constraint while preserving speaker overlap information.

    Note:
        The splitting algorithm attempts to find natural break points where there is minimal
        speaker overlap to create the new segments. This helps maintain the integrity of
        conversational dynamics in the resulting segments.
    """
    if num_jobs <= 1:
        return _split_overlapping_segments_single(cutset, max_segment_duration)
    
    from lhotse.manipulation import split_parallelize_combine
    return split_parallelize_combine(
        num_jobs, cutset, _split_overlapping_segments_single, max_len=max_segment_duration
    )


def _split_overlapping_segments_single(cutset: CutSet, max_len: float = 30) -> CutSet:
    """
    Single-process implementation of `split_overlapping_segments`.

    Args:
        cutset (CutSet): The input cuts to be split.
        max_len (float, optional): Maximum segment duration (seconds). Defaults to 30.

    Returns:
        CutSet: The resulting cut set with segments not exceeding `max_len`.
    """
    split_fn = partial(_split_cut_perseg, max_len=max_len)
    return CutSet(LazyFlattener(LazyMapper(cutset, split_fn))).to_eager()


def get_overlapping_sups(prev_group: List[List[Any]], sup: Any) -> List[List[Any]]:
    """
    Return supervisions from the previous group that overlap with the given supervision's start time.

    Args:
        prev_group (List[List[Any]]): Previous supervision group represented as
            a list of `[supervision, original_index]` pairs.
        sup (Any): The supervision to test overlap against.

    Returns:
        List[List[Any]]: The subset of `prev_group` that overlaps with `sup`.
    """
    return [s for s in prev_group if s[0].start <= sup.start < s[0].end and sup.id not in s[0].id]


def select_words_within_segment(
    sup: Any,
    segment_start: float,
    segment_end: float,
) -> Tuple[List[str], List[Any], float, float, int]:
    """
    Select words and word-alignment entries that fall within the inclusive
    interval [segment_start, segment_end].

    Args:
        sup (Any): The supervision containing text and `word` alignments.
        segment_start (float): Start of the segment (seconds).
        segment_end (float): End of the segment (seconds), inclusive.

    Returns:
        Tuple[List[str], List[Any], float, float, int]:
            - List[str]: Words within the segment.
            - List[Any]: Alignment entries corresponding to the selected words.
            - float: The start time of the first selected aligned word (or -1 if none).
            - float: The end time of the last selected aligned word (or `sup.start` if none).
            - int: The current alignment index after processing (number of consumed alignments).
    """
    words_within_segment: List[str] = []
    alignments_within_segment: List[Any] = []
    fst_word_start_time: float = -1
    last_word_end_time: float = sup.start
    alig_idx: int = 0
    current_alig = sup.alignment[WORD_ALIGNMENT_KEY][alig_idx] if alig_idx < len(sup.alignment[WORD_ALIGNMENT_KEY]) else None
    alig_symbol_split = (
        [_normalize_word(x) for x in current_alig.symbol.split()]
        if current_alig is not None else None
    )

    if WORD_ALIGNMENT_KEY in sup.alignment and len(sup.alignment[WORD_ALIGNMENT_KEY]) > 0:
        for w in sup.text.split():
            adjusted_w = _normalize_word(w)

            # If the current word matches with the alignment.
            if current_alig is not None and (adjusted_w == alig_symbol_split[0] or (
                    adjusted_w in ALIGNMENT_WORD_MAP and ALIGNMENT_WORD_MAP[adjusted_w] == alig_symbol_split[0])):

                if current_alig.end > segment_end:
                    break

                if current_alig.start >= segment_start:
                    if fst_word_start_time == -1:
                        fst_word_start_time = current_alig.start
                    words_within_segment.append(w)
                    alignments_within_segment.append(sup.alignment[WORD_ALIGNMENT_KEY][alig_idx])

                alig_symbol_split.pop(0)
                last_word_end_time = current_alig.end
                if not alig_symbol_split:
                    alig_idx += 1
                    current_alig = sup.alignment[WORD_ALIGNMENT_KEY][alig_idx] if alig_idx < len(sup.alignment[WORD_ALIGNMENT_KEY]) else None
                    alig_symbol_split = (
                        [_normalize_word(x) for x in current_alig.symbol.split()]
                        if current_alig is not None else None
                    )
            else:
                # Special symbol that is not aligned -> appending to words and assuming the same timing info as the previous one.
                if fst_word_start_time != -1:
                    words_within_segment.append(w)

    return words_within_segment, alignments_within_segment, fst_word_start_time, last_word_end_time, alig_idx


def _segment_end(start_time: float, max_len: float) -> float:
    """
    Return an inclusive segment end bound using a small epsilon to avoid
    floating point boundary issues when comparing times.

    Args:
        start_time (float): Segment start time in seconds.
        max_len (float): Maximum segment duration in seconds.

    Returns:
        float: The inclusive end time bound of the segment.
    """
    return start_time + max_len - EPS


def _split_cut_perseg(cut: lhotse.cut.Cut, max_len: float = 30, use_ovl_fb_sups: bool = True):
    """
    Split a single cut into sequential segments of at most `max_len` seconds,
    attempting to preserve conversational structure and handle overlapping
    supervisions with alignment-aware boundaries.

    Args:
        cut (lhotse.cut.Cut): The input cut to split.
        max_len (float, optional): Maximum duration (seconds) per resulting segment.
            Defaults to 30.
        use_ovl_fb_sups (bool, optional): Whether to create fallback overlapping
            fragments from previous segments based on alignments. Defaults to True.

    Returns:
        List[lhotse.cut.Cut]: A list of new cuts representing the segments.
    """
    if len(cut.supervisions) == 0:
        return []

    if cut.duration < max_len:
        return [cut]

    sups = sorted(cut.supervisions, key=lambda s: s.start)

    sup_groups = []
    # Flag = True <=> speaker's utterance is unfinished.
    sup_group_flags = []

    current_sup_group: List[List[Any]] = [[fastcopy(sups[0], id=f'{sups[0].id}-{0}-{0}'), 0]]
    current_sup_group_flags: Dict[str, bool] = dict()
    fallback_sup_idx: int = -1
    is_falling_back: bool = False

    idx = 1
    while idx < len(sups) + 1:
        # print(idx)
        sup = sups[idx] if idx < len(sups) else None
        # We need to add all supervisions that start before the end of the current max 30s long segment. Then, we need to post-process the short ones and return back.
        if sup is not None and (not current_sup_group or sup.start - current_sup_group[0][0].start < max_len):
            current_sup_group.append([fastcopy(sup, id=f'{sup.id}-{idx}-{len(current_sup_group)}'), idx])
            assert sup.start >= current_sup_group[0][0].start

            if is_falling_back and len(sup_groups) > 0 and use_ovl_fb_sups:
                overlapping_sups = get_overlapping_sups(sup_groups[-1], sup)
                if overlapping_sups:
                    for _, ovl_idx in overlapping_sups:
                        # We want to get a part of the supervision that starts after sup.start.
                        # We need to skip words in the original text as well and then create a function that creates the supervision...
                        # Ideally, we want to use the same function that splits the text and cuts (below) to avoid duplicating the text.
                        ovl_sup = cut.supervisions[ovl_idx]
                        assert ovl_sup.speaker != sup.speaker

                        words_within_segment, alignments_within_segment, fst_word_start_time, last_word_end_time, alig_idx = select_words_within_segment(
                            ovl_sup, sup.start, _segment_end(sup.start, max_len)
                        )

                        if fst_word_start_time != -1:
                            current_sup_group.append([fastcopy(
                                ovl_sup,
                                id=f'{ovl_sup.id}-{ovl_idx}-{len(current_sup_group)}_ovl',
                                start=fst_word_start_time,
                                # duration=ovl_sup.alignment['word'][alig_idx].end - fst_word_start_time,
                                duration=last_word_end_time - fst_word_start_time,
                                text=' '.join(words_within_segment),
                                alignment={'word': alignments_within_segment},
                            ), ovl_idx])
                            assert fst_word_start_time >= 0
                            assert fst_word_start_time >= current_sup_group[0][0].start
            is_falling_back = False
        else:
            found_exceeding = False
            if sup is None and not current_sup_group:
                break

            start_idx = current_sup_group[0][1]
            for i, (s, sidx) in enumerate(current_sup_group):
                current_sup_group_flags[s.speaker] = False
                if s.end - current_sup_group[0][0].start > max_len:
                    found_exceeding = True

                    # Shift segment end by EPS to avoid floating point precision issues.
                    words_within_segment, alignments_within_segment, fst_word_start_time, last_word_end_time, alig_idx = select_words_within_segment(
                        s, current_sup_group[0][0].start, _segment_end(current_sup_group[0][0].start, max_len)
                    )

                    # Even the first aligned word exceeds the boundary.
                    if alig_idx == 0:
                        current_sup_group[i] = None
                    else:
                        current_sup_group_flags[s.speaker] = True
                        current_sup_group[i][0] = fastcopy(
                            current_sup_group[i][0],
                            id=f"{s.id}",
                            start=current_sup_group[i][0].start,
                            duration=last_word_end_time - current_sup_group[i][0].start,
                            text=' '.join(words_within_segment),
                            alignment={'word': alignments_within_segment},
                        )

                    if fallback_sup_idx == -1 or fallback_sup_idx > i:
                        fallback_sup_idx = sidx
                        is_falling_back = True

            current_sup_group = [x for x in current_sup_group if x is not None]

            sup_groups.append(current_sup_group)
            sup_group_flags.append(current_sup_group_flags)

            assert max(x[0].end for x in current_sup_group) - min(x[0].start for x in current_sup_group) <= max_len

            # If we've found and exceeding supervision, we will set the fallback idx and will move back.
            # If not, we need to add the current supervision to the new (current) group.
            current_sup_group = [] if found_exceeding else [[sup, idx]]
            current_sup_group_flags = dict()

        if fallback_sup_idx != -1 and fallback_sup_idx > start_idx:
            idx = fallback_sup_idx
            fallback_sup_idx = -1
        else:
            idx += 1

    assert len(sup_groups) == len(sup_group_flags)

    new_cuts = []
    for i, sg in enumerate(sup_groups):
        sg_start = sg[0][0].start
        sg_end = max(x[0].end for x in sg)

        new_cuts.append(
            fastcopy(
                cut,
                id=f"{cut.id}-{i}",
                supervisions=[s[0].with_offset(-sg_start) for s in sg],
                start=cut.start + sg_start,
                duration=sg_end - sg_start,
                custom={
                    "per_spk_flags": sup_group_flags[i],
                },
            )
        )
        assert all(x.start >= 0 for x in new_cuts[-1].supervisions)
        assert new_cuts[-1].duration <= max_len

    return new_cuts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input manifest"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output manifest"
    )
    parser.add_argument(
        "--max_len", type=int, default=30, help="Max length of the cut in seconds"
    )
    parser.add_argument(
        "--num_jobs", type=int, default=8, help="Number of parallel jobs"
    )

    args = parser.parse_args()

    cset = load_manifest(args.input)
    _prepare_segmented_data(cuts=cset,
                            output_path=args.output,
                            max_segment_duration=args.max_len,
                            num_jobs=args.num_jobs)

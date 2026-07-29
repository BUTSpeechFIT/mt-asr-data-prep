import argparse
import re
from collections import defaultdict
from typing import List, Optional, Tuple

from lhotse import fastcopy, load_manifest
from lhotse import CutSet

from pre_segment_using_alignments import EPS, WORD_ALIGNMENT_KEY, select_words_within_segment


def _session_id(recording_id: str, prefix_to_strip: Optional[str], pattern: Optional[str]) -> str:
    sid = recording_id
    if prefix_to_strip and sid.startswith(prefix_to_strip):
        sid = sid[len(prefix_to_strip):]
    if pattern:
        m = re.match(pattern, sid)
        if not m:
            raise ValueError(f"session_id_regex {pattern!r} did not match recording_id {recording_id!r}")
        sid = m.group(1) if m.groups() else m.group(0)
    return sid


def _free_intervals(seg_start: float, seg_end: float, busy: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sub-intervals of [seg_start, seg_end] that don't overlap any interval in busy."""
    clipped = []
    for b_start, b_end in busy:
        s, e = max(seg_start, b_start), min(seg_end, b_end)
        if s < e:
            clipped.append((s, e))
    if not clipped:
        return [(seg_start, seg_end)]
    clipped.sort()
    merged = [clipped[0]]
    for s, e in clipped[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    free = []
    cur = seg_start
    for s, e in merged:
        if cur < s:
            free.append((cur, s))
        cur = max(cur, e)
    if cur < seg_end:
        free.append((cur, seg_end))
    return free


def _build_reference_index(reference_supset, prefix_to_strip, session_id_regex):
    by_session = defaultdict(list)
    for s in reference_supset:
        sid = _session_id(s.recording_id, prefix_to_strip, session_id_regex)
        by_session[sid].append(s)
    return by_session


def _group_supervisions(supervisions, max_pause):
    """
    Merge consecutive supervisions (same cut, so already the same speaker) into groups
    separated by gaps > max_pause, concatenating text/alignment. This is what lets
    cross-talk-free stretches spanning several original utterances survive as one
    longer segment instead of one-fragment-per-original-utterance -- the busy-interval
    subtraction below still re-splits anywhere cross-talk actually falls inside a
    group, so this never bridges over a contaminated gap.
    """
    if not supervisions:
        return []
    sups = sorted(supervisions, key=lambda s: s.start)
    groups = [[sups[0]]]
    for s in sups[1:]:
        if s.start - groups[-1][-1].end <= max_pause:
            groups[-1].append(s)
        else:
            groups.append([s])

    merged = []
    for i, group in enumerate(groups):
        if len(group) == 1:
            merged.append(group[0])
            continue
        start = group[0].start
        end = max(s.end for s in group)
        text = " ".join(s.text for s in group)
        alignment = None
        if all(s.alignment and s.alignment.get(WORD_ALIGNMENT_KEY) for s in group):
            alignment = {
                WORD_ALIGNMENT_KEY: [a for s in group for a in s.alignment[WORD_ALIGNMENT_KEY]]
            }
        merged.append(fastcopy(
            group[0], id=f"{group[0].id}-grp{i}", start=start, duration=end - start,
            text=text, alignment=alignment,
        ))
    return merged


def _split_supervision_by_crosstalk(sup, busy_intervals, min_segment_duration):
    """
    Return a list of new SupervisionSegments covering only the sub-intervals of `sup`
    that don't overlap any interval in busy_intervals (other speakers' segments in the
    same session). Uses word-level alignment to trim text precisely; segments without
    alignment are kept only if entirely clean (can't be trimmed without word timings).
    Returned segments keep `sup`'s original (cut-relative) time frame.
    """
    seg_start, seg_end = sup.start, sup.end
    free = [
        (s, e) for s, e in _free_intervals(seg_start, seg_end, busy_intervals)
        if e - s >= min_segment_duration
    ]
    if not free:
        return []

    has_alignment = bool(sup.alignment and sup.alignment.get(WORD_ALIGNMENT_KEY))
    if not has_alignment:
        # No word timings to trim text with -- only keep the segment if it survived
        # completely untouched (single free interval spanning the whole thing).
        if len(free) == 1 and free[0][0] <= seg_start + 1e-6 and free[0][1] >= seg_end - 1e-6:
            return [sup]
        return []

    new_sups = []
    for i, (fs, fe) in enumerate(free):
        words, aligns, fst_start, last_end, _ = select_words_within_segment(sup, fs, fe)
        if fst_start == -1:
            continue
        new_sups.append(
            fastcopy(
                sup,
                id=f"{sup.id}-ct{i}",
                start=fst_start,
                duration=last_end - fst_start,
                text=" ".join(words),
                alignment={WORD_ALIGNMENT_KEY: aligns},
            )
        )
    return new_sups


def _split_long_segment(sup, max_duration):
    """
    Split `sup` into consecutive chunks no longer than max_duration, splitting only at
    word boundaries (never mid-word). Reuses select_words_within_segment -- the same
    building block already used for cross-talk trimming and for the stochastic-offset
    windowing in pre_segment_using_alignments.py -- so unaligned filler tokens (e.g.
    <FILL/>) that appear in the text without their own alignment entry are still
    stitched into the right chunk instead of being dropped or misplaced. Returned
    segments keep `sup`'s original (cut-relative) time frame. Requires alignment;
    callers should only reach here for segments that already have it (unaligned
    over-long segments are dropped, same as before).
    """
    chunks = []
    cursor = sup.start
    while cursor < sup.end - EPS:
        window_end = min(cursor + max_duration - EPS, sup.end)
        words, aligns, fst_start, last_end, _ = select_words_within_segment(sup, cursor, window_end)
        if fst_start == -1:
            # No aligned word starts in this window (e.g. a long silence); skip ahead.
            cursor = window_end + EPS
            continue
        chunks.append(
            fastcopy(
                sup,
                id=f"{sup.id}-split{len(chunks)}",
                start=fst_start,
                duration=last_end - fst_start,
                text=" ".join(words),
                alignment={WORD_ALIGNMENT_KEY: aligns},
            )
        )
        cursor = last_end
    return chunks


def _rebase_supervision(sup):
    """Re-anchor `sup` (and its word alignment, if any) to start at 0, so it can become
    the sole supervision of a cut trimmed exactly to `sup`'s own span."""
    offset = sup.start
    new_alignment = sup.alignment
    if sup.alignment and sup.alignment.get(WORD_ALIGNMENT_KEY):
        new_alignment = {
            WORD_ALIGNMENT_KEY: [a.with_offset(-offset) for a in sup.alignment[WORD_ALIGNMENT_KEY]]
        }
    return fastcopy(sup, start=0.0, alignment=new_alignment)


def main(input_manifest, reference_supset_path, output_manifest,
         reference_prefix_to_strip, session_id_regex, reference_session_id_regex,
         min_segment_duration, max_segment_duration, max_pause):
    cuts = load_manifest(input_manifest).to_eager()
    reference_supset = load_manifest(reference_supset_path)
    ref_by_session = _build_reference_index(reference_supset, reference_prefix_to_strip, reference_session_id_regex)

    def emit(cut, final_sup):
        new_cuts.append(
            fastcopy(
                cut,
                id=final_sup.id,
                start=cut.start + final_sup.start,
                duration=final_sup.duration,
                supervisions=[_rebase_supervision(final_sup)],
            )
        )

    n_in, n_clean, n_out, n_split, n_dropped_unaligned_long = 0, 0, 0, 0, 0
    new_cuts = []
    for cut in cuts:
        session_id = cut.custom.get("session_id") if cut.custom else None
        if session_id is None:
            session_id = _session_id(cut.recording_id, None, session_id_regex)

        session_sups = ref_by_session.get(session_id, [])
        n_in += len(cut.supervisions)
        for sup in _group_supervisions(cut.supervisions, max_pause):
            busy = [(s.start, s.end) for s in session_sups if s.speaker != sup.speaker]
            for clean_sup in _split_supervision_by_crosstalk(sup, busy, min_segment_duration):
                n_clean += 1
                if clean_sup.duration <= max_segment_duration:
                    emit(cut, clean_sup)
                    n_out += 1
                    continue
                # Longer than max_segment_duration (grouping can produce these) --
                # split at word boundaries into <=max_segment_duration pieces instead
                # of throwing the clean audio away.
                if not (clean_sup.alignment and clean_sup.alignment.get(WORD_ALIGNMENT_KEY)):
                    n_dropped_unaligned_long += 1
                    continue
                pieces = _split_long_segment(clean_sup, max_segment_duration)
                for piece in pieces:
                    if piece.duration < min_segment_duration:
                        continue
                    emit(cut, piece)
                    n_out += 1
                n_split += 1

    print(f"filter_crosstalk_segments: {n_in} input supervisions -> {n_clean} clean "
          f"sub-segments ({n_split} split for exceeding {max_segment_duration}s, "
          f"{n_dropped_unaligned_long} dropped for exceeding it without alignment) "
          f"-> {n_out} per-utterance cuts")

    CutSet.from_cuts(new_cuts).to_file(output_manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mitigate cross-talk bleed in close-talk/IHM recordings: given a "
        "reference multi-speaker supervision set for the same sessions (e.g. the sdm/mdm "
        "supervisions), trim each single-speaker cut's segments down to the sub-intervals "
        "where no other speaker in the same session was also talking (using word-level "
        "alignment to trim precisely). Emits one cut per surviving utterance, trimmed "
        "exactly to that utterance's span -- no windowing/grouping afterward, so no "
        "unannotated or cross-talk-contaminated audio can leak in around it."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Single-speaker (IHM/close-talk) cutset to filter")
    parser.add_argument("--reference_supset", type=str, required=True,
                        help="Multi-speaker supervision manifest for the same sessions (e.g. sdm/mdm)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--reference_prefix_to_strip", type=str, default=None,
                        help="Prefix to strip from the reference supervisions' recording_id "
                        "before matching sessions (e.g. 'sdm_' if it went through add_prefix.py)")
    parser.add_argument("--session_id_regex", type=str, default=None,
                        help="Regex (first capture group, or whole match if none) applied to "
                        "the input cut's recording_id to derive its session id. Only used when "
                        "the cut has no cut.custom['session_id'] already set.")
    parser.add_argument("--reference_session_id_regex", type=str, default=None,
                        help="Regex applied to the reference supervisions' recording_id "
                        "(after stripping --reference_prefix_to_strip) to derive session id")
    parser.add_argument("--min_segment_duration", type=float, default=0.3,
                        help="Drop clean sub-segments shorter than this (seconds)")
    parser.add_argument("--max_segment_duration", type=float, default=30.0,
                        help="Split clean sub-segments longer than this (seconds) at word "
                        "boundaries into pieces this length or shorter; segments without "
                        "alignment that exceed it are dropped (can't split precisely)")
    parser.add_argument("--max_pause", type=float, default=2.0,
                        help="Merge consecutive same-speaker supervisions separated by a gap "
                        "no larger than this (seconds) before cross-talk splitting, so "
                        "clean stretches spanning multiple original utterances survive as "
                        "one longer segment instead of one fragment per original utterance")

    args = parser.parse_args()

    main(args.input, args.reference_supset, args.output,
        args.reference_prefix_to_strip, args.session_id_regex, args.reference_session_id_regex,
        args.min_segment_duration, args.max_segment_duration, args.max_pause)

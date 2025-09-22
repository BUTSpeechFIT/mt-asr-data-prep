import argparse
import logging
from functools import partial
from pathlib import Path
from typing import Optional

import lhotse
from intervaltree import IntervalTree
from lhotse import CutSet, SupervisionSet, RecordingSet
from lhotse import fastcopy, load_manifest
from lhotse.lazy import LazyFlattener, LazyMapper

# TODO: @Dominik Klement - please refactor this :(((((

def _prepare_segmented_data(
        recordings: RecordingSet,
        supervisions: SupervisionSet,
        split: str,
        output_path: Optional[str] = None,
        return_close_talk: bool = False,
        return_multichannel: bool = False,
        max_segment_duration=30.0,
        num_jobs=1,
) -> lhotse.CutSet:
    output_path = Path(output_path) if output_path else None

    logging.info("Trimming to alignments")
    # new_sups = SupervisionSet.from_segments(LazyFlattener(LazyMapper(supervisions, _trim_to_alignments)))
    cuts: CutSet = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
    # cuts = cuts.transform_text(lambda text: text.lower())  # TODO: IMHO, it's not necessary to force lower-case (make it optional?)

    logging.info("Trimming to groups with max_pause=2s")
    cuts = cuts.trim_to_supervision_groups(max_pause=2, num_jobs=num_jobs).to_eager()
    for c in cuts:
        for s in c.supervisions:
            if 'word' in s.alignment:
                s.alignment['word'] = [a.with_offset(-c.start) for a in s.alignment['word']]
    logging.info("Windowing the overllapping segments to max 30s")
    cuts = split_overlapping_segments(cuts, max_segment_duration=max_segment_duration, num_jobs=num_jobs).to_eager()

    if output_path is not None:
        # Covers all cases: .json, .jsonl, .jsonl.gz
        if '.json' in str(output_path):
            save_path = output_path
        else:
            suffix = f"{'_multichannel' if return_multichannel else '_singlechannel'}{'_closetalk' if return_close_talk else ''}"
            fname = f"notsofar_{split}_segmented_{int(max_segment_duration)}s{suffix}_cuts.jsonl.gz"
            save_path = output_path / fname

        logging.info("Saving cuts to %s", save_path)
        cuts.to_file(save_path)

    return cuts


def _trim_to_alignments(sup: lhotse.SupervisionSegment):
    if sup.alignment is None:
        return [sup]
    alis: list[lhotse.supervision.AlignmentItem] = sup.alignment['word']
    alis.sort(key=lambda ali: (ali.start, ali.end))
    if len(alis) == 0:
        return [sup]
    new_sups = []
    for i, ali in enumerate(alis):
        new_sups.append(
            lhotse.fastcopy(sup, id=f"{sup.id}-{i}", start=ali.start, duration=ali.duration, text=ali.symbol,
                            alignment=None))
    return new_sups


def split_overlapping_segments(cutset: CutSet, max_segment_duration=30, num_jobs=1):
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
    else:
        from lhotse.manipulation import split_parallelize_combine
        return split_parallelize_combine(num_jobs, cutset, _split_overlapping_segments_single,
                                         max_len=max_segment_duration)


def _split_overlapping_segments_single(cutset: CutSet, max_len=30):
    split_fn = partial(_split_cut_perseg, max_len=max_len)
    return CutSet(LazyFlattener(LazyMapper(cutset, split_fn))).to_eager()


ALIGNMENT_WORD_MAP = {
    'mm-hmm': 'mmm',
}


def get_overlapping_sups(prev_group, sup):
    return [s for s in prev_group if s[0].start <= sup.start < s[0].end and sup.id not in s[0].id]


def _split_cut_perseg(cut: lhotse.cut.Cut, max_len=30, use_ovl_fb_sups=True):
    if len(cut.supervisions) == 0:
        return []

    if cut.duration < max_len:
        return [cut]

    sups = sorted(cut.supervisions, key=lambda s: s.start)

    sup_groups = []
    # Flag = True <=> speaker's utterance is unfinished.
    sup_group_flags = []

    current_sup_group = [[fastcopy(sups[0], id=f'{sups[0].id}-{0}-{0}'), 0]]
    current_sup_group_flags = dict()
    fallback_sup_idx = -1
    is_falling_back = False

    idx = 1
    while idx < len(sups) + 1:
        # print(idx)
        sup = sups[idx] if idx < len(sups) else None
        # We need to add all supervisions that start before the end fo current max 30s long segment. Then, we need to post-process the short ones and return back.
        if sup is not None and (not current_sup_group or sup.start - current_sup_group[0][0].start < max_len):
            current_sup_group.append([fastcopy(sup, id=f'{sup.id}-{idx}-{len(current_sup_group)}'), idx])
            assert sup.start >= current_sup_group[0][0].start

            if is_falling_back and len(sup_groups) > 0 and use_ovl_fb_sups:
                overlapping_sups = get_overlapping_sups(sup_groups[-1], sup)
                if overlapping_sups:
                    for _, ovl_idx in overlapping_sups:
                        # We want to get a part of supervision that start after sup.start.
                        # We need to skip words in the original text as well and then create a function that creates the sup...
                        # Ideally, we want to use the same function that splits the text and cuts (below) to avoid duplicating the text.
                        ovl_sup = cut.supervisions[ovl_idx]
                        assert ovl_sup.speaker != sup.speaker

                        words_within_segment = []
                        fst_word_start_time = -1
                        alig_idx = 0
                        current_alig = ovl_sup.alignment['word'][alig_idx] if alig_idx < len(
                            ovl_sup.alignment['word']) else None
                        alig_symbol_split = [
                            x.translate(str.maketrans('', '', "!\"#$%&()*+,./:;<=>?@[\\]^_`'{|}~")).lower() for x in
                            current_alig.symbol.split()] if current_alig is not None else None

                        if 'word' in ovl_sup.alignment and len(ovl_sup.alignment['word']) > 0:
                            for w in ovl_sup.text.split():
                                if fst_word_start_time != -1:
                                    words_within_segment.append(w)
                                    continue

                                adjusted_w = w.translate(
                                    str.maketrans('', '', "!\"#$%&()*+,./:;<=>?@[\\]^_`'{|}~")).lower()
                                if current_alig is not None and (adjusted_w == alig_symbol_split[0] or (
                                        adjusted_w in ALIGNMENT_WORD_MAP and ALIGNMENT_WORD_MAP[adjusted_w] ==
                                        alig_symbol_split[0])):
                                    if current_alig.start < sup.start:
                                        alig_idx += 1
                                        current_alig = ovl_sup.alignment['word'][alig_idx] if alig_idx < len(
                                            ovl_sup.alignment['word']) else None
                                        alig_symbol_split = [x.translate(
                                            str.maketrans('', '', "!\"#$%&()*+,./:;<=>?@[\\]^_`'{|}~")).lower() for x in
                                                             current_alig.symbol.split()] if current_alig is not None else None
                                    else:
                                        if fst_word_start_time == -1:
                                            fst_word_start_time = current_alig.start
                                            words_within_segment.append(w)
                                else:
                                    # We skip the words that do not have alignments.
                                    if fst_word_start_time != -1:
                                        words_within_segment.append(w)

                        if fst_word_start_time != -1:
                            current_sup_group.append([fastcopy(
                                ovl_sup,
                                id=f'{ovl_sup.id}-{ovl_idx}-{len(current_sup_group)}_ovl',
                                start=fst_word_start_time,
                                duration=ovl_sup.alignment['word'][alig_idx].end - fst_word_start_time,
                                text=' '.join(words_within_segment),
                                alignment={'word': ovl_sup.alignment['word'][alig_idx:]},
                            ), ovl_idx])
                            assert fst_word_start_time >= 0
                            assert fst_word_start_time >= current_sup_group[0][0].start
            is_falling_back = False
        else:
            found_exceeding = False
            if sup == None and not current_sup_group:
                break

            start_idx = current_sup_group[0][1]
            for i, (s, sidx) in enumerate(current_sup_group):
                current_sup_group_flags[s.speaker] = False
                if s.end - current_sup_group[0][0].start > max_len:
                    found_exceeding = True
                    # Fix this supervision and add only the words that are inside the segment & are not overlapped with other words.
                    words_within_segment = []
                    last_word_end_time = s.start
                    alig_idx = 0
                    current_alig = s.alignment['word'][alig_idx] if alig_idx < len(s.alignment['word']) else None
                    alig_symbol_split = [x.translate(str.maketrans('', '', "!\"#$%&()*+,./:;<=>?@[\\]^_`'{|}~")).lower()
                                         for x in current_alig.symbol.split()] if current_alig is not None else None

                    if 'word' in s.alignment and len(s.alignment['word']) > 0:
                        for w in s.text.split():
                            adjusted_w = w.translate(str.maketrans('', '', "!\"#$%&()*+,./:;<=>?@[\\]^_`'{|}~")).lower()

                            if current_alig is not None and (adjusted_w == alig_symbol_split[0] or (
                                    adjusted_w in ALIGNMENT_WORD_MAP and ALIGNMENT_WORD_MAP[adjusted_w] ==
                                    alig_symbol_split[0])):
                                if current_alig.end - current_sup_group[0][0].start > max_len:
                                    break
                                alig_symbol_split.pop(0)
                                words_within_segment.append(w)
                                last_word_end_time = current_alig.start + current_alig.duration
                                if not alig_symbol_split:
                                    alig_idx += 1
                                    current_alig = s.alignment['word'][alig_idx] if alig_idx < len(
                                        s.alignment['word']) else None
                                    alig_symbol_split = current_alig.symbol.translate(str.maketrans('', '',
                                                                                                    "!\"#$%&()*+,./:;<=>?@[\\]^_`'{|}~")).lower().split() if current_alig is not None else None
                            else:
                                # Special symbol that is not aligned -> appending to words and assuming the same timing info as the previous one.
                                words_within_segment.append(w)
                                print('adding without alignment', w, f'""{s.text}""',
                                      [a.symbol for a in s.alignment['word']])

                    # Even the first aligned word exceeds the boundary.
                    if alig_idx == 0:
                        current_sup_group[i] = None
                    else:
                        current_sup_group_flags[s.speaker] = True
                        current_sup_group[i][0] = fastcopy(
                            current_sup_group[i][0],
                            id=f'{s.id}',
                            start=current_sup_group[i][0].start,
                            duration=last_word_end_time - current_sup_group[i][0].start,
                            text=' '.join(words_within_segment),
                            alignment={'word': current_sup_group[i][0].alignment['word'][:alig_idx]},
                        )

                    if fallback_sup_idx == -1 or fallback_sup_idx > i:
                        fallback_sup_idx = sidx
                        is_falling_back = True

            current_sup_group = [x for x in current_sup_group if x is not None]

            sup_groups.append(current_sup_group)
            sup_group_flags.append(current_sup_group_flags)

            assert max(x[0].end for x in current_sup_group) - current_sup_group[0][0].start <= max_len

            # If we've found and exceeding supervision, we will set the fallback idx and will move back.
            # If not, we need to add the current supervision to the new (current) group.
            current_sup_group = [] if found_exceeding else [[sup, idx]]
            current_sup_group_flags = dict()

        if fallback_sup_idx != -1 and fallback_sup_idx > start_idx:
            idx = fallback_sup_idx
            # print('FALLING BACK TO SUP:', idx, words_within_segment, alig_idx)
            fallback_sup_idx = -1
        else:
            idx += 1

    assert len(sup_groups) == len(sup_group_flags)

    new_cuts = []
    for i, sg in enumerate(sup_groups):
        sg_start = sg[0][0].start
        sg_end = max(x[0].end for x in sg)

        new_cuts.append(fastcopy(
            cut,
            id=f'{cut.id}-{i}',
            supervisions=[s[0].with_offset(-sg_start) for s in sg],
            start=cut.start + sg_start,
            duration=sg_end - sg_start,
            custom={
                'per_spk_flags': sup_group_flags[i],
            }
        ))
        assert all(x.start >= 0 for x in new_cuts[-1].supervisions)
        assert new_cuts[-1].duration <= max_len

    return new_cuts


"""
Adjusting the function.
We know that the cut was already pre-split to groups, meaning the last supervision is at least 2s apart from the next one,
indicating a timestamp in Whisper inference.
This means that we can automatically flag last cut that is created by splitting as a "silence-afterwards" cut.
We know that if there was a silence segemnts, it would've been split already by the preceeding processing.
Hence, it's sufficient to flag the last cut only and skip the rest and flag with "false" flag afterwards.
"""


def _split_cut(cut: lhotse.cut.Cut, max_len=30):
    if len(cut.supervisions) == 0:
        return []

    orig_cut = cut

    ss_areas = _get_single_spk_audio_intervals(cut)

    t = IntervalTree()
    for s, e in ss_areas:
        t[s:e] = 'x'
    word_end_t = IntervalTree()
    for s in cut.supervisions:
        if t.at(s.end):
            # We can't add point since it's an interval tree. As we want to do intersection with another int. tree, we can't use some balanced one only.
            word_end_t[s.end - 1e-4:s.end + 1e-4] = 'x'

    sup_groups = []
    current_sup_group = [cut.supervisions[0]]

    for i, s in enumerate(cut.supervisions[1:]):
        # If the current word endpoint is in single-spk int, we can split, if not, we need to unconditionally add it to the current sup group
        if not current_sup_group or (not word_end_t.at(s.end) and s.end - current_sup_group[0].start <= max_len):
            current_sup_group.append(s)
        else:
            # We know that current word end point is not overlapped with any other word spoken by other speakers, so we can decide if we want to split.
            # The issue here is that we don't know when not to split - i.e. we could've split the current word but we didn't as we didn't reach the max_len limit,
            # but all the following supervisions are overlapped for the next 10s. If we'd split before, we could've put all the overlapped ones into a single group.
            if len(current_sup_group) > 0:
                other_possible_split_points = word_end_t[s.end + 1e-3:current_sup_group[
                                                                          0].start + max_len]  # We need to adjust the interval tree using the endpoints.
                if i == len(cut.supervisions[1:]) - 1:
                    other_possible_split_points = True

                # It may happen that the rest of the split is overlapped, but if we know that we cannot exceed the max_len,
                #  we set other_possible_split_points = True which means that the current group is not going to be split in the for loop
                #  but is going to be appended to the sup_groups after the forloop ends.

                # This is not correct: We need to check u
                if cut.duration - current_sup_group[0].start < max_len:
                    other_possible_split_points = True
            else:
                other_possible_split_points = True

            if len(current_sup_group) > 0 and s.end - current_sup_group[0].start >= max_len:
                sup_groups.append(current_sup_group)
                current_sup_group = [s]
            elif not other_possible_split_points:
                current_sup_group.append(s)
                sup_groups.append(current_sup_group)
                current_sup_group = []
            else:
                current_sup_group.append(s)

    if current_sup_group:
        sup_groups.append(current_sup_group)

    sup_groups = [sorted(sups, key=lambda s: (s.start, s.end)) for sups in sup_groups]
    start_groups = [min(s.start for s in sups) for sups in sup_groups]
    end_groups = [max(s.end for s in sups) for sups in sup_groups]
    cuts = [
        lhotse.fastcopy(cut, id=f"{cut.id}-{i}", supervisions=[s.with_offset(-start) for s in sups],
                        start=cut.start + start, duration=end - start)
        for i, (sups, start, end) in enumerate(zip(sup_groups, start_groups, end_groups))
    ]

    per_spk_supervisions = {}
    for sup in orig_cut.supervisions:
        if sup.speaker not in per_spk_supervisions:
            per_spk_supervisions[sup.speaker] = []
        per_spk_supervisions[sup.speaker].append(sup)

    for spk, sups in per_spk_supervisions.items():
        per_spk_supervisions[spk] = sorted(sups, key=lambda s: (s.start, s.end))

    per_spk_supervisions_idxes = {spk: 0 for spk in per_spk_supervisions.keys()}
    for c in cuts:
        cut_spks = CutSet.from_cuts([c]).speakers
        per_spk_flags = {spk: False for spk in cut_spks}
        for spk in cut_spks:
            last_spk_sup_within_cut = None
            for i, sup in enumerate(per_spk_supervisions[spk][per_spk_supervisions_idxes[spk]:]):
                if c.start + sup.start < c.end:
                    last_spk_sup_within_cut = sup
                    per_spk_supervisions_idxes[spk] += 1
                else:
                    break

            # print(orig_cut.start + per_spk_supervisions[spk][per_spk_supervisions_idxes[spk]].start - (c.start + last_spk_sup_within_cut.end))

            # 2 seconds
            is_next_sup_close = last_spk_sup_within_cut is not None and len(per_spk_supervisions[spk]) > \
                                per_spk_supervisions_idxes[spk] and (orig_cut.start + per_spk_supervisions[spk][
                per_spk_supervisions_idxes[spk]].start - (c.start + last_spk_sup_within_cut.end)) < 2
            per_spk_flags[spk] = not is_next_sup_close

        c.custom = {
            'per_spk_flags': per_spk_flags
        }

    return cuts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the input manifest')
    parser.add_argument('--output', type=str, required=True, help='Path to the output manifest')
    parser.add_argument('--max_len', type=int, default=30, help='Max length of the cut in seconds')
    parser.add_argument('--num_jobs', type=int, default=8, help='Number of parallel jobs')

    args = parser.parse_args()

    cset = load_manifest(args.input)
    rs, ss, _ = cset.decompose()

    _prepare_segmented_data(recordings=rs,
                            supervisions=ss,
                            split=None,
                            output_path=args.output,
                            return_close_talk=False,
                            return_multichannel=False,
                            max_segment_duration=args.max_len,
                            num_jobs=args.num_jobs)

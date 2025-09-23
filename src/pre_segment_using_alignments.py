import argparse
import logging
from functools import partial
from pathlib import Path
from typing import Optional

import lhotse
from lhotse import CutSet
from lhotse import fastcopy, load_manifest
from lhotse.lazy import LazyFlattener, LazyMapper

# TODO: @Dominik Klement - please refactor this :(((((


def filter_punctuation_aligments(cut):
    new_sups = []
    for sup in cut.supervisions:
        new_aligments = [word for word in sup.alignment['word'] if  len(word.symbol.translate(str.maketrans('', '', "!\"#$%&()*+,./:;<=>?@[\\]^_`'{|}~"))) > 0]
        sup.alignment['word'] = new_aligments
        new_sups.append(sup)
    cut.supervisions = new_sups
    return cut


def _prepare_segmented_data(
        cuts: CutSet,
        split: str,
        output_path: Optional[str] = None,
        return_close_talk: bool = False,
        return_multichannel: bool = False,
        max_segment_duration=30.0,
        num_jobs=1,
) -> lhotse.CutSet:
    output_path = Path(output_path) if output_path else None
    logging.info("Trimming to groups with max_pause=2s")
    cuts = cuts.trim_to_supervision_groups(max_pause=2, num_jobs=num_jobs).to_eager()
    for c in cuts:
        for s in c.supervisions:
            if 'word' in s.alignment:
                s.alignment['word'] = [a.with_offset(-c.start) for a in s.alignment['word']]
    logging.info("Windowing the overlapping segments to max 30s")

    cuts = cuts.map(filter_punctuation_aligments).to_eager()

    cuts = split_overlapping_segments(cuts, max_segment_duration=max_segment_duration, num_jobs=num_jobs).to_eager()

    logging.info("Saving the output")
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the input manifest')
    parser.add_argument('--output', type=str, required=True, help='Path to the output manifest')
    parser.add_argument('--max_len', type=int, default=30, help='Max length of the cut in seconds')
    parser.add_argument('--num_jobs', type=int, default=8, help='Number of parallel jobs')

    args = parser.parse_args()

    cset = load_manifest(args.input)
    _prepare_segmented_data(cuts=cset,
                            split=None,
                            output_path=args.output,
                            return_close_talk=False,
                            return_multichannel=False,
                            max_segment_duration=args.max_len,
                            num_jobs=args.num_jobs)

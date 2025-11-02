import argparse
import random
from functools import reduce

import numpy as np

import lhotse
from lhotse import CutSet
from lhotse.cut.mixed import MixedCut, MixTrack


def mix_two_recordings(len_1, len_2, allowed_pause):
    rec2_offset = np.random.uniform(
        low=-len_1 - len_2 - allowed_pause, high=allowed_pause
    )
    # we start with rec1 followed by rec2 -> positive value means rec2 is offset by inserting pause after rec1
    # if -len1 is sampled rec1 is fully overlapped with rec2
    # if -len_1-len_2-allowed_pause is sampled first goes rec2 followed by pause and rec1
    if -rec2_offset <= len_1:
        return 0, len_1 + rec2_offset
    else:
        return -(len_1 + rec2_offset), 0


def sample_offsets(durations, allowed_pause, max_len):
    # first we pair-wise mix other recordings
    N = len(durations)

    prev_rec_dur = durations[0]
    offsets = np.zeros(N)
    for i in range(1, N):
        other_rec_dur = durations[i]
        offset_1, offset_2 = mix_two_recordings(
            prev_rec_dur, other_rec_dur, allowed_pause
        )
        offsets[:] += offset_1
        offsets[i] = offset_2
        prev_rec_dur = max(offset_1 + prev_rec_dur, offset_2 + other_rec_dur)

    for i in range(N):
        if offsets[i] + durations[i] > max_len:
            offsets[i] = max_len - durations[i]

    return offsets


def generate_mixture(cuts, max_len, allowed_pause):
    lens = [cut.duration for cut in cuts]

    offsets = sample_offsets(lens, allowed_pause=allowed_pause, max_len=max_len)

    tracks = [MixTrack(cut=cut, offset=offset) for cut, offset in zip(cuts, offsets)]

    mixture_id = "-".join([f"{track.cut.id}_{track.offset:.2f}_" for track in tracks])
    mixture = MixedCut(id=mixture_id, tracks=tracks)

    return mixture


def get_cut_spks(cut):
    spks = set()
    for suppervision in cut.supervisions:
        spks.add(suppervision.speaker)
    return sorted(spks)


def main(
    cutsets_to_mix, output_path, num_mixtures, max_len, num_speakers, allowed_pause
):
    csets = [lhotse.load_manifest(cutset) for cutset in cutsets_to_mix]
    cset = reduce(lambda a, b: a + b, csets)
    cset = cset.filter(lambda cut: cut.duration <= max_len)
    per_speaker_samples = {}
    for cut in cset:
        speakers = get_cut_spks(cut)
        for speaker in speakers:
            per_speaker_samples[speaker] = [*per_speaker_samples.get(speaker, []), cut]

    speakers = sorted(per_speaker_samples.keys())
    for speaker in speakers:
        per_speaker_samples[speaker] = CutSet.from_cuts(per_speaker_samples[speaker])

    new_cuts = []
    for _ in range(num_mixtures):
        sampled_speakers = random.sample(speakers, num_speakers)
        cuts_to_mix = [
            per_speaker_samples[speaker].sample() for speaker in sampled_speakers
        ]
        new_cuts.append(
            generate_mixture(cuts_to_mix, max_len=max_len, allowed_pause=allowed_pause)
        )

    final_cset = CutSet.from_cuts(new_cuts)
    final_cset.to_file(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifests", nargs="+", required=True)
    parser.add_argument("--output_manifest", type=str, required=True)
    parser.add_argument("--max_len", type=float, default=30.0)
    parser.add_argument("--num_speakers", type=int, default=3)
    parser.add_argument("--num_mixtures", type=int, default=10000)
    parser.add_argument("--allowed_pause", type=float, default=2.0)

    args = parser.parse_args()

    main(
        cutsets_to_mix=args.input_manifests,
        output_path=args.output_manifest,
        num_mixtures=args.num_mixtures,
        max_len=args.max_len,
        num_speakers=args.num_speakers,
        allowed_pause=args.allowed_pause,
    )

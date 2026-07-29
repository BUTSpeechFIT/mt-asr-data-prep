import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from lhotse import CutSet, fastcopy, load_manifest
from lhotse.cut import MultiCut


def _split_cut(cut, audio_dir: Path):
    """
    Split a MultiCut into one MonoCut per channel (e.g. one per speaker's individual
    headset mic), each saved to its own audio file. Unlike downmixing, no channels are
    summed -- each output MonoCut only contains that channel's own audio and the
    supervisions already scoped to it by lhotse's per-channel supervision assignment.

    Channels with no supervisions (e.g. an unused mic slot) are dropped. Non-MultiCuts
    pass through unchanged.
    """
    if not isinstance(cut, MultiCut):
        return [cut]

    results = []
    for mono_cut in cut.to_mono(mono_downmix=False):
        if not mono_cut.supervisions:
            continue
        saved = mono_cut.save_audio(audio_dir / f"{mono_cut.id}.flac")
        # Tag with the original (pre-split) session id so downstream steps (e.g.
        # cross-talk filtering) can match this channel back to its full multi-speaker
        # timeline without having to guess it back out of the split cut's own id.
        saved = fastcopy(saved, custom={**(saved.custom or {}), "session_id": cut.recording_id})
        results.append(saved)
    return results


def main(input_manifest: str, output_manifest: str, audio_dir: str, num_jobs: int):
    audio_dir_path = Path(audio_dir)
    audio_dir_path.mkdir(parents=True, exist_ok=True)

    cuts = load_manifest(input_manifest)
    worker = partial(_split_cut, audio_dir=audio_dir_path)

    if num_jobs <= 1:
        new_cuts = [c for cut in cuts for c in worker(cut)]
    else:
        with ProcessPoolExecutor(max_workers=num_jobs) as ex:
            new_cuts = [c for cuts_ in ex.map(worker, cuts) for c in cuts_]

    CutSet.from_cuts(new_cuts).to_file(output_manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split multi-channel cuts (e.g. AMI IHM, one channel per speaker's "
        "headset mic) into one single-channel cut per channel, each saved as its own "
        "audio file. Unlike downmix_recordings.py, channels are NOT summed."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input cutset manifest"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output cutset manifest"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory where the per-channel mono audio files will be stored",
    )
    parser.add_argument(
        "--num_jobs", type=int, default=1, help="Number of parallel workers"
    )

    args = parser.parse_args()

    main(args.input, args.output, args.audio_dir, args.num_jobs)

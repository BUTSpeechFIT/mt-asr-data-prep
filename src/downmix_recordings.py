import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from lhotse import CutSet, fastcopy, load_manifest
from lhotse.cut import MultiCut


def _downmix_cut(cut, audio_dir: Path, id_suffix: str):
    if not isinstance(cut, MultiCut):
        return cut

    mono_cut = cut.to_mono(mono_downmix=True)
    mono_cut = mono_cut.save_audio(audio_dir / f"{cut.id}{id_suffix}.flac")

    # to_mono() downmixes per-channel mono cuts and then merges their supervisions,
    # so a supervision that originally spanned all channels (channel=[0..N-1], the
    # common case for AISHELL-4 / AliMeeting) gets duplicated once per channel.
    deduped_supervisions = list({s.id: s for s in mono_cut.supervisions}.values())

    new_id = f"{cut.id}{id_suffix}"
    return fastcopy(
        mono_cut,
        id=new_id,
        recording=fastcopy(mono_cut.recording, id=new_id),
        supervisions=[
            fastcopy(s, id=f"{s.id}{id_suffix}", recording_id=new_id)
            for s in deduped_supervisions
        ],
    )


def main(input_manifest: str, output_manifest: str, audio_dir: str, id_suffix: str, num_jobs: int):
    audio_dir_path = Path(audio_dir)
    audio_dir_path.mkdir(parents=True, exist_ok=True)

    cuts = load_manifest(input_manifest)
    worker = partial(_downmix_cut, audio_dir=audio_dir_path, id_suffix=id_suffix)

    if num_jobs <= 1:
        new_cuts = [worker(c) for c in cuts]
    else:
        with ProcessPoolExecutor(max_workers=num_jobs) as ex:
            new_cuts = list(ex.map(worker, cuts))

    CutSet.from_cuts(new_cuts).to_file(output_manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sum (downmix) multi-channel cuts to a single channel and save the "
        "result as a single audio file per cut, so future loading does not require "
        "reading and mixing all channels every time."
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
        help="Directory where the downmixed mono audio files will be stored",
    )
    parser.add_argument(
        "--id_suffix",
        type=str,
        default="-downmix",
        help="Suffix appended to cut/recording/supervision IDs of downmixed cuts, to keep "
        "them distinct from the original multi-channel manifest",
    )
    parser.add_argument(
        "--num_jobs", type=int, default=1, help="Number of parallel workers"
    )

    args = parser.parse_args()

    main(args.input, args.output, args.audio_dir, args.id_suffix, args.num_jobs)

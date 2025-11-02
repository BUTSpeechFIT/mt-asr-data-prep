import argparse

from lhotse import load_manifest
from lhotse.cut import MixedCut


def main(input_manifest, output_manifest, orig_prefix, new_prefix):
    cset = load_manifest(input_manifest)
    for cut in cset:
        if isinstance(cut, MixedCut):
            for track in cut.tracks:
                for src in track.cut.recording.sources:
                    src.source = src.source.replace(orig_prefix, new_prefix)
        else:
            for src in cut.recording.sources:
                src.source = src.source.replace(orig_prefix, new_prefix)

    cset.to_file(output_manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest", type=str, required=True)
    parser.add_argument("--output_manifest", type=str, required=True)
    parser.add_argument("--orig_prefix", type=str, required=True)
    parser.add_argument("--new_prefix", type=str, required=True)

    args = parser.parse_args()

    main(args.input_manifest, args.output_manifest, args.orig_prefix, args.new_prefix)

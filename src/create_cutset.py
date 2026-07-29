import argparse

from lhotse import CutSet, fix_manifests, load_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_recset",
        type=str,
        required=True,
        help="Path to the recordings manifest",
    )
    parser.add_argument(
        "--input_supset",
        type=str,
        default=None,
        help="Path to the supervisions manifest. Omit for recording-only corpora "
        "(e.g. noise/augmentation data) that have no supervisions.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output manifest"
    )

    args = parser.parse_args()

    rc = load_manifest(args.input_recset)

    if args.input_supset is not None:
        ss = load_manifest(args.input_supset)
        rc, ss = fix_manifests(rc, ss)
        CutSet.from_manifests(recordings=rc, supervisions=ss).to_file(args.output)
    else:
        CutSet.from_manifests(recordings=rc).to_file(args.output)

import argparse

from lhotse import load_manifest


def main(input_manifest, output_manifest, max_len):
    cset = load_manifest(input_manifest)
    cset = cset.filter(lambda c: c.duration < max_len)
    cset.to_file(output_manifest)


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

    args = parser.parse_args()
    main(args.input, args.output, args.max_len)

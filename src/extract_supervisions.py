import argparse

import lhotse


def main(input_cutset_path: str, output_path: str):
    cutset = lhotse.load_manifest(input_cutset_path)
    supset = cutset.decompose()[1]
    supset.to_file(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()

    main(args.cutset_path, args.output_path)

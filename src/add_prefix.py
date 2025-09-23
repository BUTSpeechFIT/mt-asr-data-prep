import argparse

from lhotse import load_manifest


def main(input_manifest, output_manifest, prefix):
    cset = load_manifest(input_manifest)
    for r in cset:
        r.id = prefix + "_" + r.id
        r.recording.id = prefix + "_" + r.recording.id
        for supervision in r.supervisions:
            supervision.id = prefix + "_" + supervision.id
    cset.to_file(output_manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_manifest', type=str, required=True)
    parser.add_argument('--output_manifest', type=str, required=True)
    parser.add_argument('--prefix', type=str, required=True)

    args = parser.parse_args()

    main(args.input_manifest, args.output_manifest, args.prefix)

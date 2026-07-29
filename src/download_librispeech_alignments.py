import argparse

from lhotse.recipes.librispeech import download_librispeech


def main(target_dir: str, dataset_parts: str, alignments_url: str = None):
    kwargs = {}
    if alignments_url is not None:
        kwargs["alignments_url"] = alignments_url
    download_librispeech(target_dir, dataset_parts=dataset_parts, alignments=True, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download word-level alignments for LibriSpeech "
        "(https://github.com/CorentinJ/librispeech-alignments) and merge them into an "
        "existing 'lhotse download librispeech' directory. Skips the audio download if "
        "it was already completed. Requires the 'gdown' package."
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Directory previously passed to 'lhotse download librispeech' (contains LibriSpeech/)",
    )
    parser.add_argument(
        "--dataset_parts",
        type=str,
        default="librispeech",
        help="'librispeech' (full corpus), 'mini_librispeech', or a specific split name",
    )
    parser.add_argument(
        "--alignments_url",
        type=str,
        default=None,
        help="Override the Google Drive URL for the alignments zip (e.g. a personal "
        "mirror), in case the default CorentinJ/librispeech-alignments link is "
        "rate-limited by Google Drive.",
    )
    args = parser.parse_args()

    main(args.target_dir, args.dataset_parts, args.alignments_url)

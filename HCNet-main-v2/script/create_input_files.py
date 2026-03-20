#!/usr/bin/env python3
import argparse
import os
import sys


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, repo_root)

    from utils import create_input_files  # noqa: E402

    parser = argparse.ArgumentParser(description="Create UCM input files for HCNet.")
    parser.add_argument("--dataset", default="UCM", help="Dataset name.")
    parser.add_argument(
        "--karpathy_json_path",
        default=os.path.join(repo_root, "data", "UCM", "dataset.json"),
        help="Path to captions JSON.",
    )
    parser.add_argument(
        "--image_folder",
        default=os.path.join(repo_root, "data", "UCM", "images"),
        help="Path to image folder.",
    )
    parser.add_argument(
        "--captions_per_image",
        type=int,
        default=5,
        help="How many captions each image has.",
    )
    parser.add_argument(
        "--min_word_freq",
        type=int,
        default=4,
        help="Minimum word frequency.",
    )
    parser.add_argument(
        "--output_folder",
        default=os.path.join(repo_root, "data", "UCM"),
        help="Output folder for generated files.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=100,
        help="Maximum caption length.",
    )
    args = parser.parse_args()

    if not (os.path.exists(args.output_folder) and os.path.isdir(args.output_folder)):
        os.makedirs(args.output_folder)

    print("Dataset:", args.dataset)
    print("Captions JSON:", args.karpathy_json_path)
    print("Image folder:", args.image_folder)
    print("Output folder:", args.output_folder)

    create_input_files(
        dataset=args.dataset,
        karpathy_json_path=args.karpathy_json_path,
        image_folder=args.image_folder,
        captions_per_image=args.captions_per_image,
        min_word_freq=args.min_word_freq,
        output_folder=args.output_folder,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()

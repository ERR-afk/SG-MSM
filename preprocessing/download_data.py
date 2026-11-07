#!/usr/bin/env python3
import os
import argparse
import pandas as pd

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

from datasets import load_dataset


def process_split(split_name, domain, base_path, audio_dir):
    """
    Download audio files and create metadata CSV for a given split.
    """
    print(f"Processing {domain} - {split_name} split...")

    dataset = load_dataset(
        "DBD-research-group/BirdSet",
        domain,
        split=split_name,
        streaming=True,
        trust_remote_code=True,
    )

    metadata = []
    downloaded_files = set()

    for i, item in enumerate(dataset):
        if i == 0:
            print(f"\nAvailable fields for {split_name}:", item.keys())

        ogg_filename = os.path.basename(item["filepath"])
        ogg_path = os.path.join(audio_dir, ogg_filename)

        if ogg_filename not in downloaded_files:
            with open(ogg_path, "wb") as f:
                f.write(item["audio"]["bytes"])
            downloaded_files.add(ogg_filename)

        data_dict = {"filepath": ogg_filename}

        for key, value in item.items():
            if key != "audio":
                data_dict[key] = (
                    str(value) if isinstance(value, (list, dict)) else value
                )

        metadata.append(data_dict)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} files")

    csv_filename = os.path.join(base_path, f"{domain}_{split_name}_metadata.csv")
    pd.DataFrame(metadata).to_csv(csv_filename, index=False)
    print(f"Saved {len(metadata)} samples to {csv_filename}")
    print(f"Saved {len(downloaded_files)} unique audio files to {audio_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download BirdSet data for a specific domain"
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain to download (e.g., PER, SSW, UHH, NES, SNE, HSN)",
    )
    parser.add_argument(
        "--data_dir", type=str, default="../data", help="Base directory to save data"
    )
    args = parser.parse_args()

    base_path = os.path.join(args.data_dir, f"Birdset_{args.domain}")
    train_audio_path = os.path.join(base_path, "train_audio")
    test_audio_path = os.path.join(base_path, "test_audio")
    test_5s_audio_path = os.path.join(base_path, "test_5s_audio")

    for dir_path in [base_path, train_audio_path, test_audio_path, test_5s_audio_path]:
        os.makedirs(dir_path, exist_ok=True)

    print(f"\nStarting {args.domain} dataset processing...")

    process_split("train", args.domain, base_path, train_audio_path)
    process_split("test", args.domain, base_path, test_audio_path)
    process_split("test_5s", args.domain, base_path, test_5s_audio_path)

    print(f"\nAll {args.domain} splits processed successfully!")
    print(f"Data saved to: {base_path}")


if __name__ == "__main__":
    main()

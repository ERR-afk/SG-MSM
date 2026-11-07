#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")

TARGET_SR = 32000


def process_file(file_path, output_dir):
    try:
        file_name = Path(file_path).stem
        output_path = os.path.join(output_dir, f"{file_name}.npy")

        if os.path.exists(output_path):
            return True

        audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
        audio = audio.astype(np.float32)
        np.save(output_path, audio)
        return True
    except Exception as e:
        return f"Error processing {file_path}: {e}"


def convert_directory(input_dir, output_dir):
    print(f"Converting: {input_dir}")
    print(f"Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    search_patterns = ["*.wav", "*.ogg", "*.flac", "*.mp3"]
    audio_files = []
    for pattern in search_patterns:
        audio_files.extend(glob.glob(os.path.join(input_dir, pattern)))
        audio_files.extend(
            glob.glob(os.path.join(input_dir, "**", pattern), recursive=True)
        )

    audio_files = list(set(audio_files))

    if not audio_files:
        print("No audio files found in directory!")
        return

    print(f"Found {len(audio_files)} audio files. Starting multi-core conversion...")

    num_cores = os.cpu_count() or 4
    success_count = 0
    error_messages = []

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(process_file, f, output_dir): f for f in audio_files}

        for future in tqdm(
            as_completed(futures),
            total=len(audio_files),
            desc="Converting",
            unit="file",
        ):
            result = future.result()
            if result is True:
                success_count += 1
            else:
                error_messages.append(result)

    print(f"Finished! Successfully converted: {success_count}/{len(audio_files)}")
    if error_messages:
        print(f"Encountered {len(error_messages)} errors. First 5:")
        for err in error_messages[:5]:
            print(f"   - {err}")


def main():
    parser = argparse.ArgumentParser(description="Convert BirdSet audio to .npy arrays")
    parser.add_argument(
        "--domain", type=str, required=True, help="Domain to process (e.g., PER, SSW)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="../data", help="Base directory of data"
    )
    args = parser.parse_args()

    base_path = os.path.join(args.data_dir, f"Birdset_{args.domain}")

    train_input = os.path.join(base_path, "train_audio")
    test_5s_input = os.path.join(base_path, "test_5s_audio")

    tmp_path = os.path.join(base_path, "tmp")
    train_output = os.path.join(tmp_path, "train_audio_npy")
    test_5s_output = os.path.join(tmp_path, "test_5s_audio_npy")

    convert_directory(train_input, train_output)
    convert_directory(test_5s_input, test_5s_output)


if __name__ == "__main__":
    main()

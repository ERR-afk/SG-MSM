#!/usr/bin/env python3
import os
import argparse
import ast
import random
import warnings

import pandas as pd
import numpy as np
import librosa
import noisereduce as nr
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")


def calculate_snr_with_noisereduce(audio, sr):
    """Calculate SNR using noisereduce with stationary=True"""
    denoised_audio = nr.reduce_noise(y=audio, sr=sr, stationary=True)
    noise = audio - denoised_audio
    signal_power = np.mean(np.square(denoised_audio))
    noise_power = np.mean(np.square(noise))
    if noise_power == 0:
        return 100.0
    return float(10 * np.log10(signal_power / noise_power))


def spectral_entropy(y, sr, n_fft=2048, normalize=True):
    """Calculate spectral entropy"""
    S = np.abs(librosa.stft(y, n_fft=n_fft))
    power_spec = np.mean(S**2, axis=1)
    power_spec_norm = power_spec / np.sum(power_spec)
    epsilon = 1e-10
    entropy = -np.sum(power_spec_norm * np.log2(power_spec_norm + epsilon))
    if normalize:
        max_entropy = np.log2(n_fft / 2 + 1)
        entropy = entropy / max_entropy
    return entropy


def parse_multilabel(multilabel_str):
    """Parse ebird_code_multilabel safely"""
    if pd.isna(multilabel_str):
        return []
    try:
        return ast.literal_eval(multilabel_str)
    except:
        return []


def extract_minute_segment(filepath):
    """Extract minute segment from filepath"""
    parts = filepath.replace(".ogg", "").split("_")
    if len(parts) >= 5:
        start_time = int(parts[-2])
        minute_segment = start_time // 60
        return minute_segment
    return None


def find_purest_noise_per_minute(recording_name, test_5s_df, audio_base_path):
    """Find lowest SNR chunks per minute segment in the 5s chunks"""
    try:
        recording_chunks = test_5s_df[
            test_5s_df["filepath"].str.startswith(recording_name)
        ]
        if len(recording_chunks) == 0:
            return []

        all_chunks = recording_chunks.copy()
        all_chunks["minute_segment"] = all_chunks["filepath"].apply(
            extract_minute_segment
        )
        all_chunks = all_chunks[all_chunks["minute_segment"].notna()]

        minute_results = []
        for minute_segment, minute_chunks in all_chunks.groupby("minute_segment"):
            chunk_metrics = []
            for _, row in minute_chunks.iterrows():
                audio_path = os.path.join(audio_base_path, row["filepath"])
                if not os.path.exists(audio_path):
                    continue
                try:
                    y, sr = librosa.load(audio_path, sr=32000)
                    snr = calculate_snr_with_noisereduce(y, sr)
                    entropy = spectral_entropy(y, sr)
                    chunk_metrics.append(
                        {
                            "filepath": row["filepath"],
                            "snr": snr,
                            "entropy": entropy,
                            "minute_segment": int(minute_segment),
                        }
                    )
                except:
                    continue

            if len(chunk_metrics) > 0:
                lowest_snr_chunk = min(chunk_metrics, key=lambda x: x["snr"])
                npy_filename = lowest_snr_chunk["filepath"].replace(".ogg", ".npy")
                minute_results.append(
                    {
                        "filename": npy_filename,
                        "SNR": lowest_snr_chunk["snr"],
                        "entropy": lowest_snr_chunk["entropy"],
                        "segment": lowest_snr_chunk["minute_segment"] + 1,
                        "recording_name": recording_name,
                    }
                )
        return minute_results
    except:
        return []


def sliding_window_snr_analysis(
    audio_data, target_segment, window_duration=5.0, overlap_duration=2.5, sr=32000
):
    try:
        start_sample = int((target_segment - 1) * 60 * sr)
        end_sample = int(target_segment * 60 * sr)
        if end_sample > len(audio_data):
            end_sample = len(audio_data)

        segment_audio = audio_data[start_sample:end_sample]
        window_samples = int(window_duration * sr)
        step_samples = int((window_duration - overlap_duration) * sr)

        num_windows = int((len(segment_audio) - window_samples) / step_samples) + 1

        snr_values = []
        for i in range(num_windows):
            start_window = i * step_samples
            end_window = start_window + window_samples
            if end_window > len(segment_audio):
                break
            snr = calculate_snr_with_noisereduce(
                segment_audio[start_window:end_window], sr
            )
            snr_values.append(snr)
        return snr_values
    except:
        return []


def partition_snr_categories(noise_snr, sliding_window_snrs, min_values=1):
    if not sliding_window_snrs:
        return {"raw_S1": None, "raw_S2": None, "raw_S3": None}

    s3_vals = [s for s in sliding_window_snrs if noise_snr <= s <= noise_snr + 5]
    s2_vals = [s for s in sliding_window_snrs if noise_snr + 5 < s <= noise_snr + 10]
    s1_vals = [s for s in sliding_window_snrs if s > noise_snr + 10]

    return {
        "raw_S1": np.mean(s1_vals) if len(s1_vals) >= min_values else None,
        "raw_S2": np.mean(s2_vals) if len(s2_vals) >= min_values else None,
        "raw_S3": np.mean(s3_vals) if len(s3_vals) >= min_values else None,
    }


def process_sliding_windows_for_recording(
    recording_name, profiles, continuous_audio_path
):
    try:
        audio_path = os.path.join(continuous_audio_path, f"{recording_name}.ogg")
        if not os.path.exists(audio_path):
            return profiles

        audio_data, sr = librosa.load(audio_path, sr=32000)

        for profile in profiles:
            snr = profile["SNR"]
            segment = profile["segment"]

            sliding_snrs = sliding_window_snr_analysis(audio_data, segment)
            categories = partition_snr_categories(snr, sliding_snrs)
            profile.update(categories)
        return profiles
    except:
        return profiles


def perform_clustering(df):
    """Apply KMeans (K=3) to map raw SNR values into final High, Medium, Low clusters."""
    all_raw_snrs = []
    for col in ["raw_S1", "raw_S2", "raw_S3"]:
        all_raw_snrs.extend(df[col].dropna().values.tolist())

    if len(all_raw_snrs) < 3:
        print("Warning: Not enough data points for KMeans clustering!")
        return df

    X = np.array(all_raw_snrs).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    centroids = kmeans.cluster_centers_.flatten()
    sorted_idx = np.argsort(centroids)[::-1]

    mapping = {
        sorted_idx[0]: "HIGH (S1)",
        sorted_idx[1]: "MEDIUM (S2)",
        sorted_idx[2]: "LOW (S3)",
    }

    df["S1"] = np.nan
    df["S2"] = np.nan
    df["S3"] = np.nan
    df["S4"] = np.nan

    for idx, row in df.iterrows():
        if row["entropy"] < 0.35:
            df.at[idx, "S4"] = row["SNR"] + random.uniform(0.1, 0.3)
        else:
            for col in ["raw_S1", "raw_S2", "raw_S3"]:
                if pd.notna(row[col]):
                    cluster_id = kmeans.predict([[row[col]]])[0]
                    category = mapping[cluster_id]
                    if category == "HIGH (S1)":
                        df.at[idx, "S1"] = row[col]
                    elif category == "MEDIUM (S2)":
                        df.at[idx, "S2"] = row[col]
                    elif category == "LOW (S3)":
                        df.at[idx, "S3"] = row[col]

    df = df.drop(columns=["raw_S1", "raw_S2", "raw_S3"])
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Systematic Noise Profile Mining for SG-Mix (All Phases)"
    )
    parser.add_argument(
        "--domain", type=str, required=True, help="Domain to process (e.g., PER, SSW)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="../data", help="Base directory of data"
    )
    args = parser.parse_args()

    domain = args.domain
    base_path = os.path.join(args.data_dir, f"Birdset_{domain}")
    input_csv = os.path.join(base_path, f"{domain}_test_5s_metadata.csv")
    audio_5s_path = os.path.join(base_path, "test_5s_audio")
    audio_continuous_path = os.path.join(base_path, "test_audio")
    output_csv = os.path.join(base_path, "sound_profile_npy.csv")

    print(f"Systematic Noise Profile Mining for {domain}")

    print("Loading metadata...")
    test_5s_df = pd.read_csv(input_csv)

    unique_recordings = set()
    for filepath in test_5s_df["filepath"]:
        if isinstance(filepath, str):
            parts = filepath.replace(".ogg", "").split("_")
            if len(parts) >= 5:
                recording_name = "_".join(parts[:-2])
                unique_recordings.add(recording_name)
    unique_recordings = sorted(list(unique_recordings))
    print(f"Found {len(unique_recordings)} unique 1-hour recordings")

    print("Starting Phase 1: Mining purest noise chunks (base SNR)...")
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(find_purest_noise_per_minute)(rec, test_5s_df, audio_5s_path)
        for rec in tqdm(unique_recordings, desc="Phase 1", ncols=80)
    )

    all_profiles = []
    for r in results:
        all_profiles.extend(r)

    print("Starting Phase 2: Sliding window analysis on continuous audio...")
    profiles_by_recording = {}
    for p in all_profiles:
        profiles_by_recording.setdefault(p["recording_name"], []).append(p)

    phase2_results = Parallel(n_jobs=-1, verbose=0)(
        delayed(process_sliding_windows_for_recording)(
            rec, profiles_by_recording[rec], audio_continuous_path
        )
        for rec in tqdm(profiles_by_recording.keys(), desc="Phase 2", ncols=80)
    )

    final_profiles = []
    for r in phase2_results:
        final_profiles.extend(r)

    print("Starting Phase 3: KMeans clustering (K=3) and S4 generation...")
    df = pd.DataFrame(final_profiles)

    df = perform_clustering(df)

    cols = ["filename", "SNR", "entropy", "S1", "S2", "S3", "S4"]
    df = df[cols]

    df.to_csv(output_csv, index=False)

    print("Isolating pure noise profiles into dedicated directory...")
    import shutil

    domain_noise_dir = os.path.join(base_path, "tmp", "domain_noise_npy")
    test_5s_npy_dir = os.path.join(base_path, "tmp", "test_5s_audio_npy")
    os.makedirs(domain_noise_dir, exist_ok=True)

    copied_count = 0
    for filename in df["filename"]:
        src = os.path.join(test_5s_npy_dir, filename)
        dst = os.path.join(domain_noise_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied_count += 1

    print(f"Noise Profile Mining Complete for {domain}")
    print(f"Output saved to: {output_csv}")
    print(f"Copied {copied_count} noise chunks to: {domain_noise_dir}")
    print(f"Total valid profiles: {len(df)}")


if __name__ == "__main__":
    main()

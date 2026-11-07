"""
Evaluate trained PyTorch CNN models on validation set.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from glob import glob

warnings.filterwarnings("ignore")
try:
    from training_utils import LitCls, audio_to_processed_mel
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from training_utils import LitCls, audio_to_processed_mel
import timm


class SmartMultilabelT1Accuracy:
    """
    🎯 SMART T1-Accuracy: Automatically skips empty segments
    Logic: For each sample with positive labels, check if top-1 prediction matches ANY label.
    Empty segments (no positive labels) are EXCLUDED from computation.
    This matches AudioProtoPNet methodology and ensures fair evaluation.
    """

    def __init__(self):
        self.correct = 0
        self.total = 0
        self.skipped_empty = 0

    def update(self, preds, targets):
        """
        Args:
            preds: np.ndarray (B, num_classes) - predicted probabilities [0,1]
            targets: np.ndarray (B, num_classes) - binary targets {0,1}
        """
        batch_size = preds.shape[0]
        top1_indices = np.argmax(preds, axis=1)
        for i in range(batch_size):
            positive_labels = np.where(targets[i] == 1)[0]
            if len(positive_labels) == 0:
                self.skipped_empty += 1
                continue
            top1_class = top1_indices[i]
            if top1_class in positive_labels:
                self.correct += 1
            self.total += 1

    def compute(self):
        """Return T1-Accuracy (excluding empty segments)"""
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def reset(self):
        """Reset counters"""
        self.correct = 0
        self.total = 0
        self.skipped_empty = 0


class MetricAccumulator:
    """Accumulates predictions and labels for batch-wise scikit-learn metric computation"""

    def __init__(self):
        self.all_preds = []
        self.all_labels = []

    def update(self, preds, labels):
        """Accumulate batch predictions and labels"""
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def compute_cmap(self):
        """Compute CMAP (Average Precision) using scikit-learn"""
        if len(self.all_preds) == 0:
            return 0.0
        all_preds = np.vstack(self.all_preds)
        all_labels = np.vstack(self.all_labels)
        return average_precision_score(all_labels, all_preds, average="macro")

    def compute_auroc(self):
        """Compute AUROC using scikit-learn"""
        if len(self.all_preds) == 0:
            return 0.0
        all_preds = np.vstack(self.all_preds)
        all_labels = np.vstack(self.all_labels)
        return roc_auc_score(all_labels, all_preds, average="macro")

    def reset(self):
        """Reset accumulated data"""
        self.all_preds = []
        self.all_labels = []


def load_validation_data(validation_csv, audio_dir, species_mapping_csv):
    """
    Load and preprocess validation data
    """
    print("Loading validation data...")
    print(f"Loading species mapping from: {species_mapping_csv}")
    species_df = pd.read_csv(species_mapping_csv)
    species_df = species_df.sort_values("species_id")
    species_list = species_df["ebird_code"].tolist()
    num_classes = len(species_list)
    dataset_name = Path(species_mapping_csv).stem.split("_")[0]
    print(f"Loaded {num_classes} {dataset_name} species")
    print(f"Loading validation CSV from: {validation_csv}")
    val_df = pd.read_csv(validation_csv)
    print(f"Loaded {len(val_df)} validation samples")
    label_counts = val_df[species_list].sum(axis=1)
    num_empty = (label_counts == 0).sum()
    print(
        f"Total samples: {len(val_df)}, Empty segments: {num_empty}, Non-empty: {len(val_df) - num_empty}"
    )
    print(f"Pre-computing spectrograms...")
    specs_list = []
    labels_list = []
    for idx in tqdm(range(len(val_df)), desc="Pre-computing spectrograms"):
        row = val_df.iloc[idx]
        filename = str(row["filename"]).strip()
        starttime = float(row["starttime"]) if not pd.isna(row["starttime"]) else 0.0
        audio_path = os.path.join(audio_dir, filename)
        if not os.path.exists(audio_path):
            audio_path = audio_path.replace(".npy", ".ogg")
            if not os.path.exists(audio_path):
                print(f"Error: Could not find {filename}")
                continue
            import librosa

            full_audio, _ = librosa.load(audio_path, sr=32000)
            full_audio = torch.from_numpy(full_audio).float()
        else:
            try:
                full_audio = torch.from_numpy(np.load(audio_path)).float()
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        start_sample = int(starttime * 32000)
        end_sample = start_sample + (5 * 32000)
        if start_sample < full_audio.shape[0]:
            if end_sample <= full_audio.shape[0]:
                audio_segment = full_audio[start_sample:end_sample]
            else:
                audio_segment = full_audio[start_sample:]
        else:
            print(f"Start sample beyond file length for {filename}")
            continue
        target_length = 5 * 32000
        if len(audio_segment) < target_length:
            if len(audio_segment) == 0:
                continue
            repeat_count = int(np.ceil(target_length / len(audio_segment)))
            audio_segment = audio_segment.repeat(repeat_count)[:target_length]
        elif len(audio_segment) > target_length:
            audio_segment = audio_segment[:target_length]
        spec = audio_to_processed_mel(audio_segment, sr=32000)
        specs_list.append(spec.cpu().numpy())
        labels = row[species_list].astype(np.float32).values
        labels_list.append(labels)
    specs = np.stack(specs_list)
    labels = np.stack(labels_list)
    print(
        f"Pre-computation complete. Specs: {specs.shape}, Labels: {labels.shape}, Memory: {specs.nbytes / 1e9:.2f} GB"
    )
    return specs, labels, dataset_name, num_classes, species_list


def load_model_from_checkpoint(
    checkpoint_path, num_classes, model_name="efficientnet_b3", device="cuda:0"
):
    """Load trained PyTorch model from checkpoint"""
    print(f"Loading model from checkpoint...")
    base_model = timm.create_model(
        model_name, pretrained=False, num_classes=num_classes, in_chans=3
    )
    lit_cls = LitCls(
        base_model,
        learning_rate=9e-4,
        cutmix_p=0.0,
        augment_config=None,
        cutmix_type="cyclic",
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    lit_cls.load_state_dict(checkpoint["state_dict"])
    lit_cls.eval()
    lit_cls.to(device)
    print(f"Model loaded successfully")
    return lit_cls


def evaluate_checkpoint(model, specs, labels, batch_size=128, device="cuda:0"):
    """Evaluate a single checkpoint on pre-computed spectrograms"""
    metric_accumulator = MetricAccumulator()
    t1_acc = SmartMultilabelT1Accuracy()
    num_samples = len(specs)
    num_batches = (num_samples + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Inference", leave=False):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_specs = torch.from_numpy(specs[start_idx:end_idx]).float().to(device)
            batch_labels = labels[start_idx:end_idx]
            logits = model.model(batch_specs)
            probs = torch.sigmoid(logits).cpu().numpy()
            metric_accumulator.update(probs, batch_labels)
            t1_acc.update(probs, batch_labels)
    return {
        "cmap": metric_accumulator.compute_cmap(),
        "auroc": metric_accumulator.compute_auroc(),
        "t1_accuracy": t1_acc.compute(),
        "num_samples": num_samples,
        "num_empty_skipped": t1_acc.skipped_empty,
    }


def evaluate_model(
    checkpoint_path=None,
    checkpoint_dir=None,
    validation_csv="",
    audio_dir="",
    species_mapping_csv="",
    model_name="efficientnet_b3",
    batch_size=128,
    gpu_id=0,
):
    """Main evaluation function"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch CNN Model Evaluation")
    specs, labels, dataset_name, num_classes, species_list = load_validation_data(
        validation_csv=validation_csv,
        audio_dir=audio_dir,
        species_mapping_csv=species_mapping_csv,
    )
    if checkpoint_path is not None:
        checkpoint_files = [checkpoint_path]
    elif checkpoint_dir is not None:
        checkpoint_files = sorted(glob(os.path.join(checkpoint_dir, "*.ckpt")))
    else:
        raise ValueError("Must provide either checkpoint_path or checkpoint_dir!")
    all_results = []
    for ckpt_path in checkpoint_files:
        ckpt_name = Path(ckpt_path).name
        print(f"\nEvaluating: {ckpt_name}")
        model = load_model_from_checkpoint(
            checkpoint_path=ckpt_path,
            num_classes=num_classes,
            model_name=model_name,
            device=device,
        )
        results = evaluate_checkpoint(model, specs, labels, batch_size, device)
        results["checkpoint"] = ckpt_name
        results["checkpoint_path"] = ckpt_path
        all_results.append(results)
        print(
            f"CMAP: {results['cmap']:.6f} | AUROC: {results['auroc']:.6f} | T1-Acc: {results['t1_accuracy']:.6f}"
        )
        del model
        torch.cuda.empty_cache()
    print("\nEvaluation Complete - Summary")
    if len(all_results) > 1:
        df_results = pd.DataFrame(all_results)
        best_cmap_idx = df_results["cmap"].idxmax()
        best_auroc_idx = df_results["auroc"].idxmax()
        best_t1_idx = df_results["t1_accuracy"].idxmax()
        print(f"\nBest Checkpoints:")
        print(
            f"Best CMAP: {df_results.loc[best_cmap_idx, 'checkpoint']} ({df_results.loc[best_cmap_idx, 'cmap']:.6f})"
        )
        print(
            f"Best AUROC: {df_results.loc[best_auroc_idx, 'checkpoint']} ({df_results.loc[best_auroc_idx, 'auroc']:.6f})"
        )
        output_csv = os.path.join(
            checkpoint_dir if checkpoint_dir else os.path.dirname(checkpoint_path),
            f"evaluation_results_{dataset_name}.csv",
        )
        df_results.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PyTorch models")
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint", type=str, help="Path to single checkpoint file"
    )
    checkpoint_group.add_argument(
        "--checkpoint_dir", type=str, help="Path to directory with checkpoints"
    )
    parser.add_argument(
        "--validation_csv", type=str, required=True, help="Path to validation CSV file"
    )
    parser.add_argument(
        "--audio_dir", type=str, required=True, help="Path to audio directory"
    )
    parser.add_argument(
        "--species_mapping", type=str, required=True, help="Path to species mapping CSV"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="efficientnet_b3",
        help="Model architecture name (default: efficientnet_b3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for inference (default: 128)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (default: 0)")
    args = parser.parse_args()
    evaluate_model(
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        validation_csv=args.validation_csv,
        audio_dir=args.audio_dir,
        species_mapping_csv=args.species_mapping,
        model_name=args.model_name,
        batch_size=args.batch_size,
        gpu_id=args.gpu,
    )

import numpy as np
import pandas as pd
import librosa
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
import random
import os

import noisereduce as nr

import torch
import torchaudio
import torchaudio.transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection, MeanMetric, Accuracy, F1Score, AveragePrecision
from torchmetrics.classification import MultilabelAUROC
from pytorch_lightning import LightningModule, LightningDataModule
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm
from torch.multiprocessing import Manager

class MultilabelT1Accuracy:
    """
    T1 Accuracy for multilabel classification - fair comparison with AudioProtoPNet
    
    Logic: For each sample, check if the top-1 predicted class matches ANY positive label
    This is exactly how AudioProtoPNet calculates T1 accuracy for multilabel tasks
    """
    def __init__(self):
        self.correct = 0
        self.total = 0
    
    def update(self, preds, targets):
        """
        Args:
            preds: torch.Tensor (B, num_classes) - predicted probabilities [0,1]
            targets: torch.Tensor (B, num_classes) - binary targets {0,1}
        """
        batch_size = preds.shape[0]
        
        top1_indices = torch.argmax(preds, dim=1)
        
        for i in range(batch_size):
            top1_class = top1_indices[i].item()
            positive_labels = targets[i].nonzero(as_tuple=True)[0]
            
            if top1_class in positive_labels:
                self.correct += 1
            self.total += 1
    
    def compute(self):
        """Return current T1 accuracy"""
        if self.total == 0:
            return 0.0
        return self.correct / self.total
    
    def reset(self):
        """Reset counters"""
        self.correct = 0
        self.total = 0

_GLOBAL_COLOR_MAP = None

def _get_shared_color_map(n_bins=129):
    """Get or create the single shared color map for all feature engineering"""
    global _GLOBAL_COLOR_MAP
    if _GLOBAL_COLOR_MAP is None:
        cmap = torch.zeros((n_bins, 3), dtype=torch.float32)
        n_sections = n_bins // 3
        for i in range(n_sections):
            cmap[i, 0] = 1 - i/n_sections
            cmap[i, 1] = i/n_sections
            idx = n_sections + i
            cmap[idx, 1] = 1 - i/n_sections
            cmap[idx, 2] = i/n_sections
            idx = 2 * n_sections + i
            cmap[idx, 2] = 1 - i/n_sections
            cmap[idx, 0] = i/n_sections
        _GLOBAL_COLOR_MAP = cmap.unsqueeze(1)
    return _GLOBAL_COLOR_MAP

def audio_to_processed_mel(audio, sr=32000):
    """Single shared feature engineering function for both training and validation"""
    if not isinstance(audio, torch.Tensor):
        audio = torch.from_numpy(audio).float()
    
    device = audio.device
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=129,
        f_min=100,
        f_max=15000,
        power=2.0
    ).to(device)
    
    spec = mel_transform(audio)
    
    spec_min = torch.min(spec)
    spec_max = torch.max(spec)
    spec_normalized = (spec - spec_min) / (spec_max - spec_min + 1e-8)
    
    epsilon = 1e-6
    spec_log = torch.log(torch.clamp_min(spec_normalized, epsilon))
    
    spec_log_min = torch.min(spec_log)
    spec_log_max = torch.max(spec_log)
    spec_log_normalized = (spec_log - spec_log_min) / (spec_log_max - spec_log_min + 1e-8)
    
    color_map = _get_shared_color_map().to(spec_log_normalized.device)
    x_rgb = spec_log_normalized.unsqueeze(-1) * color_map
    x_rgb = torch.flip(x_rgb, dims=[0])
    x_rgb = x_rgb.permute(2, 0, 1)
    return x_rgb

def calculate_snr_with_noisereduce(audio_tensor, sr, return_denoised=False):
    """
    Calculate signal-to-noise ratio using noisereduce with stationary=True for noise reduction.
    
    Args:
        audio_tensor (torch.Tensor): Audio signal as tensor
        sr (int): Sample rate
        return_denoised (bool, optional): If True, return (snr, denoised_audio, noise_audio)
        
    Returns:
        float: SNR value (if return_denoised=False)
        tuple: (snr, denoised_audio, noise_audio) (if return_denoised=True)
    """
    if isinstance(audio_tensor, torch.Tensor):
        audio_np = audio_tensor.detach().cpu().numpy()
    else:
        audio_np = audio_tensor
    
    denoised_audio = nr.reduce_noise(y=audio_np, sr=sr, stationary=True)
    noise = audio_np - denoised_audio
    signal_power = np.mean(np.square(denoised_audio))
    noise_power = np.mean(np.square(noise))
    
    if noise_power == 0:
        snr = 100.0
    else:
        snr = 10 * np.log10(signal_power / noise_power)
    
    snr = float(snr)
    
    if return_denoised:
        return snr, denoised_audio, noise
    else:
        return snr

def find_optimal_alpha_original(train_audio, noise_audio, target_snr, sr, 
                               acceptable_tolerance=0.003,
                               seed_left_snr=None, seed_right_snr=None):
    """Method 1: Accurate binary search - calls noisereduce for every alpha iteration"""
    if isinstance(train_audio, torch.Tensor):
        train_audio_np = train_audio.detach().cpu().numpy()
        noise_audio_np = noise_audio.detach().cpu().numpy()
    else:
        train_audio_np = train_audio
        noise_audio_np = noise_audio
    
    def calculate_mixed_snr(alpha):
        mixed_audio = (alpha * train_audio_np) + ((1-alpha) * noise_audio_np)
        
        max_val = np.max(np.abs(mixed_audio))
        if max_val > 0:
            mixed_audio = mixed_audio / max_val
        
        return calculate_snr_with_noisereduce(mixed_audio, sr=sr)
    
    left_alpha = 0.0
    right_alpha = 1.0
    
    best_alpha = left_alpha
    min_diff = float('inf')
    iterations = 0
    max_iterations = 30
    
    left_snr = seed_left_snr
    right_snr = seed_right_snr
    
    while iterations < max_iterations:
        iterations += 1
        middle_alpha = (left_alpha + right_alpha) / 2
        
        if left_snr is None:
            left_snr = calculate_mixed_snr(left_alpha)
            
        if right_snr is None:
            right_snr = calculate_mixed_snr(right_alpha)
            
        middle_snr = calculate_mixed_snr(middle_alpha)
        
        left_diff = abs(target_snr - left_snr)
        middle_diff = abs(target_snr - middle_snr)
        right_diff = abs(target_snr - right_snr)
        
        if left_diff < min_diff:
            min_diff = left_diff
            best_alpha = left_alpha
        if middle_diff < min_diff:
            min_diff = middle_diff
            best_alpha = middle_alpha
        if right_diff < min_diff:
            min_diff = right_diff
            best_alpha = right_alpha
            
        if min_diff <= acceptable_tolerance:
            break
            
        if middle_snr > target_snr:
            right_alpha = middle_alpha
            right_snr = middle_snr
        else:
            left_alpha = middle_alpha
            left_snr = middle_snr
    
    return best_alpha


def find_optimal_alpha_cached(train_audio, noise_audio, target_snr, sr, 
                             acceptable_tolerance=0.003,
                             train_denoised=None, train_noise=None,
                             noise_denoised=None, noise_noise=None,
                             seed_left_snr=None, seed_right_snr=None):
    """
    Method 2 (Cached): Pre-compute signal/noise components with optional optimizations
    
     Accept pre-computed denoised components to avoid redundant noisereduce calls!
     Accept seed SNR values to avoid redundant left/right calculations!
    BACKWARD COMPATIBLE: All parameters are optional - works with legacy calls!
    """
    if isinstance(train_audio, torch.Tensor):
        train_audio_np = train_audio.detach().cpu().numpy()
        noise_audio_np = noise_audio.detach().cpu().numpy()
    else:
        train_audio_np = train_audio
        noise_audio_np = noise_audio
    
    if train_denoised is not None and train_noise is not None:
        signal_train = train_denoised
        noise_train = train_noise
    else:
        signal_train = nr.reduce_noise(y=train_audio_np, sr=sr, stationary=True)
        noise_train = train_audio_np - signal_train
    
    if noise_denoised is not None and noise_noise is not None:
        signal_noise = noise_denoised
        noise_noise_comp = noise_noise
    else:
        signal_noise = nr.reduce_noise(y=noise_audio_np, sr=sr, stationary=True)
        noise_noise_comp = noise_audio_np - signal_noise
    
    def calculate_mixed_snr_cached(alpha):
        mixed_signal = alpha * signal_train + (1-alpha) * signal_noise
        mixed_noise = alpha * noise_train + (1-alpha) * noise_noise_comp
        
        total_mixed = mixed_signal + mixed_noise
        
        max_val = np.max(np.abs(total_mixed))
        if max_val > 0:
            norm_factor = 1.0 / max_val
            mixed_signal = mixed_signal * norm_factor
            mixed_noise = mixed_noise * norm_factor
        
        signal_power = np.mean(np.square(mixed_signal))
        noise_power = np.mean(np.square(mixed_noise))
        
        if noise_power == 0:
            snr = 100.0
        else:
            snr = 10 * np.log10(signal_power / noise_power)
        
        return float(snr)
    
    left_alpha = 0.0
    right_alpha = 1.0
    
    best_alpha = left_alpha
    min_diff = float('inf')
    iterations = 0
    max_iterations = 30
    
    left_snr = seed_left_snr
    right_snr = seed_right_snr
    
    while iterations < max_iterations:
        iterations += 1
        middle_alpha = (left_alpha + right_alpha) / 2
        
        if left_snr is None:
            left_snr = calculate_mixed_snr_cached(left_alpha)
            
        if right_snr is None:
            right_snr = calculate_mixed_snr_cached(right_alpha)
            
        middle_snr = calculate_mixed_snr_cached(middle_alpha)
        
        left_diff = abs(target_snr - left_snr)
        middle_diff = abs(target_snr - middle_snr)
        right_diff = abs(target_snr - right_snr)
        
        if left_diff < min_diff:
            min_diff = left_diff
            best_alpha = left_alpha
        if middle_diff < min_diff:
            min_diff = middle_diff
            best_alpha = middle_alpha
        if right_diff < min_diff:
            min_diff = right_diff
            best_alpha = right_alpha
            
        if min_diff <= acceptable_tolerance:
            break
            
        if middle_snr > target_snr:
            right_alpha = middle_alpha
            right_snr = middle_snr
        else:
            left_alpha = middle_alpha
            left_snr = middle_snr
    
    return best_alpha

def find_optimal_alpha_hybrid(train_audio, noise_audio, target_snr, sr, 
                             acceptable_tolerance=0.003,
                             train_denoised=None, train_noise=None,
                             noise_denoised=None, noise_noise=None,
                             seed_left_snr=None, seed_right_snr=None):
    """
    Method 3 (Hybrid): 
    
    1. Use cached method for fast rough estimate (with pre-computed denoised components!)
    2. Take ONE accurate sample point at ±0.025 offset
    3. Linear interpolation to find final alpha
    
     Pass pre-computed denoised components to avoid redundant noisereduce calls!
    """
    if isinstance(train_audio, torch.Tensor):
        train_audio_np = train_audio.detach().cpu().numpy()
        noise_audio_np = noise_audio.detach().cpu().numpy()
    else:
        train_audio_np = train_audio
        noise_audio_np = noise_audio
    
    alpha_cached = find_optimal_alpha_cached(
        train_audio_np, noise_audio_np, target_snr, sr, acceptable_tolerance,
        train_denoised=train_denoised, train_noise=train_noise,
        noise_denoised=noise_denoised, noise_noise=noise_noise,
        seed_left_snr=seed_left_snr, seed_right_snr=seed_right_snr
    )
    
    mixed_audio_cached = (alpha_cached * train_audio_np) + ((1-alpha_cached) * noise_audio_np)
    max_val = np.max(np.abs(mixed_audio_cached))
    if max_val > 0:
        mixed_audio_cached = mixed_audio_cached / max_val
    
    achieved_snr_cached = calculate_snr_with_noisereduce(mixed_audio_cached, sr)
    
    snr_error = abs(achieved_snr_cached - target_snr)
    if snr_error <= 0.05:
        return alpha_cached
    
    sample_offset = 0.025
    if achieved_snr_cached < target_snr:
        alpha_sample = alpha_cached + sample_offset
    else:
        alpha_sample = alpha_cached - sample_offset
    
    alpha_sample = max(0.0, min(1.0, alpha_sample))
    
    mixed_audio_sample = (alpha_sample * train_audio_np) + ((1-alpha_sample) * noise_audio_np)
    max_val = np.max(np.abs(mixed_audio_sample))
    if max_val > 0:
        mixed_audio_sample = mixed_audio_sample / max_val
        
    achieved_snr_sample = calculate_snr_with_noisereduce(mixed_audio_sample, sr)
    
    snr_slope = achieved_snr_sample - achieved_snr_cached
    
    if abs(snr_slope) < 1e-6:
        alpha_final = alpha_cached
    else:
        alpha_final = alpha_cached + (alpha_sample - alpha_cached) * (target_snr - achieved_snr_cached) / (achieved_snr_sample - achieved_snr_cached)
        
        alpha_final = max(0.0, min(1.0, alpha_final))
    
    return alpha_final

def mix_audio_with_noise_tensor(train_audio, noise_audio, target_snr, sr=32000, 
                               mix_type='default', mixing_params=None):
    """
    SG-Mix algorithm to achieve target_snr
    """
    train_snr, train_denoised, train_noise = calculate_snr_with_noisereduce(train_audio, sr, return_denoised=True)
    noise_snr, noise_denoised, noise_noise = calculate_snr_with_noisereduce(noise_audio, sr, return_denoised=True)
    
    params = mixing_params[mix_type]
    acceptable_tolerance = params['acceptable_tolerance']
    
    boundary_db = 0.5
    near_noise_boundary = abs(target_snr - noise_snr) <= boundary_db
    near_train_boundary = abs(target_snr - train_snr) <= boundary_db
    
    if near_noise_boundary or near_train_boundary:
        alpha = find_optimal_alpha_original(
            train_audio, noise_audio, target_snr, sr,
            acceptable_tolerance=acceptable_tolerance,
            seed_left_snr=noise_snr,
            seed_right_snr=train_snr
        )
    else:
        alpha = find_optimal_alpha_hybrid(
            train_audio, noise_audio, target_snr, sr,
            acceptable_tolerance=acceptable_tolerance,
            train_denoised=train_denoised, train_noise=train_noise,
            noise_denoised=noise_denoised, noise_noise=noise_noise,
            seed_left_snr=noise_snr, seed_right_snr=train_snr
        )
    
    mixed_audio = (alpha * train_audio) + ((1-alpha) * noise_audio)
    
    mixed_audio = mixed_audio / torch.max(torch.abs(mixed_audio))
    
    return mixed_audio

class CyclicCrossSNRCutMix:
    """
    Implements cyclic cross-SNR CutMix: S1→S2→S3→S4→S1
    Maintains data count conservation and applies category-specific brightness augmentation
    
    - Direct tensor indexing (no nested dictionaries)
    - Category tracking via simple list
    - Vectorized mask operations
    - Preserves ALL CutMix principles from 1.py
    """
    def __init__(self, cutmix_probability=0.5, cutmix_type='cyclic'):
        self.cutmix_probability = cutmix_probability
        self.cutmix_type = cutmix_type
        
        valid_types = ['cyclic', 'random', 'self']
        if self.cutmix_type not in valid_types:
            raise ValueError(f"cutmix_type must be one of {valid_types}, got '{self.cutmix_type}'")
        
        self.snr_brightness_params = {
            'S1': {'min': 0.70, 'max': 0.80},
            'S2': {'min': 0.70, 'max': 0.80},
            'S3': {'min': 0.70, 'max': 0.80},
            'S4': {'min': 0.80, 'max': 0.90},
            'S5': {'min': 0.85, 'max': 0.95},
            'default': {'min': 0.75, 'max': 0.85}
        }
        
        print(f"CutMix initialized: type={self.cutmix_type}, probability={cutmix_probability}")
    
    def _apply_self_cutmix(self, specs, labels):
        """Self-CutMix - """
        eligible_indices = list(range(len(labels)))
        if len(eligible_indices) < 2:
            return specs, labels
        
        shuffled_eligible_indices = self._get_derangement(eligible_indices)
        w = specs.shape[-1]
        start_pos = w // 4
        section_width = w // 2
        brightness_range = self.snr_brightness_params.get('default', {'min': 0.75, 'max': 0.85})
        mixed_output = specs.clone()
        
        for orig_idx, shuf_idx in zip(eligible_indices, shuffled_eligible_indices):
            brightness_factor = random.uniform(brightness_range['min'], brightness_range['max'])
            orig_middle = specs[orig_idx, :, :, start_pos:start_pos+section_width]
            shuf_middle = specs[shuf_idx, :, :, start_pos:start_pos+(w-section_width)]
            shuf_middle = shuf_middle * brightness_factor
            mixed_output[orig_idx, :, :, :section_width] = orig_middle
            mixed_output[orig_idx, :, :, section_width:] = shuf_middle
            labels[orig_idx] = torch.maximum(labels[orig_idx], labels[shuf_idx])
        
        return mixed_output, labels
    
    def _get_derangement(self, indices):
        """Sattolo's algorithm - """
        result = indices.copy()
        n = len(result)
        for i in range(n-1, 0, -1):
            j = random.randrange(0, i)
            result[i], result[j] = result[j], result[i]
        del indices
        return result
    
    def _apply_cyclic_cross_snr_cutmix_from_categories(self, batch_specs, batch_labels, batch_categories):
        """
        Routes to appropriate CutMix strategy based on cutmix_type:
        - Single category: Self-CutMix
        - Multi-category + 'self': Per-category self-cutmix
        - Multi-category + 'cyclic': Fixed cyclic pairing (S1->S2->S3->S4->S1)
        - Multi-category + 'random': Random cyclic pairing
        """
        if torch.rand(1).item() > self.cutmix_probability:
            return batch_specs, batch_labels
        
        unique_cats = sorted(set(batch_categories))
        
        if len(unique_cats) == 1:
            return self._apply_self_cutmix(batch_specs, batch_labels)
        
        if self.cutmix_type == 'self':
            return self._apply_per_category_self_cutmix(batch_specs, batch_labels, batch_categories, unique_cats)
        
        elif self.cutmix_type == 'cyclic':
            category_order = unique_cats
            return self._apply_cyclic_pairing(batch_specs, batch_labels, batch_categories, category_order)
        
        elif self.cutmix_type == 'random':
            category_order = unique_cats.copy()
            random.shuffle(category_order)
            return self._apply_cyclic_pairing(batch_specs, batch_labels, batch_categories, category_order)
    
    def _apply_cyclic_pairing(self, batch_specs, batch_labels, batch_categories, category_order):
        """
        CORE CYCLIC LOGIC: Apply cyclic pairing based on given category order
        
        Args:
            category_order: List of categories in desired cyclic order
                           (sorted for 'cyclic', shuffled for 'random')
        
        This is the EXACT SAME logic as before, just extracted for reuse!
        """
        category_masks = {
            cat: torch.tensor([i for i, c in enumerate(batch_categories) if c == cat],
                            device=batch_specs.device)
            for cat in category_order
        }
        
        batch_size_per_cat = len(category_masks[category_order[0]])
        
        if batch_size_per_cat < 2:
            return batch_specs, batch_labels
        
        eligible_indices = list(range(batch_size_per_cat))
        
        mixed_specs = batch_specs.clone()
        mixed_labels = batch_labels.clone()
        
        w = batch_specs.shape[-1]
        start_pos = w // 4
        section_width = w // 2
        
        for i, current_cat in enumerate(category_order):
            next_cat = category_order[(i + 1) % len(category_order)]
            
            current_mask = category_masks[current_cat]
            next_mask = category_masks[next_cat]
            
            shuffled_indices = self._get_derangement(eligible_indices.copy())
            
            brightness_range = self.snr_brightness_params.get(next_cat, self.snr_brightness_params['default'])
            
            for local_orig_idx, local_shuf_idx in zip(eligible_indices, shuffled_indices):
                global_orig_idx = current_mask[local_orig_idx].item()
                global_shuf_idx = next_mask[local_shuf_idx].item()
                
                brightness_factor = random.uniform(brightness_range['min'], brightness_range['max'])
                
                orig_middle = batch_specs[global_orig_idx, :, :, start_pos:start_pos+section_width]
                shuf_middle = batch_specs[global_shuf_idx, :, :, start_pos:start_pos+(w-section_width)]
                shuf_middle = shuf_middle * brightness_factor
                
                mixed_specs[global_orig_idx, :, :, :section_width] = orig_middle
                mixed_specs[global_orig_idx, :, :, section_width:] = shuf_middle
                
                mixed_labels[global_orig_idx] = torch.maximum(
                    batch_labels[global_orig_idx],
                    batch_labels[global_shuf_idx]
                )
        
        return mixed_specs, mixed_labels
    
    def _apply_per_category_self_cutmix(self, batch_specs, batch_labels, batch_categories, unique_cats):
        """
        Apply self-cutmix independently within each SNR category.
        Samples only mix with other samples from their exact same category.
        """
        category_masks = {
            cat: torch.tensor([i for i, c in enumerate(batch_categories) if c == cat],
                            device=batch_specs.device)
            for cat in unique_cats
        }
        
        mixed_specs = batch_specs.clone()
        mixed_labels = batch_labels.clone()
        
        w = batch_specs.shape[-1]
        start_pos = w // 4
        section_width = w // 2
        
        for cat in unique_cats:
            cat_indices = category_masks[cat].tolist()
            
            if len(cat_indices) < 2:
                continue
            
            shuffled_cat_indices = self._get_derangement(cat_indices)
            
            brightness_range = self.snr_brightness_params.get(cat, self.snr_brightness_params['default'])
            
            for orig_local_idx, shuf_local_idx in zip(cat_indices, shuffled_cat_indices):
                brightness_factor = random.uniform(brightness_range['min'], brightness_range['max'])
                
                orig_middle = batch_specs[orig_local_idx, :, :, start_pos:start_pos+section_width]
                shuf_middle = batch_specs[shuf_local_idx, :, :, start_pos:start_pos+(w-section_width)]
                shuf_middle = shuf_middle * brightness_factor
                
                mixed_specs[orig_local_idx, :, :, :section_width] = orig_middle
                mixed_specs[orig_local_idx, :, :, section_width:] = shuf_middle
                
                mixed_labels[orig_local_idx] = torch.maximum(
                    batch_labels[orig_local_idx],
                    batch_labels[shuf_local_idx]
                )
        
        return mixed_specs, mixed_labels

class BirdDataset(Dataset):
    def __init__(self, df, **kwargs):
        df = df.reset_index(drop=True)

        self.LABELS = kwargs.get('labels', sorted(df['primary_label'].unique()))
        
        if 'train_audio_path' not in kwargs:
            raise ValueError("train_audio_path must be provided in kwargs")
        self.train_audio_path = kwargs['train_audio_path']
        self.sound_profiles_path = kwargs.get('sound_profiles_path', None)
        self.sound_profiles_data = kwargs.get('sound_profiles_data', None)
        
        self.sg_mix_mode = kwargs.get('sg_mix_mode', True)
        self.snr_categories = kwargs.get('snr_categories', ["S1", "S2", "S3", "S4"])
        self.focal_mixing_probability = kwargs.get('focal_mixing_probability', 0.5)
        
        self.overlap_max = kwargs.get('overlap_max', 4)
        
        self.no_sg_msm_duplicate = kwargs.get('no_sg_msm_duplicate', 1)
        
        self.mixing_params = {
            'S1': {'acceptable_tolerance': 0.05},
            'S2': {'acceptable_tolerance': 0.05},
            'S3': {'acceptable_tolerance': 0.05},
            'S4': {'acceptable_tolerance': 0.025},
            'S5': {'acceptable_tolerance': 0.05},
            'focal': {'acceptable_tolerance': 0.05}
        }
        
        self.file_info = {}
        
        if 'instance_start_array' in df.columns:
            print("Using NEW consolidated CSV format with instance_start_array")
            
            for _, row in df.iterrows():
                filename = row['filename']
                
                import ast
                instance_times = ast.literal_eval(row['instance_start_array'])
                
                if self.LABELS == ["No_Bird", "Bird"]:
                    labels = np.array([0.0, 1.0], dtype=np.float32)
                elif self.LABELS == ["Bird"]:
                    labels = np.array([1.0], dtype=np.float32)
                else:
                    labels = row[self.LABELS].values.astype(np.float32)
                
                
                id_snr = row['id_snr'] if 'id_snr' in row else 0.0
                id_entropy = row['id_entropy'] if 'id_entropy' in row else 0.0
                
                self.file_info[filename] = {
                    'instance_times': instance_times,
                    'labels': labels,
                    'id_snr': id_snr,
                    'id_entropy': id_entropy
                }
        else:
            print("Using LEGACY segmented CSV format - consider using consolidated format")
            
            instance_cols = [f'instance{i+1}_start' for i in range(5)]
            
            for (filename, offset_segm), group in df.groupby(['filename', 'offset_segm']):
                instance_times = group[instance_cols].iloc[0].values
                
                if self.LABELS == ["No_Bird", "Bird"]:
                    labels = np.array([0.0, 1.0], dtype=np.float32)
                elif self.LABELS == ["Bird"]:
                    labels = np.array([1.0], dtype=np.float32)
                else:
                    labels = group[self.LABELS].iloc[0].values.astype(np.float32)
                
                id_snr = group['id_snr'].iloc[0] if 'id_snr' in group.columns else 0.0
                id_entropy = group['id_entropy'].iloc[0] if 'id_entropy' in group.columns else 0.0
                
                self.file_info[(filename, offset_segm)] = {
                    'instance_times': instance_times,
                    'labels': labels,
                    'mask': ~np.isnan(instance_times),
                    'id_snr': id_snr,
                    'id_entropy': id_entropy
                }
        
        self.all_file_keys = list(self.file_info.keys())
        
        self.snr_entropy_index = []
        for file_key, file_data in self.file_info.items():
            id_snr = file_data.get('id_snr', 0.0)
            id_entropy = file_data.get('id_entropy', 0.0)
            self.snr_entropy_index.append((file_key, id_snr, id_entropy))
                
        
        self.disable_caching = kwargs.get('disable_caching', False)
        
        if not self.disable_caching:
            self.manager = Manager()
            self.audio_cache = self.manager.dict()
            self.noise_cache = self.manager.dict()
        else:
            self.manager = None
            self.audio_cache = {}
            self.noise_cache = {}
        
        self.used_secondary_files = set() if self.sg_mix_mode is not None else None
        print(f"Secondary tracking enabled for SG_Mix={self.sg_mix_mode}: {self.used_secondary_files is not None}")

    def __len__(self):
        return len(self.file_info)

    def __getitem__(self, idx):
        """
        Returns processed audio tensors and labels.
        
        Process:
        1. Load single random instance audio
        2. Apply Stage 1: Focal Mixing
        3. Apply Stage 2: Multi-SNR Domain Mixing
        4. Return stacked tensors for GPU processing
        """
        file_keys = list(self.file_info.keys())
        file_key = file_keys[idx]
        
        if isinstance(file_key, tuple):
            filename, offset_segm = file_key
            audio = self._get_single_random_instance_audio_legacy(filename, offset_segm)
        else:
            filename = file_key
            audio = self._get_single_random_instance_audio_consolidated(filename)

        file_info = self.file_info[file_key]
        
        if self.LABELS == ["No_Bird", "Bird"]:
            labels = torch.tensor([0.0, 1.0], dtype=torch.float32)
        elif self.LABELS == ["Bird"]:
            labels = torch.tensor([1.0], dtype=torch.float32)
        else:
            labels = torch.from_numpy(file_info['labels']).float()
            
        id_entropy = file_info.get('id_entropy', 0.0)
        id_snr = file_info.get('id_snr', 0.0)

        mixed_audio, mixed_labels = self._apply_stage1_focal_mixing(
            audio, labels, id_entropy
        )

        final_audio_or_variants, final_labels = self._apply_stage2_multi_snr_mixing(
            mixed_audio, mixed_labels, id_entropy, id_snr
        )

        if self.sg_mix_mode == True:
            audio_variants = {}
            label_variants = {}
            
            for category in self.snr_categories:
                if category in final_audio_or_variants['snr_variants']:
                    audio_variants[category] = final_audio_or_variants['snr_variants'][category]['audio']
                    label_variants[category] = final_audio_or_variants['snr_variants'][category]['labels']

            return {
                'raw_audio': {'snr_variants': audio_variants},
                'labels': {'snr_variants': label_variants}
            }
        elif self.sg_mix_mode is None and getattr(self, 'no_sg_msm_duplicate', 1) > 1:
            audio_variants = {}
            label_variants = {}
            for i in range(self.no_sg_msm_duplicate):
                dup_key = f"dup_{i+1}"
                audio_variants[dup_key] = final_audio_or_variants
                label_variants[dup_key] = final_labels
                
            return {
                'raw_audio': {'snr_variants': audio_variants},
                'labels': {'snr_variants': label_variants}
            }
        else:
            return {
                'raw_audio': final_audio_or_variants,
                'labels': final_labels
            }

    def _make_exactly_5_seconds(self, audio, sample_rate=32000):
        """Ensure audio segment is exactly 5 seconds by padding or trimming"""
        target_length = 5 * sample_rate
        
        if len(audio) > target_length:
            return audio[:target_length]
        elif len(audio) < target_length:
            repeat_count = int(torch.ceil(torch.tensor(target_length / len(audio))).item())
            return audio.repeat(repeat_count)[:target_length]
        else:
            return audio

    def _get_single_random_instance_audio_consolidated(self, filename):
        """ NEW consolidated format - select random instance from ALL available instances"""
        file_info = self.file_info[filename]
        instance_times = file_info['instance_times']
        
        if len(instance_times) == 0:
            raise RuntimeError(f"No valid instance start times for {filename}")
        
        selected_time = random.choice(instance_times)
        
        return self._load_audio_segment(filename, selected_time)
    
    def _get_single_random_instance_audio_legacy(self, filename, offset_segm):
        """LEGACY SUPPORT: Select and process one random valid instance from specific segment"""
        file_info = self.file_info[(filename, offset_segm)]
        instance_times = file_info['instance_times']
        mask = file_info['mask']
        
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            raise RuntimeError(f"No valid instance start times for {filename} @ {offset_segm}")
        
        selected_idx = random.choice(valid_indices)
        selected_time = instance_times[selected_idx]
        
        return self._load_audio_segment(filename, selected_time)
    
    def _load_audio_segment(self, filename, selected_time):
        """SHARED: Load audio segment for given filename and start time"""
        disable_caching = hasattr(self, 'disable_caching') and self.disable_caching
        
        if not disable_caching and filename in self.audio_cache:
            full_audio = self.audio_cache[filename]
        else:
            audio_path = f"{self.train_audio_path}{filename}"
            try:
                full_audio = torch.from_numpy(np.load(audio_path)).float()
                if not disable_caching:
                    self.audio_cache[filename] = full_audio
            except Exception as e:
                raise RuntimeError(f"Error loading audio file {filename}: {e}")
        
        start_sample = int(selected_time * 32000)
        end_sample = start_sample + (5 * 32000)
        
        if start_sample < full_audio.shape[0]:
            if end_sample <= full_audio.shape[0]:
                audio_segment = full_audio[start_sample:end_sample]
            else:
                audio_segment = full_audio[start_sample:]
        else:
            raise RuntimeError(f"Start sample beyond file length for {filename}: {start_sample} >= {full_audio.shape[0]}")
        
        if start_sample == 0:
            audio_segment = audio_segment[:int(2.5 * 32000)]
        elif start_sample == int(0.5 * 32000):
            audio_segment = audio_segment[:int(2.5 * 32000)]
        
        audio_segment = self._make_exactly_5_seconds(audio_segment, 32000)
        
        return audio_segment

    def _pick_random_from_dataset(self, constraint_fn=None, prefer_unused_secondaries=False):
        """
        Selects a random secondary sample from the dataset.
        
        Implements an auto-clearing tracking system to maximize diversity across epochs.
        Applies constraints (e.g. SNR boundaries, entropy limits) if provided.
        """
        if constraint_fn is None:
            if prefer_unused_secondaries and self.used_secondary_files is not None:
                unused_entries = [
                    (file_key, id_snr, id_entropy) 
                    for file_key, id_snr, id_entropy in self.snr_entropy_index
                    if self._extract_filename_from_key(file_key) not in self.used_secondary_files
                ]
                
                if len(unused_entries) > 0:
                    file_key, _, _ = random.choice(unused_entries)
                    filename = self._extract_filename_from_key(file_key)
                    self.used_secondary_files.add(filename)
                else:
                    print(f"Auto-clearing used secondary files ({len(self.used_secondary_files)} → 0) - unused pool exhausted")
                    self.used_secondary_files.clear()
                    file_key, _, _ = random.choice(self.snr_entropy_index)
                    filename = self._extract_filename_from_key(file_key)
                    self.used_secondary_files.add(filename)
            else:
                file_key, _, _ = random.choice(self.snr_entropy_index)
        else:
            if prefer_unused_secondaries and self.used_secondary_files is not None:
                
                unused_with_constraints = [
                    (file_key, id_snr, id_entropy) 
                    for file_key, id_snr, id_entropy in self.snr_entropy_index
                    if (constraint_fn({'id_snr': id_snr, 'id_entropy': id_entropy}) and 
                        self._extract_filename_from_key(file_key) not in self.used_secondary_files)
                ]
                
                if len(unused_with_constraints) > 0:
                    file_key, _, _ = random.choice(unused_with_constraints)
                    filename = self._extract_filename_from_key(file_key)
                    self.used_secondary_files.add(filename)
                else:
                    base_constraint_fn = lambda seg: not ((seg['id_snr'] <= -15) and (seg['id_entropy'] <= 0.35))
                    unused_base_only = [
                        (file_key, id_snr, id_entropy) 
                        for file_key, id_snr, id_entropy in self.snr_entropy_index
                        if (base_constraint_fn({'id_snr': id_snr, 'id_entropy': id_entropy}) and 
                            self._extract_filename_from_key(file_key) not in self.used_secondary_files)
                    ]
                    
                    if len(unused_base_only) > 0:
                        has_snr_constraint = 'id_snr' in str(constraint_fn) and '>' in str(constraint_fn)
                        if has_snr_constraint:
                            file_key, _, _ = max(unused_base_only, key=lambda x: x[1])
                        else:
                            file_key, _, _ = random.choice(unused_base_only)
                        
                        filename = self._extract_filename_from_key(file_key)
                        self.used_secondary_files.add(filename)
                    else:
                        print(f"Auto-clearing used secondary files ({len(self.used_secondary_files)} → 0) - unused pool exhausted")
                        self.used_secondary_files.clear()
                        
                        fresh_unused_with_constraints = [
                            (file_key, id_snr, id_entropy) 
                            for file_key, id_snr, id_entropy in self.snr_entropy_index
                            if constraint_fn({'id_snr': id_snr, 'id_entropy': id_entropy})
                        ]
                        
                        if len(fresh_unused_with_constraints) > 0:
                            file_key, _, _ = random.choice(fresh_unused_with_constraints)
                            filename = self._extract_filename_from_key(file_key)
                            self.used_secondary_files.add(filename)
                        else:
                            file_key, _, _ = random.choice(self.snr_entropy_index)
            else:
                eligible_entries = [
                    (file_key, id_snr, id_entropy) 
                    for file_key, id_snr, id_entropy in self.snr_entropy_index
                    if constraint_fn({'id_snr': id_snr, 'id_entropy': id_entropy})
                ]
                
                if len(eligible_entries) == 0:
                    file_key, _, _ = random.choice(self.snr_entropy_index)
                else:
                    file_key, _, _ = random.choice(eligible_entries)
        
        full_file_info = self.file_info[file_key]
        
        if isinstance(file_key, tuple):
            filename, offset_segm = file_key
            audio = self._get_single_random_instance_audio_legacy(filename, offset_segm)
        else:
            filename = file_key
            audio = self._get_single_random_instance_audio_consolidated(filename)
        
        return {
            'audio': audio,
            'labels': torch.from_numpy(full_file_info['labels']).float(),
            'id_snr': full_file_info['id_snr'],
            'id_entropy': full_file_info['id_entropy']
        }
    
    def _extract_filename_from_key(self, file_key):
        """ Extract filename from file_key (supports both consolidated and legacy formats)"""
        if isinstance(file_key, tuple):
            return file_key[0]
        else:
            return file_key
    
    def reset_secondary_tracking(self):
        """ Reset secondary tracking at epoch start (called by trainer callbacks)"""
        if self.used_secondary_files is not None:
            old_count = len(self.used_secondary_files)
            self.used_secondary_files.clear()
            print(f"Reset secondary tracking: {old_count} → 0 files")

    def _apply_stage1_focal_mixing(self, audio, labels, id_entropy):
        """
        Stage 1: Focal Mixing
        
        - sg_mix_mode=True: Pass through to Stage 2
        - sg_mix_mode=False: Uniform random mixing
        - sg_mix_mode=None: SNR-controlled focal mixing
        """
        if self.sg_mix_mode is None:
            return self._apply_snr_controlled_focal_mixing_worker(audio, labels)
            
        elif self.sg_mix_mode == False:
            return self._apply_uniform_random_mixing_worker(audio, labels)
            
        elif self.sg_mix_mode == True:
            return audio, labels

    def _apply_uniform_random_mixing_worker(self, audio, labels):
        """ Worker-optimized uniform random mixing with unused tracking """
        if torch.rand(1).item() > self.focal_mixing_probability:
            return audio, labels
        
        overlap = random.choice(range(2, self.overlap_max + 1))
        
        current_audio = audio
        current_labels = labels
        
        for round_num in range(overlap - 1):
            random_example = self._pick_random_from_dataset(
                constraint_fn=None,
                prefer_unused_secondaries=True
            )
            
            alpha = random.uniform(0.1, 0.9)
            current_audio = alpha * current_audio + (1-alpha) * random_example['audio']
            current_audio = current_audio / torch.max(torch.abs(current_audio))
            
            current_labels = torch.maximum(current_labels, random_example['labels'])
        
        return current_audio, current_labels

    def _apply_snr_controlled_focal_mixing_worker(self, audio, labels):
        """Worker-optimized SNR-controlled mixing with dataset-wide sampling"""
        if torch.rand(1).item() > self.focal_mixing_probability:
            return audio, labels
        
        overlap = random.choice(range(2, self.overlap_max + 1))
        
        current_audio = audio
        current_labels = labels
        current_snr = calculate_snr_with_noisereduce(audio, 32000)
        
        for round_num in range(overlap - 1):
            constraint_fn = lambda seg: (not ((seg['id_snr'] <= -15) and (seg['id_entropy'] <= 0.35))) and (seg['id_snr'] > current_snr)
            random_example = self._pick_random_from_dataset(
                constraint_fn=constraint_fn, 
                prefer_unused_secondaries=True
            )
            
            target_snr = current_snr + 1.0
            current_audio = mix_audio_with_noise_tensor(
                random_example['audio'], current_audio, target_snr, 32000,
                mix_type='focal', mixing_params=self.mixing_params
            )
            
            current_snr = calculate_snr_with_noisereduce(current_audio, 32000)
            current_labels = torch.maximum(current_labels, random_example['labels'])
        
        return current_audio, current_labels

    def _apply_stage2_multi_snr_mixing(self, audio, labels, id_entropy, id_snr):
        """
        STAGE 2: MultiSNRDomainMixer in Worker ()
        
        - SG_Mix=None: NO noise mixing (pass through Stage 1 result)
        - SG_Mix=False: Random noise mixing (mix Stage 1 result with random noise profile)
        - SG_Mix=True: Multi-SNR domain mixing with hierarchical logic
        
        Returns: (final_audio_or_variants, final_labels) - ALWAYS returns both!
        """
        eligible = True
        
        if self.sg_mix_mode is None:
            return audio, labels
            
        elif self.sg_mix_mode == False:
            mixed_audio = self._apply_worker_random_noise_mixing(audio, labels)
            return mixed_audio, labels
            
        elif self.sg_mix_mode == True:
            snr_variants = self._apply_worker_multi_snr_variants(audio, labels, eligible, id_snr, id_entropy)
            return snr_variants, None
        
        return audio, labels

    def _apply_worker_multi_snr_variants(self, audio, labels, eligible, id_snr, id_entropy=None):
        """
        Generates Multi-SNR domain variants for the given sample.
        
        Caches the overlap decision and secondary examples across all categories
        to ensure consistency and improve processing speed.
        """
        available_categories = self.snr_categories
        
        snr_variants = {}
        
        hierarchical_decision = (self.overlap_max >= 2) and (torch.rand(1).item() < self.focal_mixing_probability)
        
        cached_overlap = None
        cached_secondary_examples = None
        
        for category in available_categories:
            if category.startswith("noise_"):
                base_cat = category.replace("noise_", "")
                
                profile = self._select_worker_noise_profile(base_cat, float('inf'))
                if profile is not None:
                    mixed_audio = self._load_worker_noise_profile(profile['filename'])
                else:
                    _, _, mixed_audio = calculate_snr_with_noisereduce(audio, 32000, return_denoised=True)
                    mixed_audio = self._make_exactly_5_seconds(torch.from_numpy(mixed_audio).float(), 32000)
                
                if self.LABELS == ["No_Bird", "Bird"]:
                    mixed_labels = torch.tensor([1.0, 0.0], dtype=torch.float32)
                elif self.LABELS == ["Bird"]:
                    mixed_labels = torch.tensor([0.0], dtype=torch.float32)
                else:
                    mixed_labels = torch.zeros_like(labels)
                    
            else:
                if hierarchical_decision:
                    mixed_audio, mixed_labels = self._apply_worker_hierarchical_mixing(
                        audio, labels, id_snr, category, id_entropy,
                        cached_overlap, cached_secondary_examples
                    )
                    if cached_overlap is None:
                        cached_overlap = getattr(self, '_last_overlap', None)
                    if cached_secondary_examples is None:
                        cached_secondary_examples = getattr(self, '_last_secondary_examples', None)
                else:
                    mixed_audio, mixed_labels = self._apply_worker_individual_mixing(
                        audio, labels, id_snr, category, id_entropy
                    )
            
            snr_variants[category] = {
                'audio': mixed_audio,
                'labels': mixed_labels
            }
        
        return {'snr_variants': snr_variants}

    def _apply_worker_hierarchical_mixing(self, audio, labels, id_snr, category, id_entropy=None, 
                                          cached_overlap=None, cached_secondary_examples=None):
        """
        Applies hierarchical mixing logic and caches secondary examples for reuse.
        """
        background, bg_labels = self._apply_worker_individual_mixing(audio, labels, id_snr, category, id_entropy)
        background_snr = calculate_snr_with_noisereduce(background, 32000)
        
        if cached_overlap is None:
            overlap = random.choice(range(2, self.overlap_max + 1))
            self._last_overlap = overlap
        else:
            overlap = cached_overlap
        
        if cached_secondary_examples is None:
            secondary_examples = []
            current_audio = background
            current_snr = background_snr
            
            for round_num in range(overlap - 1):
                constraint_fn = lambda seg: (not ((seg['id_snr'] <= -15) and (seg['id_entropy'] <= 0.35))) and (seg['id_snr'] > current_snr)
                secondary = self._pick_random_from_dataset(
                    constraint_fn=constraint_fn, 
                    prefer_unused_secondaries=True
                )
                
                secondary_examples.append(secondary)
                
                category_margin = 0.2
                target_snr = current_snr + category_margin
                background_mixing_params = {category: {'acceptable_tolerance': 0.025}}
                
                current_audio = mix_audio_with_noise_tensor(
                    secondary['audio'], current_audio, target_snr, 32000,
                    mix_type=category, mixing_params=background_mixing_params
                )
                
                current_snr = calculate_snr_with_noisereduce(current_audio, 32000)
            
            self._last_secondary_examples = secondary_examples
            secondary_examples_to_use = secondary_examples
        else:
            secondary_examples_to_use = cached_secondary_examples
        
        current_audio = background
        current_labels = bg_labels
        current_snr = background_snr
        
        for round_num, secondary in enumerate(secondary_examples_to_use):
            category_margin = 0.2
            target_snr = current_snr + category_margin
            background_mixing_params = {category: {'acceptable_tolerance': 0.025}}
            
            current_audio = mix_audio_with_noise_tensor(
                secondary['audio'], current_audio, target_snr, 32000,
                mix_type=category, mixing_params=background_mixing_params
            )
            
            current_snr = calculate_snr_with_noisereduce(current_audio, 32000)
            current_labels = torch.maximum(current_labels, secondary['labels'])
        
        return current_audio, current_labels

    def _apply_worker_random_noise_mixing(self, audio, labels):
        """
        Stage 2 (sg_mix_mode=False): Random noise mixing.
        Mixes Stage 1 output with a random noise profile.
        """
        if self.sound_profiles_data is None:
            return audio
        
        try:
            random_profile = self.sound_profiles_data.sample(1).iloc[0]
            noise_audio = self._load_worker_noise_profile(random_profile['filename'])
            
            alpha = random.uniform(0.1, 0.9)
            mixed_audio = alpha * audio + (1-alpha) * noise_audio
            mixed_audio = mixed_audio / torch.max(torch.abs(mixed_audio))
            
            return mixed_audio
        except Exception as e:
            print(f"Random noise mixing error: {e}")
            return audio

    def _apply_worker_individual_mixing(self, audio, labels, id_snr, category, id_entropy=None):
        """ Individual multi-SNR mixing """
        profile = self._select_worker_noise_profile(category, id_snr)
        
        if profile is not None:
            target_snr = float(profile[category])
            mixed_audio = self._mix_with_worker_profile(audio, profile, target_snr, category)
            return mixed_audio, labels
        else:
            return audio, labels

    def _get_worker_category_margin(self, category):
        """Get category margin for worker context"""
        margins = {
            'S1': 0.2, 'S2': 0.2, 'S3': 0.2, 'S4': 0.2, 'S5': 0.2
        }
        return margins.get(category, 3.0)

    def _get_worker_category_threshold(self, category):
        """Get category threshold for worker context"""
        thresholds = {
            'S1': -13.0, 'S2': -21.0, 'S3': -27.0, 'S4': -33.0, 'S5': -39.0
        }
        return thresholds.get(category, 0.0)

    def _get_worker_primary_label(self, labels):
        """Extract primary label for worker context"""
        for i, label_value in enumerate(labels):
            if label_value == 1:
                return f"label_{i}"
        return "default"

    def _select_worker_noise_profile(self, category, id_snr):
        """Select noise profile in worker context (optimized for 14K profile pool)"""
        if self.sound_profiles_data is None:
            return None
        
        if isinstance(id_snr, torch.Tensor):
            id_snr = id_snr.item()
        
        try:
            category_values = pd.to_numeric(self.sound_profiles_data[category], errors='coerce')
            
            candidates = self.sound_profiles_data[
                (category_values.notna()) & 
                (category_values < id_snr)
            ]
            
            if len(candidates) == 0:
                return None
            
            return candidates.sample(1).iloc[0]
        except Exception as e:
            print(f"Noise profile selection error for {category}: {e}")
            return None

    def _mix_with_worker_profile(self, audio, profile, target_snr, category):
        """ Mix with noise profile in worker context """
        try:
            noise_audio = self._load_worker_noise_profile(profile['filename'])
            return mix_audio_with_noise_tensor(
                audio, noise_audio, target_snr, 32000,
                mix_type=category, mixing_params=self.mixing_params
            )
        except Exception as e:
            print(f"Worker mixing error: {e}")
            return audio

    def _load_worker_noise_profile(self, filename):
        """Load noise profile in worker context with OPTIONAL caching"""
        disable_caching = hasattr(self, 'disable_caching') and self.disable_caching
        
        if not disable_caching and filename in self.noise_cache:
            full_audio = self.noise_cache[filename]
        else:
            noise_path = f"{self.sound_profiles_path}{filename}"
            full_audio = torch.from_numpy(np.load(noise_path)).float()
            if not disable_caching:
                self.noise_cache[filename] = full_audio
        
        target_length = 5 * 32000
        if len(full_audio) >= target_length:
            return full_audio[:target_length]
        else:
            repeat_count = int(torch.ceil(torch.tensor(target_length / len(full_audio))).item())
            return full_audio.repeat(repeat_count)[:target_length]

class BirdDataModule(LightningDataModule):
    def __init__(self, train_dataset, batch_size=64):
        super().__init__()
        self.train_dataset = train_dataset
        self.batch_size = batch_size

    def _collate_fn(self, batch):
        """
        Collates batches and flattens them into stacked tensors for efficient GPU processing.
        Tracks categories for CutMix alignment.
        """
        first_item = batch[0]
        
        if isinstance(first_item['raw_audio'], dict) and 'snr_variants' in first_item['raw_audio']:
            batch_audio_list = []
            batch_labels_list = []
            batch_categories_list = []
            
            for item in batch:
                audio_variants = item['raw_audio']['snr_variants']
                label_variants = item['labels']['snr_variants']
                
                for category in sorted(audio_variants.keys()):
                    batch_audio_list.append(audio_variants[category])
                    batch_labels_list.append(label_variants[category])
                    batch_categories_list.append(category)
            
            batch_audio = torch.stack(batch_audio_list)
            batch_labels = torch.stack(batch_labels_list)
            
            return {
                'raw_audio': batch_audio,
                'labels': batch_labels,
                'categories': batch_categories_list
            }
        else:
            batch_audio_list = [item['raw_audio'] for item in batch]
            batch_labels_list = [item['labels'] for item in batch]
            
            batch_audio = torch.stack(batch_audio_list)
            batch_labels = torch.stack(batch_labels_list)
            
            return {
                'raw_audio': batch_audio,
                'labels': batch_labels,
                'categories': None
            }
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=32,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            shuffle=True,
            collate_fn=self._collate_fn
        )

class BirdModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.classifier = backbone.classifier
        self.num_classes = backbone.classifier.out_features
        backbone.classifier = nn.Identity()

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class LitCls(LightningModule):
    def __init__(self, model, learning_rate=1e-4, min_learning_rate=1e-6, cutmix_p=0, augment_config=None, focal_mixing_p=0.5, cutmix_type='cyclic'):
        super().__init__()
        
        self.model = BirdModel(model)
        self.num_labels = self.model.num_classes
        
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        
        self.loss_type = None
        self.loss_fn = None
        
        self.cutmix_handler = CyclicCrossSNRCutMix(
            cutmix_probability=cutmix_p,
            cutmix_type=cutmix_type
        )

        if self.num_labels > 2:
            metric_basic = MetricCollection({
                "BCE": MeanMetric(),
                "CMAP": AveragePrecision(task='multilabel', num_labels=self.num_labels, average='macro'),
            })
            
            self.train_t1_acc = MultilabelT1Accuracy()
            self.train_basic = metric_basic.clone(prefix='train_')
            self.train_auroc = MultilabelAUROC(num_labels=self.num_labels, average='macro', prefix='train_')
            
            self.train_f1 = None
        else:
            from torchmetrics.classification import BinaryAUROC, MulticlassAUROC
            
            metric_basic = MetricCollection({
                "BCE": MeanMetric(),
            })
            
            self.train_t1_acc = None
            self.train_basic = metric_basic.clone(prefix='train_')
            self.train_auroc = BinaryAUROC() if self.num_labels == 2 else MulticlassAUROC(num_classes=self.num_labels, average='macro')
            self.train_f1 = None

    def configure_loss(self, loss_type):
        self.loss_type = loss_type
        if loss_type == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == 'ce':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Must be 'bce' or 'ce'")
            
    def forward(self, x):
        return self.model(x)

    def _get_probabilities(self, logits):
        if self.loss_type == 'bce':
            return torch.sigmoid(logits)
        elif self.loss_type == 'ce':
            return torch.softmax(logits, dim=1)

    def training_step(self, batch, batch_idx):
        batch_audio = batch['raw_audio']
        batch_labels = batch['labels']
        batch_categories = batch.get('categories', None)
        
        batch_specs = self._batch_audio_to_spectrograms(batch_audio)
        
        if batch_categories is None:
            final_specs, final_labels = self.cutmix_handler._apply_self_cutmix(
                batch_specs, batch_labels
            )
        else:
            final_specs, final_labels = self.cutmix_handler._apply_cyclic_cross_snr_cutmix_from_categories(
                batch_specs, batch_labels, batch_categories
            )
        
        logits = self.model(final_specs)
        train_loss = self.loss_fn(logits, final_labels)
        
        probs = self._get_probabilities(logits).detach()
        self.train_basic["BCE"].update(train_loss)
        
        if self.num_labels > 2:
            self.train_basic["CMAP"].update(probs, final_labels.int())
            self.train_t1_acc.update(probs, final_labels.int())
            self.train_auroc.update(probs, final_labels.int())
            
            self.log('train_loss', train_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log('train_cmap', self.train_basic["CMAP"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train_t1_acc', self.train_t1_acc.compute(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train_auroc', self.train_auroc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        else:
            if self.num_labels == 2:
                self.train_f1.update(probs, final_labels.argmax(dim=1))
                self.train_auroc.update(probs, final_labels.argmax(dim=1))
            else:
                self.train_f1.update(probs.squeeze(-1), final_labels.squeeze(-1).int())
                self.train_auroc.update(probs.squeeze(-1), final_labels.squeeze(-1).int())
                
            self.log('train_loss', train_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log('train_f1', self.train_f1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train_auroc', self.train_auroc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                
        return train_loss

    def _batch_audio_to_spectrograms(self, audio_batch):
        device = audio_batch.device
        B = audio_batch.shape[0]
        
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000,
            n_fft=2048,
            hop_length=512,
            n_mels=129,
            f_min=100,
            f_max=15000,
            power=2.0
        ).to(device)
        
        spec_batch = mel_transform(audio_batch)
        
        F, T = spec_batch.shape[1], spec_batch.shape[2]
        spec_flat = spec_batch.reshape(B, -1)
        
        spec_min = spec_flat.min(dim=1, keepdim=True)[0]
        spec_max = spec_flat.max(dim=1, keepdim=True)[0]
        spec_normalized = (spec_flat - spec_min) / (spec_max - spec_min + 1e-8)
        
        epsilon = 1e-6
        spec_log = torch.log(torch.clamp_min(spec_normalized, epsilon))
        
        spec_log_min = spec_log.min(dim=1, keepdim=True)[0]
        spec_log_max = spec_log.max(dim=1, keepdim=True)[0]
        spec_log_normalized = (spec_log - spec_log_min) / (spec_log_max - spec_log_min + 1e-8)
        
        spec_log_normalized = spec_log_normalized.view(B, F, T)
        
        color_map = _get_shared_color_map().to(device)
        x_rgb = spec_log_normalized.unsqueeze(-1) * color_map
        x_rgb = torch.flip(x_rgb, dims=[1])
        x_rgb = x_rgb.permute(0, 3, 1, 2)
        return x_rgb

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.min_learning_rate
        )
        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        self.train_basic.reset()
        if hasattr(self, 'train_auroc') and self.train_auroc is not None: self.train_auroc.reset()
        if hasattr(self, 'train_t1_acc') and self.train_t1_acc is not None: self.train_t1_acc.reset()
        if hasattr(self, 'train_f1') and self.train_f1 is not None: self.train_f1.reset()
        
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'train_dataset'):
            train_dataset = self.trainer.datamodule.train_dataset
            if hasattr(train_dataset, 'reset_secondary_tracking'):
                train_dataset.reset_secondary_tracking()

class CachedBatchLoader:
    def __init__(self, batches):
        self.batches = batches
        
    def __iter__(self):
        for batch in self.batches:
            yield batch
            
    def __len__(self):
        return len(self.batches)
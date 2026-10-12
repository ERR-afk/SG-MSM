"""
Main training script for the SG-MSM pipeline.
Handles configuration validation, data loading, and PyTorch Lightning trainer setup.
"""
import numpy as np
import pandas as pd
import json
import warnings
import argparse
import os

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-с', '--cfg', help='Model config file path', dest='cfg_path')
    args = vars(parser.parse_args())

    with open(args['cfg_path'], 'r') as f:
        CFG = json.load(f)

    domain_name = CFG.get('domain_name', 'NES')
    gpu_index = CFG.get('gpu_index', '0')
    wandb_name = CFG.get('wandb_name', CFG.get('name', 'experiment'))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    print(f"Set CUDA_VISIBLE_DEVICES to {gpu_index} for domain {domain_name}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = CFG.get('data_root', os.path.join(project_root, 'data'))

    base_path = f"{data_root}/Birdset_{domain_name}"
    train_lab_path = f'{base_path}/tmp/{domain_name}_train_prepared.csv'
    train_audio_path = f'{base_path}/tmp/train_audio_npy/'
    sound_profiles_path = f'{base_path}/tmp/domain_noise_npy/'
    sound_profiles_csv = f'{base_path}/tmp/sound_profile_npy_cluster.csv'
    
    models_path = f"{project_root}/model_weights/{domain_name}/{wandb_name}"
    os.makedirs(models_path, exist_ok=True)
    
    def validate_enhanced_config(CFG):
        """
        Validates configuration parameters and sets defaults for the SG-MSM pipeline.
        
        Args:
            CFG (dict): Configuration dictionary loaded from JSON.
            
        Returns:
            dict: The validated and populated configuration dictionary.
        """
        CFG['model_name'] = CFG.get('model_name', 'efficientnet_b3')
        CFG['loss_type'] = CFG.get('loss_type', 'bce')
        
        valid_loss_types = ['bce', 'ce']
        if CFG['loss_type'] not in valid_loss_types:
            raise ValueError(f"loss_type must be one of {valid_loss_types}, got: {CFG['loss_type']}")
        
        if 'SG_Mix' not in CFG:
            CFG['SG_Mix'] = True
            print("SG_Mix not specified, using default: True")
        if CFG['SG_Mix'] not in [True, False, None]:
            raise ValueError(f"SG_Mix must be true, false, or null, got: {CFG['SG_Mix']}")
        
        CFG['no_sg_msm_duplicate'] = CFG.get('no_sg_msm_duplicate', 1)
        CFG['focal_mixing_probability'] = CFG.get('focal_mixing_probability', 0.5)
        CFG['cross_snr_cutmix_probability'] = CFG.get('cross_snr_cutmix_probability', 0.3)
        CFG['snr_categories'] = CFG.get('snr_categories', ["S1", "S2", "S3", "S4"])
        CFG['overlap_max'] = CFG.get('overlap_max', 4)
        
        CFG['cutmix_type'] = CFG.get('cutmix_type', 'cyclic')
        valid_cutmix_types = ['cyclic', 'random', 'self']
        if CFG['cutmix_type'] not in valid_cutmix_types:
            raise ValueError(f"cutmix_type must be one of {valid_cutmix_types}, got: {CFG['cutmix_type']}")
        
        if CFG['cutmix_type'] in ['cyclic', 'random']:
            if not CFG['SG_Mix'] or len(CFG['snr_categories']) <= 1:
                print(f"WARNING: cutmix_type='{CFG['cutmix_type']}' requires SG_Mix=True and len(snr_categories)>1")
                print(f"Falling back to 'self' cutmix for single/no category case")
        
        print(f"CONFIGURATION SUMMARY:")
        print(f"  Model Name: {CFG['model_name']}")
        print(f"  Loss Type: {CFG['loss_type']}")
        print(f"  SG-MSM Configuration:")
        print(f"    SG_Mix: {CFG['SG_Mix']}")
        print(f"    Focal Mixing Probability: {CFG['focal_mixing_probability']}")
        print(f"    Cross-SNR CutMix Probability: {CFG['cross_snr_cutmix_probability']}")
        print(f"    CutMix Type: {CFG['cutmix_type']}")
        print(f"    SNR Categories: {CFG['snr_categories']}")
        print(f"    Overlap Max: {CFG['overlap_max']}")
        
        return CFG
    
    CFG = validate_enhanced_config(CFG)
    
    print("Importing PyTorch and Lightning components...")
    import torch
    torch.set_float32_matmul_precision('medium')
    
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
    
    import timm
    import wandb
    
    from training_utils import BirdDataset, BirdDataModule, LitCls
    
    base_batch_size = CFG['batch_size']
    print(f"PyTorch Training Batch Size: {base_batch_size} samples")

    if CFG['SG_Mix'] is True:
        effective_batch_size = base_batch_size * len(CFG['snr_categories'])
        print(f"Effective PyTorch Batch Size: {base_batch_size} × {len(CFG['snr_categories'])} SNR variants = {effective_batch_size} total samples")
    elif CFG['SG_Mix'] is None and CFG.get('no_sg_msm_duplicate', 1) > 1:
        dup = CFG['no_sg_msm_duplicate']
        effective_batch_size = base_batch_size * dup
        print(f"Effective PyTorch Batch Size: {base_batch_size} × {dup} duplicates (SG_Mix=null) = {effective_batch_size} total samples")
    
    print(f"LEARNING RATE CONFIGURATION:")
    print(f"   Learning Rate: {CFG['learning_rate']}")
    print(f"   Min Learning Rate: {CFG.get('min_learning_rate', 1e-5)}")
    print(f"   Epochs: {CFG['epochs_num']}")
    
    if CFG['SG_Mix'] is True:
        effective_batch_size = base_batch_size * len(CFG['snr_categories'])
        print(f"   Effective Batch Size: {effective_batch_size} (for learning rate scaling reference)")
    elif CFG['SG_Mix'] is None and CFG.get('no_sg_msm_duplicate', 1) > 1:
        effective_batch_size = base_batch_size * CFG['no_sg_msm_duplicate']
        print(f"   Effective Batch Size: {effective_batch_size} (for learning rate scaling reference)")

    data = pd.read_csv(train_lab_path)
    
    species_mapping_path = f'{base_path}/tmp/{domain_name}_species_mapping.csv'
    species_mapping = pd.read_csv(species_mapping_path)
    species_mapping = species_mapping.sort_values('species_id')
    LABELS = species_mapping['ebird_code'].tolist()
    
    print(f"Number of labels from species mapping: {len(LABELS)}")
    
    CFG['num_classes'] = len(LABELS)

    sound_profiles_data = pd.read_csv(sound_profiles_csv).fillna("")
    print(f"Loaded {len(sound_profiles_data)} sound profiles")

    train_df = data.reset_index(drop=True)
    low_entropy_count = len(data[data['id_entropy'] <= 0.35])
    print(f"Training dataset size: {len(train_df)} (including {low_entropy_count} low-entropy samples)")

    train_dataset = BirdDataset(df=train_df,
                                train_audio_path=train_audio_path,
                                sound_profiles_path=sound_profiles_path,
                                sound_profiles_data=sound_profiles_data,
                                sg_mix_mode=CFG['SG_Mix'],
                                snr_categories=CFG['snr_categories'],
                                focal_mixing_probability=CFG['focal_mixing_probability'],
                                overlap_max=CFG['overlap_max'],
                                labels=LABELS,
                                disable_caching=True,
                                no_sg_msm_duplicate=CFG['no_sg_msm_duplicate'])
                               
    datamodule = BirdDataModule(train_dataset, batch_size=CFG['batch_size'])

    print(f"Creating PyTorch CNN model: {CFG['model_name']}...")
    model = timm.create_model(
        CFG['model_name'], pretrained=True,
        num_classes=len(LABELS), 
        in_chans=3,
    )
    
    lit_cls = LitCls(
        model, 
        learning_rate=CFG['learning_rate'],
        min_learning_rate=CFG.get('min_learning_rate', 1e-5),
        cutmix_p=CFG['cross_snr_cutmix_probability'],
        focal_mixing_p=CFG['focal_mixing_probability'],
        augment_config=train_dataset,
        cutmix_type=CFG['cutmix_type']
    )
    
    lit_cls.configure_loss(CFG['loss_type'])

    checkpoint_callback = ModelCheckpoint(
        dirpath=models_path,
        filename='{epoch}',
        save_top_k=-1,
        save_weights_only=False,
        save_on_train_epoch_end=True,
        save_last=False,
        every_n_epochs=1
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    rich_progress = RichProgressBar()

    wandb_project = f"BirdCLEF_{domain_name}"
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=wandb_name
    )

    trainer = Trainer(
        logger=wandb_logger,
        num_sanity_val_steps=0,
        max_epochs=CFG['epochs_num'],
        accumulate_grad_batches=1,
        callbacks=[rich_progress, lr_monitor, checkpoint_callback],
        log_every_n_steps=1,
        enable_progress_bar=True,
        accelerator='gpu',
        devices=1
    )
    
    resume_ckpt_path = None
    if os.path.exists(models_path):
        import glob
        import re
        ckpt_files = glob.glob(os.path.join(models_path, 'epoch=*.ckpt'))
        if ckpt_files:
            def get_epoch(filename):
                match = re.search(r'epoch=(\d+)', filename)
                return int(match.group(1)) if match else -1
            
            latest_ckpt = max(ckpt_files, key=get_epoch)
            resume_ckpt_path = latest_ckpt
            print(f"Auto-resuming from {latest_ckpt}")
            
    trainer.fit(lit_cls, datamodule=datamodule, ckpt_path=resume_ckpt_path)
    trainer.save_checkpoint(f'{models_path}/{wandb_name}.ckpt')
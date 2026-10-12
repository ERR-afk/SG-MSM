#!/usr/bin/env python3
import os
import json
import subprocess
import glob
from pathlib import Path

def main():
    # Define the 6 domains required for the paper
    domains = ["PER", "HSN", "SSW", "SNE", "NES", "UHH"]
    
    # Path setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(base_dir, "configs")
    tmp_config_dir = os.path.join(configs_dir, "generated_configs")
    
    # Ensure temporary config directory exists
    os.makedirs(tmp_config_dir, exist_ok=True)

    # Automatically find all 5 base JSON configs
    # We sort them to ensure consistent ordering (config_1, config_2, etc.)
    base_configs = sorted(glob.glob(os.path.join(configs_dir, "config_*.json")))
    
    if not base_configs:
        print(f"Error: No config files found in {configs_dir}")
        return

    print("=================================================")
    print("   SG-MSM Training Automation Script")
    print("=================================================")
    print(f"Found {len(base_configs)} base configurations.")
    print(f"Will run across {len(domains)} domains: {', '.join(domains)}")
    print(f"Total experiments to run: {len(base_configs) * len(domains)}")
    print("=================================================\n")

    total_runs = len(base_configs) * len(domains)
    current_run = 1

    for config_path in base_configs:
        config_filename = os.path.basename(config_path)
        
        # Load the base config
        with open(config_path, 'r') as f:
            base_cfg = json.load(f)
            
        base_wandb_name = base_cfg.get("wandb_name", "experiment")
        base_name = base_cfg.get("name", "experiment")

        for domain in domains:
            print(f"\n[{current_run}/{total_runs}] Starting run for Domain: {domain} using Config: {config_filename}")
            
            # Create a deep copy of the configuration dict for this specific domain
            run_cfg = base_cfg.copy()
            
            # Inject the domain specific overrides
            run_cfg["domain_name"] = domain
            
            # Ensure wandb logs and checkpoint folders are cleanly separated by domain
            # We strip any existing '_PER' or '_SNE' just in case they were left in the base config
            clean_wandb = base_wandb_name.replace("_SNE", "").replace("_PER", "")
            clean_name = base_name.replace("_SNE", "").replace("_PER", "")
            
            run_cfg["wandb_name"] = f"{clean_wandb}_{domain}"
            run_cfg["name"] = f"{clean_name}_{domain}"
            
            # Generate a temporary config file path
            tmp_config_filename = f"{config_filename.replace('.json', '')}_{domain}.json"
            tmp_config_path = os.path.join(tmp_config_dir, tmp_config_filename)
            
            # Save the modified config
            with open(tmp_config_path, 'w') as f:
                json.dump(run_cfg, f, indent=4)
                
            print(f"  -> Generated specific config: {tmp_config_path}")
            print(f"  -> Launching training process...")
            
            # Execute the training script
            # We use subprocess.run which will block until the training script completes
            # If the script crashes, it throws an error and stops the loop, which is safe.
            training_script = os.path.join(base_dir, "src", "training.py")
            try:
                subprocess.run(
                    ["python3", training_script, "-c", tmp_config_path],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Error: Training failed for {domain} with {config_filename}.")
                print("Stopping the automation script to prevent further errors.")
                return
            
            print(f"✅ Successfully completed {domain} with {config_filename}.")
            current_run += 1

    print("\n🎉 All experiments have completed successfully!")

if __name__ == "__main__":
    main()

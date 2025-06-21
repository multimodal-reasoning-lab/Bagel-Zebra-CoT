import torch
import os
import json
from pprint import pprint

def load_and_print_data_status(checkpoint_path):
    """
    Load and print the data status from a checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint directory (e.g., "results/checkpoints/0002000")
    """
    data_status_path = os.path.join(checkpoint_path, "data_status.pt")
    
    if not os.path.exists(data_status_path):
        print(f"❌ No data_status.pt found at: {data_status_path}")
        return
    
    print(f"📁 Loading data status from: {data_status_path}")
    print("=" * 80)
    
    # Load the data status
    data_status = torch.load(data_status_path, map_location="cpu", weights_only=True)
    
    print(f"🔍 Data status type: {type(data_status)}")
    print(f"📊 Number of ranks: {len(data_status) if isinstance(data_status, list) else 'Not a list'}")
    print("=" * 80)
    
    # Print detailed breakdown
    if isinstance(data_status, list):
        for rank_idx, rank_data in enumerate(data_status):
            print(f"\n🎯 RANK {rank_idx}:")
            print("-" * 40)
            
            if isinstance(rank_data, dict):
                for dataset_name, worker_data in rank_data.items():
                    print(f"  📂 Dataset: {dataset_name}")
                    
                    if isinstance(worker_data, dict):
                        for worker_id, position in worker_data.items():
                            print(f"    👷 Worker {worker_id}: {position}")
                    else:
                        print(f"    📍 Position: {worker_data}")
                    print()
            else:
                print(f"  ⚠️  Unexpected rank data type: {type(rank_data)}")
                print(f"      Content: {rank_data}")
    else:
        print("⚠️  Data status is not a list. Content:")
        pprint(data_status)

def find_latest_checkpoint(checkpoint_dir="results/checkpoints"):
    """Find the latest checkpoint directory"""
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    step_dirs = [d for d in os.listdir(checkpoint_dir) 
                 if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.isdigit()]
    
    if not step_dirs:
        print(f"❌ No checkpoint steps found in: {checkpoint_dir}")
        return None
    
    latest_step = max(step_dirs, key=int)
    latest_path = os.path.join(checkpoint_dir, latest_step)
    print(f"🔍 Found latest checkpoint: {latest_path}")
    return latest_path

if __name__ == "__main__":

    load_and_print_data_status("/dev/shm/results_2/checkpoints/0000250")
    
    # Option 3: Examine all checkpoints
    # checkpoint_dir = "results/checkpoints"
    # if os.path.exists(checkpoint_dir):
    #     for step_dir in sorted(os.listdir(checkpoint_dir)):
    #         if step_dir.isdigit():
    #             checkpoint_path = os.path.join(checkpoint_dir, step_dir)
    #             print(f"\n{'='*100}")
    #             print(f"CHECKPOINT STEP: {step_dir}")
    #             print(f"{'='*100}")
    #             load_and_print_data_status(checkpoint_path)
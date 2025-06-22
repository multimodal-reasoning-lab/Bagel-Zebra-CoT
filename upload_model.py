#!/usr/bin/env python3
"""
Script to convert model to bfloat16 and upload to Hugging Face Hub
"""

import os
import torch
from safetensors.torch import load_file, save_file
from huggingface_hub import HfApi, upload_file, create_repo
import argparse

def convert_to_bf16(input_path, output_path):
    """
    Convert model from safetensors to bfloat16 and save
    
    Args:
        input_path (str): Path to input model.safetensors
        output_path (str): Path to save the bf16 version
    """
    print(f"Loading model from {input_path}...")
    
    # Load the safetensors file
    state_dict = load_file(input_path)
    
    print("Converting to bfloat16...")
    # Convert all tensors to bfloat16
    bf16_state_dict = {}
    for key, tensor in state_dict.items():
        if tensor.dtype in [torch.float32, torch.float16]:
            bf16_state_dict[key] = tensor.to(dtype=torch.bfloat16)
        else:
            # Keep non-float tensors as is (like int tensors for embeddings)
            bf16_state_dict[key] = tensor
    
    print(f"Saving bf16 model to {output_path}...")
    # Save as safetensors
    save_file(bf16_state_dict, output_path)
    
    # Print file size comparison
    original_size = os.path.getsize(input_path) / (1024**3)
    new_size = os.path.getsize(output_path) / (1024**3)
    print(f"Original size: {original_size:.2f} GB")
    print(f"BF16 size: {new_size:.2f} GB")
    print(f"Size reduction: {((original_size - new_size) / original_size * 100):.1f}%")
    
    return output_path

def upload_model_files(repo_id, checkpoint_dir, bf16_model_path, token=None):
    """
    Upload model files to Hugging Face Hub
    """
    # Initialize the API
    api = HfApi(token=token)
    
    # Create repository
    print(f"Creating repository {repo_id}...")
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            repo_type="model",
            exist_ok=True
        )
        print(f"✓ Repository {repo_id} created/verified")
    except Exception as e:
        print(f"Repository creation info: {str(e)}")
    
    # Files to upload
    files_to_upload = [
        ("data_status.pt", os.path.join(checkpoint_dir, "data_status.pt")),
        ("model_bf16.safetensors", bf16_model_path)
    ]
    
    print(f"\nUploading files to {repo_id}...")
    
    for repo_filename, local_path in files_to_upload:
        if not os.path.exists(local_path):
            print(f"Warning: {local_path} not found, skipping...")
            continue
            
        print(f"Uploading {repo_filename}...")
        file_size = os.path.getsize(local_path)
        print(f"  File size: {file_size / (1024**3):.2f} GB")
        
        try:
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_filename,
                repo_id=repo_id,
                token=token,
                repo_type="model"
            )
            print(f"✓ Successfully uploaded {repo_filename}")
            
        except Exception as e:
            print(f"✗ Failed to upload {repo_filename}: {str(e)}")
    
    print(f"\nUpload completed! Check your model at: https://huggingface.co/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Convert model to bf16 and upload to Hugging Face Hub")
    parser.add_argument("--checkpoint", type=str, default="110",
                        help="Checkpoint number (will be formatted as 0000XXX)")
    parser.add_argument("--repo_id", type=str, default=None, 
                        help="Repository ID on Hugging Face (auto-generated if not provided)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to checkpoint directory (auto-generated if not provided)")
    parser.add_argument("--input_model", type=str, default=None,
                        help="Path to input model.safetensors (auto-generated if not provided)")
    parser.add_argument("--output_model", type=str, 
                        default="model_bf16.safetensors",
                        help="Path to save bf16 model")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face token")
    parser.add_argument("--convert_only", action="store_true",
                        help="Only convert to bf16, don't upload")
    
    args = parser.parse_args()
    
    # Format checkpoint number with 0000 prefix
    checkpoint_formatted = f"0000{args.checkpoint}"
    
    # Auto-generate paths if not provided
    if args.repo_id is None:
        args.repo_id = f"vlm-reasoning-cot/h200-ckpt-{checkpoint_formatted}"
    
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"results/checkpoints/{checkpoint_formatted}"
    
    if args.input_model is None:
        args.input_model = f"results/checkpoints/{checkpoint_formatted}/model.safetensors"
    
    # Convert to bf16
    bf16_model_path = convert_to_bf16(args.input_model, args.output_model)
    
    if not args.convert_only:
        # Upload to HF
        upload_model_files(args.repo_id, args.checkpoint_dir, bf16_model_path, args.token)
    else:
        print("Conversion completed. Skipping upload (--convert_only flag used)")

if __name__ == "__main__":
    main()
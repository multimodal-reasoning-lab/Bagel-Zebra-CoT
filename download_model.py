from huggingface_hub import snapshot_download

HF_HOME = "/dev/shm/"

# Download Bagel-Zebra-CoT
save_dir = HF_HOME + "models/Bagel-Zebra-CoT"
repo_id = "multimodal-reasoning-lab/Bagel-Zebra-CoT"
cache_dir = HF_HOME + save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)

from huggingface_hub import snapshot_download

HF_HOME = "/mnt/wsfuse/kaiyuyue/cache/huggingface"
repo_id = "multimodal-reasoning-lab/Bagel-Zebra-CoT"

snapshot_download(
    cache_dir=HF_HOME,
    repo_id=repo_id,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)

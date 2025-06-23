### BAGEL Training Zebra-CoT

## Setup

install conda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
eval "$(/root/miniconda3/bin/conda shell.bash hook)" && conda init
```

```bash
git clone https://github.com/LeonLixyz/bagel-training
cd bagel-training
conda create -n bagel python=3.10 -y
conda activate bagel
pip install -r requirements.txt
pip install flash_attn --no-build-isolation
```


download data
```bash
apt-get install -y git-lfs
mkdir -p /dev/shm/data/Zebra-CoT
git clone https://huggingface.co/datasets/vlm-reasoning-cot/Zebra-CoT-tar /dev/shm/data/
git clone https://huggingface.co/datasets/vlm-reasoning-cot/Zebra-CoT-text-tar /dev/shm/data/text
cd text/
tar -xzf zebra_cot.jsonl.tar.gz
```

download model
```bash
python download_model.py
```


### Inference

```bash
python infz_bf16.py
```
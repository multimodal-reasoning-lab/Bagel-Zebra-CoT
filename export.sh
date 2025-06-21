export HF_HOME=/dev/shm/
export PYTHONPATH=$PYTHONPATH:$(pwd)
conda install -c nvidia cuda-nvcc 
pip install flash-attn --no-build-isolation
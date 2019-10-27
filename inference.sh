#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --output=decode.log.out
#SBATCH --error=decode.log.err

source ~/miniconda3/bin/activate nmt_env_standard
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64/:/users/limsi_nmt/minhquang/cuDNNv6/lib64

which python
python -c 'import tensorflow; print("tensorflow OK"); import opennmt; print("opennmt OK")'
python -u inference.py --config_file /mnt/beegfs/home/pham/multi-domain-nmt/LDR_7/configs/sparse_src_masking_71.yml --eval_step 195000

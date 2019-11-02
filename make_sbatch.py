import sys
index = (sys.argv[1])
f = open("sparse_src_masking_%s.sh"%index,"w")
print("#!/bin/bash",file=f)
print("#SBATCH --gres=gpu:3",file=f)
print("#SBATCH --nodes=1",file=f)
print("#SBATCH --time=24:00:00",file=f)
print("#SBATCH --cpus-per-task=5",file=f)
print("#SBATCH --output=sparse_src_masking_%s.log.out"%index,file=f)
print("#SBATCH --error=sparse_src_masking_%s.log.err"%index,file=f)
print("#SBATCH --partition=all",file=f)
print("source ~/miniconda3/bin/activate tf2",file=f)
print("export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64/",file=f)

print("which python",file=f)
print("python -c 'import tensorflow; print(\"tensorflow OK\"); import opennmt; print(\"opennmt OK\")'",file=f)
print("python -u practice.py train --config configs/config_%s.yml"%index,file=f)

f.close()

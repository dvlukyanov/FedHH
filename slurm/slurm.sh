#!/bin/bash
#SBATCH --job-name=ddp
#SBATCH --output=ddp-output_%j_%N.txt
#SBATCH --error=ddp-error_%j_%N.txt
#SBATCH --ntasks=4  # number of tasks
#SBATCH --nodes=4 # number of nodes
#SBATCH --ntasks-per-node=1 # tasks per node
#SBATCH --cpus-per-task=8 # cpu per task
#SBATCH --gpus-per-node v100:2
#SBATCH --mem 16gb
#SBATCH --time=00:30:00
#SBATCH --constraint interconnect_hdr

CONDA_BIN=$(whereis -b conda | awk '{print $2}' | head -n 1)
SERVER=$(hostname -s)

srun --unbuffered --ntasks=4 --output=ddp-output_%j_%N_%t.txt --error=ddp-error_%j_%N_%t.txt ${CONDA_BIN} run -n thesis /usr/bin/python3 /home/dlukyan/fedhh/code/ddp/exp.py --server=${SERVER} --port=12345
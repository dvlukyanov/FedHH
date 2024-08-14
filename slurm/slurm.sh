#!/bin/bash
#SBATCH --job-name=fedhh
#SBATCH --output=fedhh-output_%j_%N.txt
#SBATCH --error=fedhh-error_%j_%N.txt
#SBATCH --ntasks=4  # number of tasks
#SBATCH --nodes=4 # number of nodes
#SBATCH --ntasks-per-node=1 # tasks per node
#SBATCH --cpus-per-task=8 # cpu per task
#SBATCH --gpus-per-node v100:1
#SBATCH --mem 16gb
#SBATCH --time=00:30:00
#SBATCH --constraint interconnect_hdr

CONDA_ENV_NAME="thesis"
CONDA_ENV_PATH=$(conda env list | grep "$CONDA_ENV_NAME" | awk '{print $2}')
if [ -z "$CONDA_ENV_PATH" ]; then
    echo "Conda environment '$CONDA_ENV_NAME' not found."
    exit 1
fi
PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"

CONDA_BIN=$(whereis -b conda | awk '{print $2}' | head -n 1)
SERVER=$(hostname)
NOTIFIER_SLACK=[SLACK_WEBHOOK_URL_PLACEHOLDER]

srun --unbuffered --ntasks=4 --output=fedhh-output_%j_%N_%t.txt --error=fedhh-error_%j_%N_%t.txt ${CONDA_BIN} run -n thesis ${PYTHON_BIN} /storage/fedhh/code/architectures/fedhh.py --server=${SERVER} --port=12345 --workers=3 --data=/storage/fedhh/data --slack=${NOTIFIER_SLACK}
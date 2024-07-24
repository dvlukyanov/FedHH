#!/bin/bash
#SBATCH --job-name=ddp
#SBATCH --output=ddp-output_%j_%N.txt
#SBATCH --error=ddp-error_%j_%N.txt
#SBATCH --ntasks=4  # number of tasks
#SBATCH --nodes=2 # number of nodes
#SBATCH --ntasks-per-node=2 # tasks per node
#SBATCH --cpus-per-task=8 # cpu per task
#SBATCH --gpus-per-node v100:2
#SBATCH --mem 16gb
#SBATCH --time=00:30:00
#SBATCH --constraint interconnect_hdr

CONDA_BIN=$(whereis -b conda | awk '{print $2}' | head -n 1)
SERVER=$(hostname -s)

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=(${nodes// / })
echo "Nodes: ${nodes_array[@]}"

srun --unbuffered --ntasks=1 --nodelist=${nodes_array[0]} --output=ddp-output_%j_%N_%t.txt --error=ddp-error_%j_%N_%t.txt ${CONDA_BIN} run -n thesis /usr/bin/python3 /home/dlukyan/fedhh/code/ddp/exp.py --server=${nodes_array[0]} --port=12345 --main
sleep 5
srun --unbuffered --ntasks=1 --nodelist=${nodes_array[0]} --output=ddp-output_%j_%N_%t.txt --error=ddp-error_%j_%N_%t.txt ${CONDA_BIN} run -n thesis /usr/bin/python3 /home/dlukyan/fedhh/code/ddp/exp.py --server=${nodes_array[0]} --port=12345
srun --unbuffered --ntasks=1 --nodelist=${nodes_array[1]} --output=ddp-output_%j_%N_%t.txt --error=ddp-error_%j_%N_%t.txt ${CONDA_BIN} run -n thesis /usr/bin/python3 /home/dlukyan/fedhh/code/ddp/exp.py --server=${nodes_array[0]} --port=12345
srun --unbuffered --ntasks=1 --nodelist=${nodes_array[1]} --output=ddp-output_%j_%N_%t.txt --error=ddp-error_%j_%N_%t.txt ${CONDA_BIN} run -n thesis /usr/bin/python3 /home/dlukyan/fedhh/code/ddp/exp.py --server=${nodes_array[0]} --port=12345

# another version

srun --unbuffered --ntasks=1 --nodelist=${nodes_array[0]} --output=ddp-output_%j_%N_%t.txt --error=ddp-error_%j_%N_%t.txt ${CONDA_BIN} run -n thesis /usr/bin/python3 /home/dlukyan/fedhh/code/ddp/exp.py --server=${nodes_array[0]} --port=12345 --main

sleep 10

total_tasks=$SLURM_NTASKS
num_nodes=${#nodes[@]}
tasks_per_node=$((total_tasks / num_nodes))

distributed_tasks=1
for node in "${nodes[@]}"; do
    while [ $distributed_tasks -lt $tasks_per_node ]; do
        srun --unbuffered --ntasks=1 -w ${node} --output=ddp-output_%j_%N_%t.txt --error=ddp-error_%j_%N_%t.txt ${CONDA_BIN} run -n thesis /usr/bin/python3 /home/dlukyan/fedhh/code/ddp/exp.py --server=${nodes_array[0]} --port=12345
        distributed_tasks=$((distributed_tasks + 1))
    done
    distributed_tasks=0
done
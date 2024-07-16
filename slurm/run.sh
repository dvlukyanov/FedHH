#!/bin/bash

#SBATCH --job-name fedhh-[model]
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 32gb
#SBATCH --time 72:00:00
#SBATCH --gpus-per-node v100:2
#SBATCH --output=[output directory]/result-%j.txt
#SBATCH --error=[output directory]/error-%j.txt

conda run -n [conda env] python3 [code directory]/model_selection.py --model=[model] --img_dir=[images directory] --labels_file=[labels file] --seed=[seed]
#!/bin/bash

#SBATCH --job-name=automark
#SBATCH --output=automark.txt
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mail-user=berger@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=2
#SBATCH --partition=mlgroup
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --qos=batch
# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

# JOB STEPS (example: write hostname to output file, and wait 1 minute)
source /home/students/berger/anaconda3/bin/activate gpu 
srun python3 -m automark train ./configs/humanmt_fixed_bert.yml

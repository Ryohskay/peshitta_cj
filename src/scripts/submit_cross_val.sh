#!/bin/bash

#SBATCH --job-name=cross_validate
#SBATCH --output=out/save_mlm_train_%j.out
#SBATCH --error=out/save_mlm_train_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=compute
#SBATCH --mail-type=END,FAIL,REQUEUE

module load python/3.11.7

source /home/u05rn21/venv/bin/activate
python --version

cd /home/u05rn21/peshitta_cj_ssh
python -m pip install --upgrade pip
python -m pip install numpy
python -m pip install text-fabric

python -m src.scripts.save_mlm_text_data

deactivate
exit 0

#!/bin/bash

#SBATCH --job-name=cross_validate
#SBATCH --output=out/cross_validate_%j.out
#SBATCH --error=out/cross_validate_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --partition=compute
#SBATCH --mail-type=END,FAIL,REQUEUE

module load python/3.11.7

source /home/u05rn21/venv/bin/activate
python --version

cd /home/u05rn21/peshitta_cj_ssh
src/scripts/ensure_dir.sh

python -m pip install --upgrade pip
python -m pip install numpy
python -m pip install scikit-learn
python -m pip install nltk
python -m pip install text-fabric

python -m src.classifier.cross_validation

deactivate
exit 0

#!/bin/bash

#SBATCH --job-name=etcbc_eval
#SBATCH --output=out/etcbc_eval_%j.out
#SBATCH --error=out/etcbc_eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --partition=compute
#SBATCH --mail-type=END,FAIL,REQUEUE

module load python/3.11.7

source /home/u05rn21/venv/bin/activate
python --version

cd /home/u05rn21/peshitta_cj_ssh
src/scripts/ensure_dirs.sh

python -m pip install --upgrade pip
python -m pip install numpy
python -m pip install scikit-learn
python -m pip install nltk
python -m pip install text-fabric

python -m src.classifier.etcbc_aa_eval

deactivate
exit 0

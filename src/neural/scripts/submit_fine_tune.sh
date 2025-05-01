#!/bin/bash

#SBATCH --job-name=fine_tune
#SBATCH --output=out/fine_tune_%j.out
#SBATCH --error=out/fine_tune_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=compute
#SBATCH --mail-type=END,FAIL,REQUEUE

module load python/3.11.7

source $HOME/venv/bin/activate
python3 --version

python3 -m pip install --upgrade pip
python3 -m pip install -r $HOME/peshitta_cj_ssh/src/neural/requirements.txt
python3 -m pip install numpy
python3 -m pip install transformers
python3 -m pip install datasets
# python -m pip install scikit-learn
# python -m pip install text-fabric

python3 $HOME/peshitta_cj_ssh/src/neural/run_mlm.py \
    --model_name_or_path google-bert/bert-base-multilingual-uncased \
    --train_file $HOME/peshitta_cj_ssh/src/neural/data/all_verses.txt \
    --validation_split_percentage 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --line_by_line \
    --output_dir $HOME/mbert_uncased_aa

deactivate
exit 0

#!/bin/bash

echo "Running various classifiers. This may take many hours."
./src/scripts/ensure_dirs.sh
poetry run python -m src.classifier.cal_aa_eval | tee out/cal_aa_eval_log.txt
poetry run python -m src.classifier.etcbc_aa_eval | tee out/etcbc_aa_eval_log.txt
poetry run python -m src.classifier.production | tee out/production_log.txt

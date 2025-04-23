#!/bin/bash

python3 -m src.classifier.cal_aa_eval | tee out/cal_aa_eval_log.txt
python3 -m src.classifier.etcbc_aa_eval | tee out/etcbc_aa_eval_log.txt
python3 -m src.classifier.production | tee out/production_log.txt

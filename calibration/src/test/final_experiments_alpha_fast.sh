#!/bin/bash

python -u ./final_experiments_alpha_fast.py -N 100 -M 10 -K 10 -u 0.01 |& tee ./logs/final_experiments_alpha_fast_100_10_10_01_nowuml_sec.txt &
python -u ./final_experiments_alpha_fast.py -N 100 -M 100 -K 10 -u 0.01 |& tee ./logs/final_experiments_alpha_fast_100_100_10_01_nowuml_sec.txt & 
python -u ./final_experiments_alpha_fast.py -N 100 -M 10 -K 100 -u 0.01 |& tee ./logs/final_experiments_alpha_fast_100_10_100_01_nowuml_sec.txt &
python -u ./final_experiments_alpha_fast.py -N 100 -M 10 -K 10 -u 0.1 |& tee ./logs/final_experiments_alpha_fast_100_10_10_1_nowuml_sec.txt &
#python -u ./final_experiments_alpha_fast.py -N 1000 -M 10 -K 10 -u 0.01 |& tee ./logs/final_experiments_alpha_fast_1000_10_10_01.txt
#python -u ./final_experiments_alpha_fast.py -N 100 -M 10 -K 100 -u 0.01 |& tee ./logs/final_experiments_alpha_fast_100_10_10_01_1000_DEBUG.txt

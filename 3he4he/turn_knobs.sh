#!/bin/zsh

python3 train_fat.py 0.001 500 0.3
python3 train_fat.py 0.01 700 0.3

python3 mcmc_fat.py 0.001 500 0.3
python3 mcmc_fat.py 0.01 700 0.3

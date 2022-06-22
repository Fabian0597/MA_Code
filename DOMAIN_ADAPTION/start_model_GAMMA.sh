#!/bin/bash

#for looping through different models with different configurations

feature_of_interest="C:s_ist/X"
num_epochs=10
GAMMAs=(0.8 1 1.2 1.5 1.7 2 2.2 2.5)
num_pool=1

for GAMMA in ${GAMMAs[@]}; do
  python3 main.py $feature_of_interest $num_epochs $GAMMA $num_pool
done
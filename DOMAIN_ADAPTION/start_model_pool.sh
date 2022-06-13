#!/bin/bash

#for looping through different models with different configurations
feature_of_interest="C:s_ist/X"
num_epochs=10
GAMMA=1.8

nums_pool=(0 1 2 3)
for num_pool in ${nums_pool[@]}; do
    python3 main.py $feature_of_interest $num_epochs $GAMMA $num_pool
done
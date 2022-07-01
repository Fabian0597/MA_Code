#!/bin/bash

#for looping through different models with different configurations
feature_of_interest="D:P_mech./X"
num_epochs=30
GAMMAs=(0 0.1)
MMD_layer_activation_flag=( True True True False False False )

nums_pool=(0 1 2 3)
for num_pool in ${nums_pool[@]}; do
    for GAMMA in ${GAMMAs[@]}; do
        python3 main.py $feature_of_interest $num_epochs $GAMMA $num_pool ${MMD_layer_activation_flag[@]} 
    done
done
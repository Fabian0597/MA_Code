#!/bin/bash

#for looping through different models with different configurations
feature_of_interest="D:P_mech./X"
num_epochs=30
GAMMAs=(0 0.1)
GAMMA_reduction = 0.98
MMD_layer_activation_flag=( True True True False False False )

nums_pool=(0 1 2 3)

experiment_number=0

rm -r runs

for num_pool in ${nums_pool[@]}; do
    for GAMMA in ${GAMMAs[@]}; do

	((experiment_number=experiment_number+1))
        experiment_name="experiment_${experiment_number}"
        echo $experiment_name

        python3 main.py $experiment_name $num_epochs $GAMMA $GAMMA_reduction $num_pool ${MMD_layer_activation_flag[@]} $feature_of_interest 
    done
done

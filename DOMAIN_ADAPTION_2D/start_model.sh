#!/bin/bash

#for looping through different models with different configurations
#"C:s_ist/X" "C:s_soll/X" "C:s_diff/X" "C:v_(n_ist)/X" "C:v_(n_soll)/X" "C:P_mech./X" "C:Pos._Diff./X" "C:I_ist/X" "C:I_soll/X" "C:x_bottom" "C:y_bottom" "C:z_bottom" "C:x_nut" "C:y_nut" "C:z_nut" "C:x_top" "C:y_top" 


feature_of_interest=( "D:P_mech./X" "D:I_ist/X" "D:I_soll/X" )
num_epochs=50
GAMMA=0.1
GAMMA_reduction=0.98
num_pool=2
MMD_layer_activation_flag=( True False False False True True)

rm -r runs

python3 main.py $num_epochs $GAMMA $GAMMA_reduction $num_pool ${MMD_layer_activation_flag[@]} ${feature_of_interest[@]}


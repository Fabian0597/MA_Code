#!/bin/bash

#for looping through different models with different configurations
#"C:s_ist/X" "C:s_soll/X" "C:s_diff/X" "C:v_(n_ist)/X" "C:v_(n_soll)/X" "C:P_mech./X" "C:Pos._Diff./X" "C:I_ist/X" "C:I_soll/X" "C:x_bottom" "C:y_bottom" "C:z_bottom" "C:x_nut" "C:y_nut" "C:z_nut" "C:x_top" "C:y_top" 


features_of_interest=("C:z_top" "D:s_ist/X" "D:s_soll/X" "D:s_diff/X" "D:v_(n_ist)/X" "D:v_(n_soll)/X"
        "D:P_mech./X" "D:Pos._Diff./X" "D:I_ist/X" "D:I_soll/X" "D:x_bottom" "D:y_bottom" "D:z_bottom"
        "D:x_nut" "D:y_nut" "D:z_nut" "D:x_top" "D:y_top" "D:z_top" "S:x_bottom" "S:y_bottom" "S:z_bottom"
        "S:x_nut" "S:y_nut" "S:z_nut" "S:x_top" "S:y_top" "S:z_top" "S:Nominal_rotational_speed[rad/s]"
        "S:Actual_rotational_speed[µm/s]" "S:Actual_position_of_the_position_encoder(dy/dt)[µm/s]"
        "S:Actual_position_of_the_motor_encoder(dy/dt)[µm/s]")
num_epochs=50
GAMMA=0.1
num_pool=2

for feature_of_interest in ${features_of_interest[@]}; do
  python3 main.py $feature_of_interest $num_epochs $GAMMA $num_pool
done

#!/bin/bash

args1=( 10 11 12 )
arg2=1.8
arg3=3
for arg1 in "${args1[@]}"
do
    python3 main.py $arg1 $arg2 $arg3
done


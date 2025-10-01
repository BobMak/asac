#!/bin/bash
# this script moves the rlpyt logs in the experiment folder to match
# the formate expected by the "comparison_plotter.py"
# rlpyt format: ft_logs/{experiment}/{environmnet}/{algo_name+env}/{run_id}/*.tb
# comparison_plotter.py format: ft_logs/{experiment}/{environmnet}/{algo_name+env+run_id}/*.tb
EXP_NAME=${1:-"paper"}
# move the logs
for d in ft_logs/$EXP_NAME/*/*/*; do
    # skip if d points to a file instead of a directory
    if [ ! -d $d ]; then
        continue
    fi
    # get the run id
    run_id=$(echo $d | rev | cut -d'_' -f1)
    # get the algo name
    algo_name=$(echo $d | cut -d'/' -f4 | cut -d'_' -f1)
    # get the env name
    env_name=$(echo $d | cut -d'/' -f3)
    # create the new directory
    new_dir="ft_logs/$EXP_NAME/$env_name/${algo_name}_${env_name}_${run_id}"
    echo $new_dir
    mkdir -p $new_dir
    # copy the files
    sudo cp $d/* $new_dir
    # remove the old directory
    # rm -r $d
    sudo chmod 777 $new_dir/*
done

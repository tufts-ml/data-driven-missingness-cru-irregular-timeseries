#!/bin/bash
#
# Usage
# -----
# $ bash launch_cru_extrapolation_eicu.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi


for random_seed in 150 324 901 321 5 90 101 563 912 4 9823 109 339 1029 32
do
    export random_seed=$random_seed
for lr in 0.001
do
    export lr=$lr
for lsd in 10
do
    export lsd=$lsd
for mnar in 'False'
do
    export mnar=$mnar
    ## Use this line to see where you are in the loop
    echo "random_seed=$random_seed-lr=$lr=mnar=$mnar-lsd=$lsd"

    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < do_cru_extrapolation_eicu.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash do_cru_extrapolation_eicu.slurm
    fi

done
done
done
done


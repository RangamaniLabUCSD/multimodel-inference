#!/bin/bash

# Define the CSV file path
CSV_FILE="./shin_2014_loc_diff_models_params.csv"

# number of models to process
n_models=$(wc -l < ./shin_2014_loc_diff_models_params.csv)

# Initialize counter
counter=0

# Read the CSV file line by line, skipping the header
tail -n +1 "$CSV_FILE" | while IFS='.' read -r loc_diff_case diff_params _; do
    ((counter++))  # Increment counter
    savedir="../../../results/MAPK/param_est/Keyes_2020_data_locDiffs/shin_2014/$loc_diff_case"
    diff_params=$(echo "$diff_params" | tr -d '"') # remove leading and trailing quotes

    echo "Processing $loc_diff_case with diff_params: $diff_params"
    echo "($counter of $n_models)"
    

    python inference_process_location_diff_traj.py -model shin_2014_Rap1 -free_params kc47,kc43,kd39,kc45,ERK_tot,ki39,kc41,kRap1_RafAct -diff_params "$diff_params" -Rap1_state Rap1 -ERK_states pp_ERK -savedir "$savedir" -input_state EGF -EGF_conversion_factor 0.001 -ERK_states pp_ERK -prior_family "[['LogNormal(mu=-0.9403273097023963)',['sigma']],['LogNormal(mu=1.539659017826184)',['sigma']],['LogNormal(mu=-1.822631132895142)',['sigma']],['LogNormal(mu=-2.677715001306137)',['sigma']],['LogNormal(mu=-1.487220279709851)',['sigma']],['LogNormal(mu=-7.1670442769327005)',['sigma']],['LogNormal(mu=3.9265174515785985)',['sigma']],['LogNormal(mu=0.0,sigma=2.79)',[]]]" -time_conversion_factor 60 -nsamples 500 -ncores 4 -nchains 4 --skip_prior_sample
done
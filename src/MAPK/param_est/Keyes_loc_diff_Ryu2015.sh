#!/bin/bash

# Define the CSV file path
CSV_FILE="./ryu_2015_loc_diff_models_params.csv"

# number of models to process
n_models=$(wc -l < ./ryu_2015_loc_diff_models_params.csv)

# Initialize counter
counter=0

# Read the CSV file line by line, skipping the header
tail -n +1 "$CSV_FILE" | while IFS='.' read -r loc_diff_case diff_params _; do
    ((counter++))  # Increment counter
    savedir="../../../results/MAPK/param_est/Keyes_2020_data_locDiffs/ryu_2015/$loc_diff_case"
    diff_params=$(echo "$diff_params" | tr -d '"') # remove leading and trailing quotes

    echo "Processing $loc_diff_case with diff_params: $diff_params"
    echo "($counter of $n_models)"
    
    python inference_process_location_diff_traj.py -model ryu_2015_Rap1 -free_params D2,T_dusp,K_dusp,K2,dusp_ind,k_RafRap1,D_RafRap1 -Rap1_state Rap1 -diff_params "$diff_params" -savedir "$savedir" -input_state EGF -EGF_conversion_factor 6.048 -ERK_states ERK_star -prior_family "[['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=4.499809670330265)',['sigma']],['LogNormal(mu=-2.3025850929940455)',['sigma']],['LogNormal(mu=0.0)',['sigma']],['LogNormal(mu=1.791759469228055)',['sigma']],['LogNormal(sigma=3.5244297004803578, mu=1.6094379124341003)',[]],['LogNormal(sigma=3.5244297004803578, mu=1.6094379124341003)',[]]]" -time_conversion_factor 60 -nsamples 500 -ncores 4 -nchains 4 --skip_prior_sample
done
#!/bin/bash

# Define the CSV file path
CSV_FILE="./orton_2009_loc_diff_models_params.csv"

# number of models to process
n_models=$(wc -l < ./orton_2009_loc_diff_models_params.csv)

# Initialize counter
counter=4

# Read the CSV file line by line, skipping the header
tail -n +5 "$CSV_FILE" | while IFS='.' read -r loc_diff_case diff_params _; do
    ((counter++))  # Increment counter
    savedir="../../../results/MAPK/param_est/Keyes_2020_data_locDiffs/orton_2009/$loc_diff_case"
    diff_params=$(echo "$diff_params" | tr -d '"') # remove leading and trailing quotes

    echo "Processing $loc_diff_case with diff_params: $diff_params"
    echo "($counter of $n_models)"
    
    python inference_process_location_diff_traj.py -model orton_2009 -free_params km_Erk_Activation,k1_C3G_Deactivation,km_Erk_Deactivation,k1_P90Rsk_Deactivation,k1_Sos_Deactivation -diff_params "$diff_params" -Rap1_state Rap1Inactive -savedir "$savedir" -input_state EGF -EGF_conversion_factor 602214 -ERK_states ErkActive -prior_family "[['LogNormal(mu=13.822823751258499)',['sigma']],['LogNormal(mu=0.9162907318741551)',['sigma']],['LogNormal(mu=15.067270166119108)',['sigma']],['LogNormal(mu=-5.298317366548036)',['sigma']],['LogNormal(mu=0.9162907318741551)',['sigma']]]" -time_conversion_factor 60 -nsamples 500 -ncores 4 -nchains 4 --skip_prior_sample
done
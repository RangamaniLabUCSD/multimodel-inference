import arviz as az
import pandas as pd
import json
import os

import numpy as np
import diffrax
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import seaborn as sns
import jax
import sys
from scipy.stats import mode
from tqdm import tqdm
import colorcet as cc

sys.path.append("../models/")
from orton_2009 import *
# Rap1 models
from shin_2014_Rap1 import *
from ryu_2015_Rap1 import *
from vonKriegsheim_2009_Rap1 import *

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, '../')
from utils import *

rng = np.random.default_rng(seed=1234)


################ LOAD in DATA ################
savedir = '../../../results/MAPK/param_est/Keyes_2020_data_locDiffs/'

# load in the model info 
model_info = json.load(open('model_info.json', 'r'))
model_names = {'orton_2009':['all_diff', 'Rap1_diff', 'p90RsKSoS_diff', 'p90RsKSoS_Rap1_diff'],
                    'shin_2014':['all_diff', 'Rap1_diff', 'Sos_diff', 'Sos_Rap1_diff'],
                    'ryu_2015':['all_diff', 'Rap1_diff', 'DUSP_diff', 'DUSP_Rap1_diff'],}

# load in the training data
data_file = '../../../results/MAPK/Keyes_et_al_2020-fig1-data1-v2-'

inputs_CYTO, data_CYTO, data_std_CYTO, times_CYTO \
    = load_data_json(data_file+'CYTO_CYTOmax.json', data_std=True, time=True)
inputs_PM, data_PM, data_std_PM, times_PM \
    = load_data_json(data_file+'PM_PMmax.json', data_std=True, time=True)

data_file = '../../../results/MAPK/Keyes_et_al_2020-fig3-data1-v2-'
inputs_CYTO_RAP1i, data_CYTO_RAP1i, data_std_CYTO_RAP1i, \
    times_CYTO_RAP1i = load_data_json(data_file+'CYTO_RAP1inhib_CYTOmax.json', \
    data_std=True, time=True)
inputs_PM_RAP1i, data_PM_RAP1i, data_std_PM_RAP1i, times_PM_RAP1i \
    = load_data_json(data_file+'PM_RAP1inhib_PMmax.json', \
    data_std=True, time=True)

data_time_to_mins = 60

# data in each compartment are sampled at slightly different times, so we need to interpolate to align them
# use diffrax linear interpolation to get the MAPK activity at specific time point over 40mins
data_CYTO_interp = diffrax.LinearInterpolation(times_CYTO, data_CYTO)
data_std_CYTO_interp = diffrax.LinearInterpolation(times_CYTO, data_std_CYTO)
data_PM_interp = diffrax.LinearInterpolation(times_PM, data_PM)
data_std_PM_interp = diffrax.LinearInterpolation(times_PM, data_std_PM)

data_CYTO_RAP1i_interp = diffrax.LinearInterpolation(times_CYTO_RAP1i, \
    data_CYTO_RAP1i)
data_std_CYTO_RAP1i_interp = diffrax.LinearInterpolation(times_CYTO_RAP1i,\
    data_std_CYTO_RAP1i)
data_PM_RAP1i_interp = diffrax.LinearInterpolation(times_PM_RAP1i, data_PM_RAP1i)
data_std_PM_RAP1i_interp = diffrax.LinearInterpolation(times_PM_RAP1i, \
    data_std_PM_RAP1i)

min_time = np.round(np.min([times_CYTO[-1], times_PM[-1], times_CYTO_RAP1i[-1], times_PM_RAP1i[-1]]))
n_times = np.max([len(times_CYTO), len(times_PM), len(times_CYTO_RAP1i), len(times_PM_RAP1i)])
times = np.linspace(0, min_time, n_times)

# get data at standard times
data = {
    'CYTO':data_CYTO_interp.evaluate(times),
    'PM':data_PM_interp.evaluate(times),
    'CYTO_Rap1KD':data_CYTO_RAP1i_interp.evaluate(times),
    'PM_Rap1KD':data_PM_RAP1i_interp.evaluate(times)}

data_std = {
    'CYTO':data_std_CYTO_interp.evaluate(times),
    'PM':data_std_PM_interp.evaluate(times),
    'CYTO_Rap1KD':data_std_CYTO_RAP1i_interp.evaluate(times),
    'PM_Rap1KD':data_std_PM_RAP1i_interp.evaluate(times)}

SAM40_post_pred = {model:{submodel:[] for submodel in model_names[model]} for model in model_names.keys()}
RMSE = {model:{submodel:{} for submodel in model_names[model]} for model in model_names.keys()}
rel_error = {model:{submodel:{} for submodel in model_names[model]} for model in model_names.keys()}
uncertainty95 = {model:{submodel:{} for submodel in model_names[model]} for model in model_names.keys()}

# set up a color palette
# OLD
# colors = ["#40004b","#762a83","#9970ab","#c2a5cf","#e7d4e8","#f7f7f7","#d9f0d3","#a6dba0","#5aae61","#1b7837"]
# nodes = np.linspace(0, 1, len(colors))
# mymap = LinearSegmentedColormap.from_list('mycolors', list(zip(nodes, colors)))
# mpl.colormaps.register(cmap=mymap)
# colors = mymap(np.linspace(0, 1, 12))

# get standard color palette
colors = get_color_pallette()

color_idx = 0

for idx, model in enumerate(model_names.keys()):
    for submodel in model_names[model]:
        name = savedir + model + '/' + submodel + '/'
        if model in ['shin_2014', 'ryu_2015']:
            model_ = model + '_Rap1'
        else:
            model_ = model
        
        # load in the posterior predictive samples
        llike_samples = {
        'CYTO': np.load(name + submodel + '_' + model_ + '_CYTO_posterior_predictive_samples.npy'),
        'PM': np.load(name + submodel + '_' + model_ + '_PM_posterior_predictive_samples.npy'),
        'CYTO_Rap1KD': np.load(name + submodel + '_' + model_ + \
                                     '_CYTO_Rap1KD_posterior_predictive_samples.npy'),
        'PM_Rap1KD': np.load(name + submodel + '_' + model_ + \
                                     '_PM_Rap1KD_posterior_predictive_samples.npy')}
        
        # temp dicts to hold stuff
        sam40s = {sample:[] for sample in ['CYTO', 'PM', 'CYTO_Rap1KD', 'PM_Rap1KD']}
        rmses = {sample:[] for sample in ['CYTO', 'PM', 'CYTO_Rap1KD', 'PM_Rap1KD']}
        errors = {sample:[] for sample in ['CYTO', 'PM', 'CYTO_Rap1KD', 'PM_Rap1KD']}
        uncertainty = {sample:[] for sample in ['CYTO', 'PM', 'CYTO_Rap1KD', 'PM_Rap1KD']}

        for comp in ['CYTO', 'PM']:
            # plot the posterior predictive trajectories
            samples_ = np.stack([llike_samples[comp], llike_samples[comp+'_Rap1KD']])
            # add additional zeros to each sample for the ICs (this is just to make plotting easier, we know the ICs are zero)
            zer_col = np.zeros((samples_.shape[0], samples_.shape[1], 1))
            samples_ = np.concatenate([zer_col, samples_], axis=2)
            # reshape the samples_ martix so that it is n_traj x n_comp x n_times
            samples_ = np.swapaxes(samples_, 0, 1)

            data_ = np.stack([data[comp], data[comp+'_Rap1KD']])
            data_std_ = np.stack([data_std[comp], data_std[comp+'_Rap1KD']])


            plot_posterior_trajectories(samples_, data_, data_std_, times,
                colors[color_idx], ['', '_Rap1KD'], 
                name + submodel + '_' + model_ + '_' + comp + '_posterior_predictive', '',
                fname='',
                data_time_to_mins=60,
                width=1., height=0.5, 
                data_downsample=10,
                ylim=[[0.0, 1.5],[0.0, 1.5]],
                y_ticks=[[0.0, 1.0],[0.0, 1.0]],
                labels=False,)
            
            for rap1 in ['','_Rap1KD']: 
                # compute SAM40 post-pred predictions
                idx_40_min = np.argmin(np.abs(times - 40*60))
                sam40s[comp+rap1] = list(np.apply_along_axis(sustained_activity_metric, 1, 
                                                llike_samples[comp+rap1], 
                                                idx_40_min))
                
                # compute RMSE and relative error and uncertainty
                rmse = np.sqrt(np.nanmean((np.nanmean(llike_samples[comp+rap1],axis=0) - \
                    data[comp+rap1][1:])**2))
                rel_er = np.linalg.norm(np.nanmean(llike_samples[comp+rap1],axis=0) - \
                    data[comp+rap1][1:])/np.linalg.norm(data[comp+rap1][1:])
                cred95 = np.nanmean(np.squeeze(np.diff(np.nanquantile(llike_samples[comp+rap1], \
                    [0.025, 0.975], axis=0),axis=0)))
                
                rmses[comp+rap1] = float(rmse)
                errors[comp+rap1] = float(rel_er)
                uncertainty[comp+rap1] = float(cred95)

        # store SAM40 predictions and errors
        SAM40_post_pred[model][submodel] = sam40s
        RMSE[model][submodel] = rmses
        rel_error[model][submodel] = errors
        uncertainty95[model][submodel] = uncertainty
        color_idx += 1

# save SAM40 predictions
with open(savedir + 'SAM40_post_pred.json', 'w') as f:
    json.dump(SAM40_post_pred, f)

# save errors
with open(savedir + 'rel_errors.json', 'w') as f:
    json.dump(rel_error, f)

with open(savedir + 'RMSE.json', 'w') as f:
    json.dump(RMSE, f)

# save uncertainty
with open(savedir + 'uncertainty.json', 'w') as f:
    json.dump(uncertainty95, f)


######## Make additional posterior predictive plots of the (-)feedback KD ########
models_to_plot = ['orton_2009', 'shin_2014']
submodels = ['p90RsKSoS_Rap1_diff', 'Sos_Rap1_diff']
color_idxs = [3,7]

for color_idx, model, submodel in zip(color_idxs, models_to_plot, submodels):
    name = savedir + model + '/' + submodel + '/'

    if model=='shin_2014':
            model_ = model + '_Rap1'
    else:
        model_ = model

    llike_samples = {
        'CYTO': np.load(name + submodel + '_' + model_ + '_CYTO_posterior_negFeedbackKD_predictive_samples.npy'),
        'PM': np.load(name + submodel + '_' + model_ + '_PM_posterior_negFeedbackKD_predictive_samples.npy'),
        'CYTO_Rap1KD': np.load(name + submodel + '_' + model_ + \
                                     '_CYTO_Rap1KD_posterior_negFeedbackKD_predictive_samples.npy'),
        'PM_Rap1KD': np.load(name + submodel + '_' + model_ + \
                                     '_PM_Rap1KD_posterior_negFeedbackKD_predictive_samples.npy')}
    
    for comp in ['CYTO', 'PM']:
            # plot the posterior predictive trajectories
            samples_ = np.stack([llike_samples[comp], llike_samples[comp+'_Rap1KD']])
            # add additional zeros to each sample for the ICs (this is just to make plotting easier, we know the ICs are zero)
            zer_col = np.zeros((samples_.shape[0], samples_.shape[1], 1))
            samples_ = np.concatenate([zer_col, samples_], axis=2)
            # reshape the samples_ martix so that it is n_traj x n_comp x n_times
            samples_ = np.swapaxes(samples_, 0, 1)

            # set data to nan so that it doesn't plot
            data_ = np.nan*np.ones_like(np.stack([data[comp], data[comp+'_Rap1KD']]))
            data_std_ = np.nan*np.ones_like(np.stack([data_std[comp], data_std[comp+'_Rap1KD']]))

            plot_posterior_trajectories(samples_, data_, data_std_, times,
                colors[color_idx], ['', '_Rap1KD'], 
                name + submodel + '_' + model_ + '_' + comp + '_posterior_negFeedbackKD_predictive', '',
                fname='',
                data_time_to_mins=60,
                width=1., height=0.5, 
                data_downsample=10,
                ylim=[[0.0, 1.5],[0.0, 1.5]],
                y_ticks=[[0.0, 1.0],[0.0, 1.0]],
                labels=False,)
    

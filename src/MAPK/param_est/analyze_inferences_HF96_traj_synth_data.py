import arviz as az
import pandas as pd
import json
import os

import numpy as np
import diffrax
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import sys
from scipy.stats import mode
from tqdm import tqdm

from timeout import timeout

sys.path.append("../models/")
from huang_ferrell_1996 import *
from bhalla_iyengar_1999 import *
from kholodenko_2000 import *
from levchenko_2000 import *
from brightman_fell_2000 import *
from schoeberl_2002 import *
from hatakeyama_2003 import *
from hornberg_2005 import *
from birtwistle_2007 import *
from orton_2009 import *
from vonKriegsheim_2009 import *
from shin_2014 import *
from ryu_2015 import *
from kochanczyk_2017 import *
from dessauges_2022 import *

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, '../')
from utils import *

rng = np.random.default_rng(seed=1234)


################ LOAD in DATA ################
savedir = '../../../results/MAPK/param_est/HF96_traj_synthetic_data/'


# load in the model info 
model_info = json.load(open('model_info.json', 'r'))
model_names = list(model_info.keys())
display_names = [model_info[model]['display_name'] for model in model_names]

idata = {}
posterior_samples = {}
sample_times = {}

for model in model_names:
    idata[model], _, sample_times[model] = load_smc_samples_to_idata(savedir+model+'/'+model+'_smc_samples.json', sample_time=True)
    posterior_samples[model] = np.load(savedir+model+'/'+model+'_posterior_predictive_samples.npy')

# shin has 4000 samples so downsample to 2000
idxs = rng.choice(np.arange(4000), size=2000, replace=False)
posterior_samples['shin_2014'] = posterior_samples['shin_2014'][idxs,:,:]

# load in the training data
inputs, data, data_std, times = load_data_json('../../../results/MAPK/HF_96_traj_data.json', data_std=True, time=True)
data_time_to_mins = 60

inputs_dose_response, data_dose_response = load_data('../../../results/MAPK/HF_96_synthetic_data.csv')

# set up a color palette
# this is the ColorBrewer purple-green with 11 colors + three greys https://colorbrewer2.org/#type=diverging&scheme=PRGn&n=11
colors = ['#40004b','#762a83','#9970ab','#c2a5cf','#e7d4e8','#f7f7f7','#d9f0d3','#a6dba0','#5aae61','#1b7837','#00441b','#363737','#929591','#d8dcd6']
# this one gets to 10 colors by removing the darkest green
colors = ['#40004b','#762a83','#9970ab','#c2a5cf','#e7d4e8','#f7f7f7','#d9f0d3','#a6dba0','#5aae61','#1b7837','#363737','#929591','#d8dcd6']
orange = '#de8f05'


################ Write sampling times to a file ################
with open(savedir + 'SMC_runtimes.txt', 'w') as f:
    for model in model_names:
        f.write(f'{model}: {sample_times[model]/3600} hr\n')

################ Make pretty posterior predictive trajectories ################
skip_idxs = []
for idx, model in enumerate(model_names):
    if idx in skip_idxs:
        print('skipping', model)
        continue
    else:
        print('plotting', model)
        plot_posterior_trajectories(posterior_samples[model], data, data_std, times, colors[idx], 
                                        inputs, savedir+model+'/', model, data_time_to_mins=60,
                                        width=1.1, height=0.5, data_downsample=10,
                                        ylim=[[0.0, 1.2], [0.0, 1.2], [0.0, 1.2]],
                                        y_ticks=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        
################ Make pretty posterior trajectories ################
## We also need to plot and analyze trajectories that are not posterior predictive, 
#           but simply use posterior samples to compute them

# parameters for plotting that control the x and y labels
plotting_params = {
    'kholodenko_2000':[False,False,True,True],'levchenko_2000':[False,False,False,False],'hatakeyama_2003':[False,False,False,False],'hornberg_2005':[False,False,False,False],'birtwistle_2007':[False,False,False,False],'orton_2009':[True,True,True,True],'vonKriegsheim_2009':[True,True,False,False],'shin_2014':[True,True,False,False],'ryu_2015':[True,True,False,False],'kochanczyk_2017':[True,True,False,False]
}

n_traj = 400

skip_idxs = []
for idx,model in enumerate(model_names):
    if idx in skip_idxs:
        print('skipping', model)
        continue
    else:
        print('plotting', model)
        this_model_info = model_info[model]

        plot_p = plotting_params[model]

     
        # predict trajectories
        traj = predict_traj_response(model, idata[model], inputs, times, 
                                              this_model_info['input_state'], this_model_info['ERK_states'],
                                              float(this_model_info['time_conversion']),
                                              EGF_conversion_factor=float(this_model_info['EGF_conversion_factor']),
                                              nsamples=400)
        # save
        np.save(savedir+model+'/traj_predict.npy', traj)

        # plot
        plot_posterior_trajectories(traj, data, data_std, times, colors[idx], 
                                        inputs, savedir+''+model+'/',
                                        model, data_time_to_mins=60,
                                        width=1.1, height=0.5, 
                                        data_downsample=10,
                                        ylim=[[0.0, 1.2], [0.0, 1.2], [0.0, 1.2]],
                                        y_ticks=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                                        fname='_pred_traj_')
plt.close('all')

################ Make posterior dose-response curves ################
## Now we want to use posterior draws to simulate dose-response curve predictions

skip_idxs = []
for idx,model in enumerate(model_names):
    if idx in skip_idxs:
        print('skipping', model)
        continue
    else:
        print('plotting', model)
        this_model_info = model_info[model]

        plot_p = plotting_params[model]

        # create dose-response curve prediction
        dose_response = predict_dose_response(model, idata[model], inputs_dose_response,   
                                this_model_info['input_state'], this_model_info['ERK_states'], 
                                float(this_model_info['max_time']), EGF_conversion_factor=float(this_model_info['EGF_conversion_factor']), 
                                nsamples=400, timeout=30)
        print(dose_response.shape)
        # save
        np.save(savedir+model+'/dose_response_predict.npy', dose_response)

        fig, ax = plot_stimulus_response_curve(dose_response, data_dose_response, inputs_dose_response, input_name='EGF stimulus (nM)', output_name='% maximal ERK \n activity', box_color='w', data_color='r',
                                        data_std=0.1, width=1.1, height=1.1, data_marker_size=5.0, scatter_marker_size=0,
                                        title=this_model_info['display_name'], xlabel=plot_p[0],xticklabels=plot_p[1],ylabel=plot_p[2], yticklabels=plot_p[3])
        ax.set_title(ax.get_title(), fontsize=12.0)
        fig.savefig(savedir+model+'/dose_response_predict.pdf', transparent=True)

plt.close('all')
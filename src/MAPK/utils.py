# Nathaniel Linden 2023
# Utilities to help with the project
import json
import os
import sys

import numpy as np
import jax.numpy as jnp
import pandas as pd
import jax

import pymc as pm
from pymc.sampling.jax import sample_numpyro_nuts
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify
import arviz as az
import preliz as pz
import diffrax
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut

import matplotlib.pyplot as plt
import seaborn as sns

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(seed=1234)

# set matplotlib defaults
plt.style.use('custom')

# custom plotting helper funcs
# sys.path.insert(0, './dot_matplotlib/')
from plotting_helper_funcs import *

# load DIFFRAX PYTENSOR OP CODE
sys.path.insert(0, './param_est/')
from diffrax_ODE_PyTensor import *

# import models
sys.path.insert(0, '../models/')
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

###############################################################################
#### General Utils ####
###############################################################################
def construct_full_param_list(param_list, param_idxs, nominal_param_array):
    """ Constructs a full parameter list from a list of parameter indices and a dictionary of all nominal parameters in the model.
    
    We do this using jax.numpy to enable jit.

    All inputs must by arrays or jax arrays.
    """

    return nominal_param_array.at[param_idxs].set(param_list)

def parse_identifiability_results(path_to_ID_txt):

    file = open(path_to_ID_txt, 'r')
    lines = file.readlines()
    file.close()

    # parse the lines
    identifiabile = lines[1].strip(', \n')

    if len(lines) > 3:
        non_identifiable = lines[3].strip(', \n')
    else:
        non_identifiable = ''

    # split the strings
    identifiabile = identifiabile.split(', ')
    non_identifiable = non_identifiable.split(', ')

    # remove states from the lists
    identifiabile = [item for item in identifiabile if 'x' not in item]
    non_identifiable = [item for item in non_identifiable if 'x' not in item]

    return identifiabile, non_identifiable

def load_smc_samples_to_idata(samples_json, sample_time=False):
    """ Load SMC samples from json file to arviz InferenceData object """
    with open(samples_json, 'r') as f:
        data = json.load(f)
    
    # create idata object from dictionary
    # ignore sample stats because that changes with each SMC chain
    idata = az.from_dict(
        posterior =  data['posterior'],
        posterior_attrs = data['posterior_attrs'],
        # sample_stats = data['sample_stats'],
        observed_data = data['observed_data'],
        observed_data_attrs = data['observed_data_attrs'],
        log_likelihood = data['log_likelihood'],
        log_likelihood_attrs = data['log_likelihood_attrs'],
        constant_data = data['constant_data'],
        constant_data_attrs = data['constant_data_attrs'],
        attrs = data['attrs'],
    )

    sample_stats = data['sample_stats']

    if sample_time:
        return idata, sample_stats, data['sample_stats_attrs']['_t_sampling']
    else:
        return idata, sample_stats
 
def construct_y0_EGF_inputs(EGF_vals, y0, EGF_idx):
    n_repeats = len(EGF_vals)
    y0_EGF_inputs = np.repeat(y0, n_repeats, axis=0)
    y0_EGF_inputs[:, EGF_idx] = EGF_vals

    return y0_EGF_inputs

def load_data(data_file, data_std=False, time=False):
    """ Loads the data from the specified file.
    """
    # load the data
    data_df = pd.read_csv(data_file)
    inputs = np.array(data_df['stimulus'].to_numpy())
    data = np.vstack((data_df['response'].to_numpy()))

    if data_std and time:
        data_std = np.vstack((data_df['response_std'].to_numpy()))
        time = np.array(data_df['time'].to_numpy())
        return inputs, data, data_std, time
    if time and not data_std:
        time = np.array(data_df['time'].to_numpy())
        return inputs, data, time
    if data_std and not time:
        data_std = np.vstack((data_df['response_std'].to_numpy()))
        return inputs, data, data_std
    else:
        return inputs, data

def load_data_json(data_file, data_std=False, time=False):
    """ Loads the data from the specified json data file.
    """
    # load the data
    with open(data_file, 'r') as file:
        data_dict = json.load(file)
    
    inputs = np.array(data_dict['stimulus'])
    data = np.array(data_dict['response'])

    if data_std and time:
        data_std = np.array(data_dict['response_std'])
        time = np.array(data_dict['time'])
        return inputs, data, data_std, time
    if time and not data_std:
        time = np.array(data_dict['time'])
        return inputs, data, time
    if data_std and not time:
        data_std = np.array(data_dict['response_std'])
        return inputs, data, data_std
    else:
        return inputs, data

def get_param_subsample(idata, n_traj, p_dict,rng=np.random.default_rng(seed=1234)):
    dat = idata.posterior.to_dict() # convert to dictionary
    free_params = list(dat['data_vars'].keys()) # figure out which params are free
    # get total number of MCMC samples
    n_samples = np.array(dat['data_vars'][free_params[0]]['data']).reshape(-1).shape[0]

    # extract samples for free params ot dict of numpy arrays
    free_param_samples = {}
    for param in free_params:
        free_param_samples[param] = np.array(dat['data_vars'][param]['data']).reshape(-1)

    # randomly select n_traj samples
    param_samples = []
    idxs = rng.choice(np.arange(n_samples), size=n_traj, replace=False)
    for i in idxs:
        tmp = []
        for param in p_dict.keys():
            if param in free_params:
                tmp.append(free_param_samples[param][i])
            else:
                tmp.append(p_dict[param])
        param_samples.append(tmp)
 
    return np.array(param_samples)

###############################################################################
#### Solving ODEs ####
###############################################################################
@jax.jit
def solve_ss(model_dfrx_ode, y0, params, t1):
    """ simulates a model over the specified time interval and returns the 
    calculated steady-state values.
    Returns an array of shape (n_species, 1) """
    dt0=1e-3
    event_rtol=1e-6
    event_atol=1e-5
    solver = diffrax.Kvaerno5()
    event=diffrax.SteadyStateEvent(event_rtol, event_atol)
    stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6)
    t0 = 0.0

    sol = diffrax.diffeqsolve(
        model_dfrx_ode, 
        solver, 
        t0, t1, dt0, 
        tuple(y0), 
        stepsize_controller=stepsize_controller,
        discrete_terminating_event=event,
        args=params,
        max_steps=600000,
        throw=False,)
    
    return jnp.array(sol.ys)

# vmap steady state solving over different initial conds
#   this means vmapping over the y0 and assuming everything else is fixed
vsolve_ss = jax.vmap(solve_ss, in_axes=(None, 0, None, None))

@jax.jit
def solve_traj(model_dfrx_ode, y0, params, t1, ERK_indices, times):
    """ simulates a model over the specified time interval and returns the 
    calculated values.
    Returns an array of shape (n_species, 1) """
    dt0=1e-3
    solver = diffrax.Kvaerno5()
    stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6)
    t0 = 0.0
    saveat=diffrax.SaveAt(ts=times)

    sol = diffrax.diffeqsolve(
        model_dfrx_ode, 
        solver, 
        t0, 
        t1, 
        dt0, 
        tuple(y0), 
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        args=params,
        max_steps=600000,
        throw=False,)
    
    return jnp.sum(jnp.array(sol.ys)[ERK_indices, :], axis=0)

# vmap traj solving over different initial conds
#   this means vmapping over the y0 and assuming everything else is fixed
vsolve_traj = jax.vmap(solve_traj, in_axes=(None, 0, None, None, None, None))

# vmap traj solving over the parameters
vsolve_params_traj = jax.vmap(solve_traj, in_axes=(None, None, 0, None, None, None))
    
def ERK_stim_response(params, model_dfrx_ode, max_time, y0_EGF_inputs, 
                      output_states, normalization_func=None):
    """ function to compute the ERK response to EGF stimulation
        Args:
            difrx_model (diffrax.Model): diffrax model object
            EGF_inputs (np.ndarray): array of EGF inputs to simulate
            output_states (list): list of output states to sum over
            maxtime (int): max time to simulate the model
        Returns:
            normalized_ERK_response (np.ndarray): array of ERK responses to each EGF input
    """
    # vmap solve over all initial conditions
    ss = vsolve_ss(model_dfrx_ode, y0_EGF_inputs, params, max_time)
    ss = jnp.squeeze(ss)

    # sum over the output states
    erk_acts = jnp.sum(ss[:, output_states], axis=1)
    
    if normalization_func is None:
        return erk_acts/jnp.max(erk_acts), erk_acts
    else:
        return erk_acts/normalization_func(params, y0_EGF_inputs[0]), erk_acts

def ERK_stim_trajectory_set(params, model_dfrx_ode, max_time, y0_EGF_inputs, output_states, times, max_input_index=-1):
    """ function to compute the ERK response to EGF stimulation
        Args:
            difrx_model (diffrax.Model): diffrax model object
            EGF_inputs (np.ndarray): array of EGF inputs to simulate
            output_states (list): list of output states to sum over
            maxtime (int): max time to simulate the model
        Returns:
            normalized_ERK_response (np.ndarray): array of ERK trajectories to each EGF input
    """
    # vmap solve over all initial conditions
    traj = vsolve_traj(model_dfrx_ode, y0_EGF_inputs, params, max_time, output_states, times)
    traj = jnp.squeeze(traj)

    # normalize to the max val
    # return traj/jnp.max(jnp.max(traj)), traj

    # normalize to highest value of the biggest stimulus
    # TODO: why?
    return traj/traj[max_input_index,-1], traj

def predict_dose_response(model, posterior_idata, inputs, input_state, 
                          ERK_states, max_time, EGF_conversion_factor=1, nsamples=None, timeout=10):
    """ function to predict dose response curves for a given model and many posterior samples"""
    # try calling the model
    try:
        model = eval(model + '(transient=False)')
    except:
        print('Warning Model {} not found. Skipping this.'.format(model))

    # get parameter names and initial conditions
    p_dict, _ = model.get_nominal_params()
    y0_dict, y0 = model.get_initial_conditions()

    # convert EGF to required units
    inputs_native_units = inputs * EGF_conversion_factor

    # get the EGF index and ERK indices
    state_names = list(y0_dict.keys())
    EGF_idx = state_names.index(input_state)
    ERK_indices = [state_names.index(s) for s in ERK_states.split(',')]

    # make initial conditions that reflect the inputs
    y0_EGF_ins = construct_y0_EGF_inputs(inputs_native_units, np.array([y0]), EGF_idx)

    # solve the model nsamples times
    if nsamples is None:
        nsamples = posterior_idata.posterior.dims['draw']*posterior_idata.posterior.dims['chain']
    elif nsamples > posterior_idata.posterior.dims['draw']*posterior_idata.posterior.dims['chain']:
        print('Warning: nsamples > posterior samples. Using all posterior samples.')
        nsamples = posterior_idata.posterior.dims['draw']*posterior_idata.posterior.dims['chain']

    param_samples = get_param_subsample(posterior_idata, nsamples, p_dict)

    def dr_func(param):
        return ERK_stim_response(param, diffrax.ODETerm(model), max_time, y0_EGF_ins, ERK_indices)[0]

    dose_response = []
    skipped = False
    for param in tqdm(param_samples):
        # print(param)
        try: # try to run the function
            dr = func_timeout(timeout, dr_func, args=(param,))
            dose_response.append(dr)
            skipped = False
        except FunctionTimedOut: # the function timed out
            print('Function timed out. Skipping this sample. The params are {}'.format(param))
            dr = np.nan*np.ones((len(inputs)))
            dose_response.append(dr)
                                      
            if skipped: # if we skipped the last sample, increase the timeout time
                timeout = timeout*1.5
                print('Warning: two samples in a row have timed out. Increasing timeout time by 50 percent. Max waitime per iteration is now {} seconds.'.format(timeout))
            else: # if we did not skip the last sample, try again with the same timeout time
                skipped = True
        else:
            continue

    return np.array(dose_response)

def predict_traj_response(model, posterior_idata, inputs, times, input_state, 
                          ERK_states, time_conversion_factor=1, 
                          EGF_conversion_factor=1, nsamples=None,
                          max_input_index=-1):
    """ function to predict trajectories for a given model and many posterior samples"""
    # load model
    try:
        model = eval(model + '(transient=False)')
    except:
        print('Warning Model {} not found. Skipping this.'.format(model))


    # get parameter names and initial conditions
    p_dict, _ = model.get_nominal_params()
    y0_dict, y0 = model.get_initial_conditions()

    # convert EGF to required units
    inputs_native_units = inputs * EGF_conversion_factor

    # get the EGF index and ERK indices
    state_names = list(y0_dict.keys())
    EGF_idx = state_names.index(input_state)
    ERK_indices = [state_names.index(s) for s in ERK_states.split(',')]

    # make initial conditions that reflect the inputs
    y0_EGF_ins = construct_y0_EGF_inputs(inputs_native_units, np.array([y0]), EGF_idx)

    # solve the model nsamples times
    if nsamples is None:
        nsamples = posterior_idata.posterior.dims['draw']*posterior_idata.posterior.dims['chain']
    elif nsamples > posterior_idata.posterior.dims['draw']*posterior_idata.posterior.dims['chain']:
        print('Warning: nsamples > posterior samples. Using all posterior samples.')
        nsamples = posterior_idata.posterior.dims['draw']*posterior_idata.posterior.dims['chain']

    param_samples = get_param_subsample(posterior_idata, nsamples, p_dict)
    trajectories = []
    for param in tqdm(param_samples):
        trajectories.append(ERK_stim_trajectory_set(param, diffrax.ODETerm(model), 
                                                    max(times/time_conversion_factor), 
                                                    y0_EGF_ins, ERK_indices, times/time_conversion_factor, max_input_index)[0])

    return np.array(trajectories)
###############################################################################
#### PyMC Inference Utils ####
###############################################################################
def set_prior_params(model_name, param_names, nominal_params, free_param_idxs, prior_family=[['Gamma()',['alpha', 'beta']]], upper_mult=1.9, lower_mult=0.1, prob_mass_bounds=0.95,          
    saveplot=True, savedir=None):
    """ Sets the prior parameters for the MAPK models.
        Inputs:
            - param_names (list): list of parameter names
            - nominal_params (np.ndarray): array of nominal parameter values
            - free_param_idxs (list): list of indices of the free parameters
            - prior_family (str): prior family to use for the parameters. If a string will use that family for all free parameters, otherwise should be a list of strings of the same length as free_param_idxs. Each string should correspond to a pm.Distribution and pz.Distribution object, e.g., Gamma which is the default familly.
            - upper_mult (float): multiplier for the upper bound of the prior
            - lower_mult (float): multiplier for the lower bound of the prior
        Returns:
            - prior_param_dict (dict): dictionary of prior parameters for the model in syntax to use exec to set them in a pymc model object
    """

    if savedir is None:
        savedir = os.getcwd() + '/'

    # determine if a string or list of strings was passed for the prior family
    prior_family = eval(prior_family)
    if len(prior_family) == 1:
        prior_family_list = prior_family*len(free_param_idxs)
    else:
        prior_family_list = prior_family
    
    # set the prior parameters
    prior_param_dict = {}
    for i, param in enumerate(param_names):
        if i in free_param_idxs: # check if we are dealing with a free parameter
            # get the nominal value
            nominal_val = nominal_params[i]
            if nominal_val == 0:
                upper = 1.0
                lower = 1e-4
            else:
                # get the upper and lower bounds
                upper = nominal_val*upper_mult
                lower = nominal_val*lower_mult
                
                print(param, upper, lower)

            # use preliz.maxent to find the prior parameters for the specified family
            prior_fam = prior_family_list[free_param_idxs.index(i)]
            
            dist_family = eval('pz.' + prior_fam[0])
            fig, ax = get_sized_fig_ax(2.0,2.0)
            ax, results = pz.maxent(dist_family, lower, upper, prob_mass_bounds, plot=saveplot, ax=ax) # for some reason the [0] element is None

            # save the plot
            if saveplot:
                ax.set_xscale('log')
                ax.set_title(param)
                fig.savefig(savedir + model_name + param + '_prior.pdf', bbox_inches='tight', transparent=True)

            # set the prior parameters
            prior_fam_name = prior_fam[0].strip(')').split('(')[0]
            fixed_params = prior_fam[0].strip(')').split('(')[1].split(',')
            
            tmp = 'pm.' + prior_fam_name + '("' + param + '",'
            for i, hyper_param in enumerate(prior_fam[1]):
                tmp += hyper_param + '=' + str(results.x[i]) + ', '
            
            for fixed_param in fixed_params:
                if len(fixed_param) > 0:
                    tmp += (fixed_param + ', ')
            prior_param_dict[param] = tmp + ')'
            print(prior_param_dict[param])
        else:
            # set the prior parameters to the nominal value
            prior_param_dict[param] = 'pm.ConstantData("' + param + '", ' + str(nominal_params[i]) + ')'

    return prior_param_dict

def build_pymc_model(prior_param_dict, data, y0_EGF_inputs, 
                    output_states, max_time, model_dfrx_ode, model=None, 
                    simulator=ERK_stim_response, data_sigma=0.1):
    """ Builds a pymc model object for the MAPK models.

    Constructs priors for the model, and uses the ERK_stim_response function to 
    generate the stimulus response function and likelihood.
    """

    
    # Create jax functions to solve # 
    def sol_op_jax(*params):
        pred, _ = simulator(params, model_dfrx_ode, max_time, y0_EGF_inputs, output_states)
        return jnp.vstack((pred))

    # get the jitted versions
    sol_op_jax_jitted = jax.jit(sol_op_jax)

    
    # Create pytensor Op and register with jax # 
    class StimRespOp(Op):
        def make_node(self, *inputs):
            # Convert our inputs to symbolic variables
            inputs = [pt.as_tensor_variable(inp) for inp in inputs]
            # Assume the output to always be a float64 matrix
            outputs = [pt.matrix()]
            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            result = sol_op_jax_jitted(*inputs)
            outputs[0][0] = np.asarray(result, dtype="float64")
        
        def grad(self, inputs, output_grads):
            raise NotImplementedError("PyTensor gradient of StimRespOp not implemented")


    # construct Ops and register with jax_funcify
    sol_op = StimRespOp()

    @jax_funcify.register(StimRespOp)
    def sol_op_jax_funcify(op, **kwargs):
        return sol_op_jax
    
    
    # Construct the PyMC model # 
    model = pm.Model()
    with model:
        # loop over free params and construct the priors
        priors = []
        for key, value in prior_param_dict.items():
            # create PyMC variables for each parameters in the model
            prior = eval(value)
            priors.append(prior)

        # predict dose response
        prediction = sol_op(*priors)

        # assume a normal model for the data
        # sigma specified by the data_sigma param to this function
        llike = pm.Normal("llike", mu=prediction, sigma=data_sigma, observed=data)

    return model

def smc_pymc(model, mapk_model_name, savedir, nsamples=2000, 
             seed=np.random.default_rng(seed=123), ncores=1, threshold=0.5,chains=None, correlation_threshold=0.01):
    """ Function to run SMC sampling using PyMC and the independent Metropolis-Hastings kernel."""
    with model:
        idata = pm.smc.sample_smc(draws=nsamples, random_seed=seed, chains=chains,
                                  cores=ncores, progressbar=True, threshold=threshold,
                                  idata_kwargs={'log_likelihood': True},correlation_threshold=correlation_threshold,)

    # save the samples
    idata.to_json(savedir + mapk_model_name + '_smc_samples.json')

    return idata

def plot_sampling_trace_diagnoses(posterior_idata, savedir, mapk_model_name, sampling_type='smc'):

    # plot the traces with arviz
    az.plot_trace(posterior_idata, compact=False)
    plt.savefig(savedir + mapk_model_name + '_'+ sampling_type + '_traceplot.pdf',)

    # compute the effective sample size and rhat statistics
    diagnoses = {}
    diagnoses['ess'] = az.ess(posterior_idata).to_dict()
    diagnoses['rhat'] = az.rhat(posterior_idata).to_dict()

    # save the diagnoses
    with open(savedir + mapk_model_name + '_' + sampling_type + '_diagnoses.json', 'w') as f:
        json.dump(diagnoses, f)

###############################################################################
#### Plotting ####
# ###############################################################################
def plot_stimulus_response_curve(samples, data, inputs, box_color='k', data_color='r', input_name='EGF stimulus', 
                                 output_name='% maximal ERK activity',data_std=0.1,
                                 yticklabels=True,xticklabels=True,ylabel=True, xlabel=True, title=None,
                                 width=6.0, height=3.0, scatter_marker_size=50, data_marker_size=7):
    dat = {}
    for i,input in enumerate(inputs):
        dat[input] = samples[:,i]

    data_df = pd.DataFrame(dat)

    fig, ax = get_sized_fig_ax(width, height)
    sns.boxplot(data=data_df, color=box_color, ax=ax, whis=(2.5, 97.5), fill=True, 
                native_scale=True, log_scale=(10, 0), fliersize=0, width=0.65)
    ax.set_xlabel(input_name)
    ax.set_ylabel(output_name)

    errors = data_std*np.squeeze(np.ones_like(data))
    ax.scatter(inputs, data, color=data_color, marker='x', s=scatter_marker_size, zorder=10, label='synthetic data')
    ax.errorbar(inputs, np.squeeze(data), yerr=errors, color=data_color, fmt='x', markersize=data_marker_size, zorder=10)

    # ticks, labels, and limits
    ax.set_ylim([-0.25, 1.25])
    ax.set_yticks([0, 0.5, 1.0])
    if yticklabels:
        ax.set_yticklabels(['0', '50', '100'])
    else:
        ax.set_yticklabels(['', '', ''])

    ax.set_xticks([1e-3, 1e-2, 1e-1])
    if xticklabels:
        ax.set_xticklabels([1e-3, 1e-2, 1e-1])
    else:
        ax.set_xticklabels(['', '', ''])
    
    if xlabel:
        ax.set_xlabel(input_name)
    else:
        ax.set_xlabel('')
    
    if ylabel:
        ax.set_ylabel(output_name)
    else:
        ax.set_ylabel('')
    
    if title is not None:
        ax.set_title(title)

    return fig, ax

def plot_trajectory_responses_oneAxis(samples, data, inputs, times, legend_filename, input_name='EGF', 
    output_name='% maximal ERK activity', data_std=0.1, data_downsample=3, colors=['c', 'g', 'b'], width=3.0, height=3.0):

    fig, ax = get_sized_fig_ax(width, height)

    nsamples, ninputs,_ = samples.shape

    for idx in range(ninputs):
            sample = samples[:,idx,:]
            tr_dict =  {'run':{}, 'timepoint':{}, 'ERK_act':{}}
            names = ['run'+str(i) for i in range(nsamples)]
            idxs = np.linspace(0, (nsamples*times.shape[0]-1), nsamples*times.shape[0])
            cnt = 0
            for i in range(nsamples):
                    for j in range(times.shape[0]):
                            tr_dict['run'][int(idxs[cnt])] = names[i]
                            tr_dict['timepoint'][int(idxs[cnt])] = times[j]
                            tr_dict['ERK_act'][int(idxs[cnt])] = sample[i,j]
                            cnt += 1
            tr_df = pd.DataFrame.from_dict(tr_dict)

            sns.lineplot(data=tr_df,
                    x='timepoint',
                    y='ERK_act',
                    color=colors[idx],
                    legend=True,
                    label='[' + input_name + '] = {:.3f} (nM)'.format(inputs[idx]),
                    errorbar=('pi', 95), # percentile interval form 2.5th to 97.5th
                    ax=ax)

    # ax.set_ylim([0.0, 1.0])
    ax.set_xlim([0.0, max(times)])

    # plot data
    for idx,dat in enumerate(data):
            ax.errorbar(times[::data_downsample], np.squeeze(dat)[::data_downsample], yerr=data_std[idx][::data_downsample], fmt='o', linewidth=1.0, markersize=0.1, color='k')

    ax.set_xlabel('time (min)')
    ax.set_ylabel(output_name)

    leg = ax.legend(loc='right', bbox_to_anchor=(2.5,0.5))
    export_legend(leg, filename=legend_filename)
    leg.remove()

    return fig, ax

def plot_posterior_trajectories(post_preds, data, data_std, times, color, 
                                       EGF_levels, savedir, model_name, data_time_to_mins=60, 
                                       y_ticks=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                                       ylim=[[0.0, 1.2], [0.0, 1.2], [0.0, 1.2]],
                                       xticklabels=None, data_downsample=5,
                                       width=1.1, height=0.5,fname='_post_pred_',
                                       labels=True):
        # get dims
        n_traj, n_stim, n_times = post_preds.shape

        # loop over the stimuli and make a plot for each
        for stim_idx in range(n_stim):
                # mean + 95% credible plot
                tr_dict =  {'run':{}, 'timepoint':{}, 'ERK_act':{}}
                names = ['run'+str(i) for i in range(n_traj)]
                idxs = np.linspace(0, (n_traj*times.shape[0]-1), n_traj*times.shape[0])
                cnt = 0
                # TODO: add option for spaghetti plot
                for i in range(n_traj):
                        for j in range(times.shape[0]):
                                tr_dict['run'][int(idxs[cnt])] = names[i]
                                tr_dict['timepoint'][int(idxs[cnt])] = times[j]/data_time_to_mins
                                tr_dict['ERK_act'][int(idxs[cnt])] = post_preds[i,stim_idx,j]
                                cnt += 1
                tr_df = pd.DataFrame.from_dict(tr_dict)

                # make new axes and plot
                fig, ax = get_sized_fig_ax(width, height)
                sns.lineplot(data=tr_df,
                        x='timepoint',
                        y='ERK_act',
                        color=color,
                        legend=False,
                        alpha=1.0,
                        errorbar=('pi', 95), # percentile interval form 2.5th to 97.5th
                        ax=ax,
                        err_kws={'alpha':0.75,'edgecolor':'k','linewidth':0.5})
                
                # set xlims
                ax.set_xlim([0.0, max(times)/data_time_to_mins])

                # set x_ticks and labels only on bottom row
                if stim_idx+1 == n_stim:
                        ax.set_xticks([0.0, 0.5*max(times)/data_time_to_mins, max(times)/data_time_to_mins])
                        if xticklabels is None:
                            ax.set_xticklabels([0, np.ceil(0.5*max(times)/data_time_to_mins), np.ceil(max(times)/data_time_to_mins)])
                        else:
                            ax.set_xticklabels(xticklabels)
                        ax.set_xlabel('Time (min)')
                else:
                        ax.set_xticks([])
                        ax.set_xlabel('')

                # set y_ticks, labels and lims
                ax.set_yticks(y_ticks[stim_idx])

                if 100*y_ticks[stim_idx][1] > 1.0:
                        ax.set_yticklabels([0, int(100*y_ticks[stim_idx][1])])
                else:
                        ax.set_yticklabels([0, 100*y_ticks[stim_idx][1]])
                
                ax.set_ylim(ylim[stim_idx])
                # if (ax.get_ylim()[1]*0.75) < y_ticks[stim_idx][1]:
                #         ax.set_ylim([0.0, y_ticks[stim_idx][1]*1.1])
                # else:
                #         ax.set_ylim([0.0, ax.get_ylim()[1]])

                # plot the data on top, downsample by 10 for visibility
                ax.errorbar(np.hstack((times[::data_downsample],times[-1]))/data_time_to_mins, 
                            np.hstack((data[stim_idx,::data_downsample], data[stim_idx,-1])), 
                            yerr=np.hstack((data_std[stim_idx,::data_downsample], data_std[stim_idx,-1])), 
                            color='black', linestyle='', label='data')
                ax.plot(times/data_time_to_mins, data[stim_idx,:], color='black', 
                        linestyle=':', label='data')

                # set y label to nothing
                ax.set_ylabel('')

                # turn off labels if specified
                if not labels:
                        ax.set_title('')
                        ax.set_xlabel('')
                        ax.set_yticklabels([])
                        ax.set_xticklabels([])

                # save the figure
                fig.savefig(savedir+model_name+fname+str(np.round(EGF_levels[stim_idx], 3))+'.pdf', bbox_inches='tight', transparent=True)

def create_prior_predictive(model, mapk_model_name, data, inputs, savedir, 
            seed=np.random.default_rng(seed=123), trajectory=False, times=None, data_std=None, nsamples=100):
    """ Creates prior predictive samples plot of the stimulus response curve.
    """

    # sample from the prior predictive
    with model:
        prior_predictive = pm.sample_prior_predictive(nsamples, random_seed=seed)

    # extract llike values
    prior_llike = np.squeeze(prior_predictive.prior_predictive['llike'].values)

        # generate the plot
    if not trajectory:
        fig, ax = plot_stimulus_response_curve(prior_llike, data, inputs)
    else:
        fig, ax = plot_trajectory_responses_oneAxis(prior_llike, data, inputs, times,
                                            savedir+mapk_model_name+'_legend_prior_predictive.pdf', data_std=data_std)

    # save the figure
    fig.savefig(savedir + mapk_model_name + '_prior_predictive.pdf', 
                bbox_inches='tight', transparent=True)

    # save the samples
    np.save(savedir + mapk_model_name + '_prior_predictive_samples.npy', prior_llike)

    return fig, ax

def create_posterior_predictive(model, posterior_idata, mapk_model_name, data, inputs, savedir, 
            seed=np.random.default_rng(seed=123), trajectory=False, times=None, data_std=None):
    """ Creates prior predictive samples plot of the stimulus response curve.
    """

    # sample from the posterior predictive
    with model:
        posterior_predictive = pm.sample_posterior_predictive(posterior_idata, model=model, 
                                                              random_seed=seed)

    # extract llike values
    posterior_llike = np.squeeze(posterior_predictive.posterior_predictive['llike'].values)

    # generate the plot
    if not trajectory:
        # reshape accordingly
        nchains,nsamples,ninputs=posterior_llike.shape
        posterior_llike = np.reshape(posterior_llike, (nchains*nsamples, ninputs))

        fig, ax = plot_stimulus_response_curve(posterior_llike, data, inputs)
    else:
        # reshape accordingly
        nchains,nsamples,ninputs,ntime=posterior_llike.shape
        posterior_llike = np.reshape(posterior_llike, (nchains*nsamples, ninputs, ntime))
        print(posterior_llike.shape)
        fig, ax = plot_trajectory_responses_oneAxis(posterior_llike, data, inputs, times,
                                            savedir+mapk_model_name+'_legend_posterior_predictive.pdf', data_std=data_std)

    # save the figure
    fig.savefig(savedir + mapk_model_name + '_posterior_predictive.pdf', 
                bbox_inches='tight', transparent=True)

    # save the samples
    np.save(savedir + mapk_model_name + '_posterior_predictive_samples.npy', posterior_llike)

    return fig, ax

###############################################################################
#### Trajectory analysis + QoI funcs ####
###############################################################################
def time_to_max(trajectory, times):
    """ Get the time to max ERK activity and idx of max"""
    max_idx = np.argmax(trajectory)
    return times[max_idx], max_idx

def time_to_half_max(trajectory, times):
    """ Get the time to half max ERK activity and idx of half max"""
    max_idx = np.argmax(trajectory)
    max_val = trajectory[max_idx]
    half_max_val = max_val/2
    # get idx of first value that is greater than half max
    idx = np.where(trajectory > half_max_val)[0][0]
    return times[idx], idx

def full_half_max_width(trajectory, times):
    # get the half max time and idx
    half_max_time, half_max_idx = time_to_half_max(trajectory, times)
    
    # also get max_idx, max, and half-max
    max_idx = np.argmax(trajectory)
    max_val = trajectory[max_idx]
    half_max = max_val/2

    # get idx of values that are less than half max
    idxs = np.array(np.where(trajectory <= half_max)[0])
    idx = np.where(idxs > half_max_idx)
    if len(idx[0]) != 0:
        second_half_max_idx = idxs[idx[0][0]]
    else:
        second_half_max_idx = len(trajectory)-1

    return times[second_half_max_idx] - times[half_max_idx], second_half_max_idx

def sustained_activity_metric(trajectory, index_of_interest, max_val=None):
    # get max idx and max
    if max_val is None:
        max_idx = np.argmax(trajectory)
        max_val = trajectory[max_idx]

    return (trajectory[index_of_interest] - trajectory[0])/(max_val - trajectory[0])
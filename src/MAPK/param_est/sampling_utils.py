# Utilities to enable sampling for the MAPK models
# Nathaniel Linden (UCSD)
# 10-16-2023

import numpy as np
import pandas as pd
import preliz as pz
import os
import sys
import pymc as pm
import jax.numpy as jnp
import jax
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify
import diffrax

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('~/.matplotlib/custom.mplstyle')
# custom plotting helper funcs
sys.path.insert(0, '/Users/natetest/.matplotlib/')
import plotting_helper_funcs as plt_func


# tell jax to use 64bit floats
jax.config.update("jax_enable_x64", True)

sys.path.append('../')
from utils import *
from diffrax_ODE_PyTensor import *

@jax.jit
def solve_ss(model_dfrx_ode, y0, params, t1):
    """ simulates a model over the specified time interval and returns the 
    calculated steady-state values.
    Returns an array of shape (n_species, 1) """
    dt0=1e-3
    event_rtol=1e-8
    event_atol=1e-8
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
        max_steps=6000,
        throw=False,)
    
    return jnp.array(sol.ys)

# vmap steady state solving over the different inputs it over the parameters
#   this means vmapping over the y0 and assuming everything else is fixed
vsolve_ss = jax.vmap(solve_ss, in_axes=(None, 0, None, None))

def ERK_stim_response(params, model_dfrx_ode, max_time, y0_EGF_inputs, output_states):
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
    # return erk_acts/jnp.max(erk_acts), erk_acts
    return erk_acts/jnp.max(erk_acts), erk_acts

    
def construct_y0_EGF_inputs(EGF_vals, y0, EGF_idx):
    n_repeats = len(EGF_vals)
    y0_EGF_inputs = np.repeat(y0, n_repeats, axis=0)
    y0_EGF_inputs[:, EGF_idx] = EGF_vals

    return y0_EGF_inputs

def load_data(data_file):
    """ Loads the data from the specified file.
    """
    # load the data
    data_df = pd.read_csv(data_file)
    inputs = np.array(data_df['stimulus'].to_numpy())
    data = data_df['response'].to_numpy()

    return inputs, data

    
def set_prior_params(param_names, nominal_params, free_param_idxs, prior_family={'Gamma':['alpha', 'beta']}, upper_mult=1.9, lower_mult=0.1, prob_mass_bounds=0.95):
    """ Sets the prior parameters for the MAPK models.
        Inputs:
            - param_names (list): list of parameter names
            - nominal_params (np.ndarray): array of nominal parameter values
            - free_param_idxs (list): list of indices of the free parameters
            - prior_family (str): prior family to use for the parameters. If a string wil use that family for all free parameters, otherwise should be a list of strings of the same length as free_param_idxs. Each string should correspond to a pm.Distribution and pz.Distribution object, e.g., Gamma which is the default familly.
            - upper_mult (float): multiplier for the upper bound of the prior
            - lower_mult (float): multiplier for the lower bound of the prior
        Returns:
            - prior_param_dict (dict): dictionary of prior parameters for the model in syntax to use exec to set them in a pymc model object
    """

    # determine if a string or list of strings was passed for the prior family
    if len(prior_family) == 1:
        prior_family_list = [list(prior_family.keys())[0]]*len(free_param_idxs)
    else:
        if len(prior_family) != len(free_param_idxs):
            raise ValueError('prior_family must be a string or a list of strings with the same length as free_param_idxs')
        else:
            prior_family_list = list(prior_family.keys())

    # set the prior parameters
    prior_param_dict = {}
    for i, param in enumerate(param_names):
        if i in free_param_idxs: # check if we are dealing with a free parameter
            # get the nominal value
            nominal_val = nominal_params[i]
            # get the upper and lower bounds
            upper = nominal_val*upper_mult
            lower = nominal_val*lower_mult

            # use preliz.maxent to find the prior parameters for the specified family
            prior_fam = prior_family_list[free_param_idxs.index(i)]
            dist_family = eval('pz.' + prior_fam + '()')
            results = pz.maxent(dist_family, lower, upper, prob_mass_bounds, plot=False)[1] # for some reason the [0] element is None

            # set the prior parameters
            tmp = 'pm.' + prior_fam + '("' + param + '",'
            for i, hyper_param in enumerate(prior_family[prior_fam]):
                tmp += hyper_param + '=' + str(results.x[i]) + ', '

            prior_param_dict[param] = tmp + ')'
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

    #################################
    # Create jax functions to solve # 
    #################################
    def sol_op_jax(*params):
        pred, _ = simulator(params, model_dfrx_ode, max_time, y0_EGF_inputs, output_states)
        return jnp.vstack((pred))

    # get the jitted versions
    sol_op_jax_jitted = jax.jit(sol_op_jax)

    ############################################
    # Create pytensor Op and register with jax # 
    ############################################
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
    
    ############################
    # Construct the PyMC model # 
    ############################
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

def plot_stimulus_response_curve(samples, data, inputs, input_name='EGF stimulus', 
                                 output_name='% maximal ERK activity',
                                 data_std=0.1):
    dat = {}
    for i,input in enumerate(inputs):
        dat[input] = samples[:,i]

    data_df = pd.DataFrame(dat)

    fig, ax = plt.subplots()
    sns.boxplot(data=data_df, color='k', ax=ax, whis=(2.5, 97.5), fill=False, 
                native_scale=True, log_scale=(10, 0), fliersize=0)
    ax.set_xlabel(input_name)
    ax.set_ylabel(output_name)

    errors = data_std*np.squeeze(np.ones_like(data))
    ax.scatter(inputs, data, color='r', marker='x', s=50, zorder=10)
    ax.errorbar(inputs, np.squeeze(data), yerr=errors, color='r', fmt='x', markersize=7, zorder=10)

    return fig, ax

def smc_pymc(model, mapk_model_name, savedir, nsamples=2000, 
             seed=np.random.default_rng(seed=123), ncores=None):
    """ Function to run SMC sampling using PyMC and the independent Metropolis-Hastings kernel."""
    with model:
        idata = pm.smc.sample_smc(draws=nsamples, random_seed=seed, chains=None,
                                  cores=ncores, progressbar=False)

    # save the samples
    idata.to_netcdf(savedir + mapk_model_name + '_smc_samples.nc')

    return idata

def mcmc_numpyro_nuts(model, mapk_model_name, savedir, nsamples=2000, 
                      seed=np.random.default_rng(seed=123), ncores=None):
    """ Function to run NUTS sampling using Numpyro."""
    with model:
        idata = pm.sampling.jax.sample_numpyro_nuts(draws=nsamples, 
                    random_seed=seed, chains=4, idata_kwargs={'log_likelihood': True})
    
    # save the samples
    idata.to_netcdf(savedir + mapk_model_name + '_mcmc_numpyro_samples.nc')

def create_prior_predictive(model, mapk_model_name, data, inputs, savedir, 
                            nsamples=100, seed=np.random.default_rng(seed=123)):
    """ Creates prior predictive samples plot of the stimulus response curve.
    """

    # sample from the prior predictive
    with model:
        prior_predictive = pm.sample_prior_predictive(nsamples, random_seed=seed)

    # extract llike values
    prior_llike = np.squeeze(prior_predictive.prior_predictive['llike'].values)

    # generate the plot
    fig, ax = plot_stimulus_response_curve(prior_llike, data, inputs)

    # save the figure
    fig.savefig(savedir + mapk_model_name + '_prior_predictive.pdf', 
                bbox_inches='tight', transparent=True)

    # save the samples
    np.save(savedir + mapk_model_name + '_prior_predictive_samples.npy', prior_llike,)

    return fig, ax

def create_posterior_predictive(model, posterior_idata, mapk_model_name, data, inputs, savedir, 
            seed=np.random.default_rng(seed=123)):
    """ Creates prior predictive samples plot of the stimulus response curve.
    """

    # sample from the prior predictive
    with model:
        posterior_predictive = pm.sample_posterior_predictive(posterior_idata, model=model, 
                                                              random_seed=seed)

    # extract llike values
    posterior_llike = np.squeeze(posterior_predictive.posterior_predictive['llike'].values)
    nchains,nsamples,ninputs=posterior_llike.shape
    posterior_llike = np.reshape(posterior_llike, (nchains*nsamples, ninputs))

    # generate the plot
    fig, ax = plot_stimulus_response_curve(posterior_llike, data, inputs)

    # save the figure
    fig.savefig(savedir + mapk_model_name + '_posterior_predictive.pdf', 
                bbox_inches='tight', transparent=True)

    # save the samples
    np.save(savedir + mapk_model_name + '_posterior_predictive_samples.npy', posterior_llike)

    return fig, ax
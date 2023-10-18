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
    return erk_acts/jnp.max(erk_acts)

    
def construct_y0_EGF_inputs(EGF_vals, y0, EGF_idx):
    n_repeats = len(EGF_vals)
    y0_EGF_inputs = np.repeat(y0, n_repeats, axis=0)
    y0_EGF_inputs[:, EGF_idx] = EGF_vals

    return y0_EGF_inputs

def build_pymc_model(param_names, prior_param_dict, data, y0_EGF_inputs, 
                    output_states, max_time, model_dfrx_ode, model=None, 
                    simulator=ERK_stim_response, data_sigma=0.1):
    """ Builds a pymc model object for the MAPK models.

    Constructs priors for the model, and uses the ERK_stim_response function to 
    generate the stimulus response function and likelihood.
    """

    # define the jax functions for the simulator and vjp
    def sol_op_jax(*params):
        prediction, _ = simulator(params, model_dfrx_ode, max_time, 
                                  y0_EGF_inputs, output_states)
        return prediction
    
    def vjp_sol_op_jax(args, output_grads):
        _, vjpfun = jax.vjp(sol_op_jax, args)
        return vjpfun(output_grads)
    
    # get the jitted versions
    sol_op_jax_jitted = jax.jit(sol_op_jax)
    vjp_sol_op_jax_jitted = jax.jit(vjp_sol_op_jax)

    # construct Ops and register with jax_funcify
    sol_op = SolOp(sol_op_jax_jitted)
    vjp_sol_op = VJPSolOp(vjp_sol_op_jax_jitted) 

    @jax_funcify.register(SolOp)
    def sol_op_jax_funcify(op, **kwargs):
        return sol_op_jax

    @jax_funcify.register(VJPSolOp)
    def vjp_sol_op_jax_funcify(op, **kwargs):
        return vjp_sol_op_jax

    if model is None:
        model = pm.Model()
    
    with model:
        # loop over free params and construct the priors
        for key, value in prior_param_dict.items():
            # create PyMC variables for each parameters in the model
            exec(key + ' = ' + value)
    
        # construct list of parameters to pass to the simulator function
        param_list = tuple(map(eval, param_names))

        # predict dose response
        prediction = sol_op(*param_list)

        # assume a normal model for the data
        # sigma specified by the data_sigma param to this function
        llike = pm.Normal("llike", mu=prediction, sigma=data_sigma, observed=data)

    return model

    
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
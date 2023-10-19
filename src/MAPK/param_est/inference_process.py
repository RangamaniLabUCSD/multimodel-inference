from os import environ
# environ['OMP_NUM_THREADS'] = '1'
#environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
import jax.numpy as jnp
import numpy as np
import diffrax
import sys
import argparse

from SALib.sample import morris as morris_sample

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

sys.path.append("../")
from utils import *

# tell jax to use 64bit floats
jax.config.update("jax_enable_x64", True)

# print out device count
n_devices = jax.local_device_count() 
print(jax.devices())
print('Using {} jax devices'.format(n_devices))

##############################
# def arg parsers to take inputs from the command line
##############################
def parse_args():
    """ function to parse command line arguments
    """
    parser=argparse.ArgumentParser(description="Generate Morris samples for the specified model.")
    parser.add_argument("model", type=str, help="model to process.")
    parser.add_argument("free_params", type=str, help="parameters to estimate")
    parser.add_argument("-data_file", type=str, help="path to the data file. Should be a CSV with one column of inputs and another of outputs data.")
    parser.add_argument("-nsamples", type=int, default=1000, help="Number of samples to posterior samples to draw. Defaults to 1000.")
    parser.add_argument("-savedir", type=str, help="Path to save results. Defaults to current directory.")
    
    args=parser.parse_args()
    return args


def main():
    """ main function to execute command line script functionality.
    """
    args = parse_args()
    print('Processing model {}.'.format(args.model))

    # try calling the model
    try:
        model = eval(args.model + '(transient=False)')
    except:
        print('Warning Model {} not found. Skipping this.'.format(args.model))

    # get parameter names and initial conditions
    p_dict, plist = model.get_nominal_params()
    y0_dict, y0 = model.get_initial_conditions()

    # load the data
    inputs, data = load_data(args.data_file)


    # get the params to sample
    analyze_params = args.free_params.split(',')
    free_param_idxs = jnp.array([list(p_dict.keys()).index(p) for p in analyze_params])

    # construct the strings to make priors and constants
    prior_param_dict = set_prior_params(list(p_dict.keys()), plist, free_param_idxs)

    # make initial conditions that reflect the inputs
    y0_EGF_ins = construct_y0_EGF_inputs(inputs, np.array([y0]), 0)

    # construct the pymc model
    pymc_model = build_pymc_model(prior_param_dict, data, y0_EGF_ins, 
                    [-1], 540, diffrax.ODETerm(model))
    
    # prior predictive sampling
    create_prior_predictive(pymc_model, args.model, data, inputs, args.savedir, 
                            nsamples=500,)
    
    # SMC sampling
    posterior_idata = smc_pymc(pymc_model, args.model, args.savedir, 
                nsamples=args.nsamples)
    
    # posterior predictive sampling
    create_posterior_predictive(pymc_model, posterior_idata, args.model, data, 
                                inputs, args.savedir)

    
    print('Completed {}'.format(args.model))

if __name__ == '__main__':
    main()


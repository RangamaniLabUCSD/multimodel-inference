from os import environ
environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import diffrax
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import jax
import equinox as eqx
import sys
from scipy.stats import mode
import time

sys.path.append("../models/")
from hornberg_2005 import *

jax.config.update("jax_enable_x64", True)

n_devices = jax.local_device_count() 
print('Using {} jax devices'.format(n_devices))


# function to solve
def simulate_model_trajectory(model_instance, y0, params, t1=300, dt0=1e-3, n_out=1000):
    """ simulates a model over the specified time interval and returns the 
    trajectory of the model state variables."""
    ode_term = diffrax.ODETerm(model_instance)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)
    t0 = 0.0
    dt0 = 1e-3
    times = np.linspace(t0, t1, n_out)
    saveat = saveat=diffrax.SaveAt(ts=times)

    sol = diffrax.diffeqsolve(
        ode_term, 
        solver, 
        t0, t1, dt0, 
        y0, 
        saveat=saveat, stepsize_controller=stepsize_controller,
        args=tuple(params),
        max_steps=60000,
        throw=False,)
    
    return sol

# import model
HB_2005 = hornberg_2005(transient=False)
pdict, plist = HB_2005.get_nominal_params()
y0_dict, y0 = HB_2005.get_initial_conditions()

# compile + solve model
start = time.time()
sol = simulate_model_trajectory(HB_2005, y0, plist, t1=6000)
end = time.time()
print('Compile + solves took {} seconds'.format(end-start))

# repeat solve 10 times
n_solves = 10
times = []
for i in range(n_solves):
    start = time.time()
    sol = simulate_model_trajectory(HB_2005, y0, plist, t1=6000)
    end = time.time()
    times.append(end-start)

print('Average solve time: {} seconds'.format(np.mean(times)))

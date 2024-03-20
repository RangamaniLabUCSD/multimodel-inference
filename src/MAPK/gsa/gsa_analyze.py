import jax.numpy as jnp
import numpy as np
import sys
import seaborn as sns
from SALib.analyze import morris
from SALib.analyze import hdmr
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('~/.matplotlib/custom.mplstyle')
mpl.rc('xtick', labelsize=8) 
mpl.rc('ytick', labelsize=8)
mpl.rc('axes', labelsize=10) 
# mpl.rc('ylabel', labelsize=10) 

# custom plotting helper funcs
sys.path.insert(0, '/Users/natetest/.matplotlib/')
import plotting_helper_funcs as plt_func
import seaborn as sns
from adjustText import adjust_text

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


def load_and_process_samples(model_name, qoi_lambda, loaddir, trajectory=False):

    # load stuff
    param_samples = np.load(loaddir + '{}_morris_sample.npy'.format(model_name))
    if not trajectory:
        ss_samples = np.load(loaddir +  '{}_morris_ss.npy'.format(model_name))

        # apply lambda to get qoi
        qoi = np.apply_along_axis(qoi_lambda, 1, ss_samples)
    elif trajectory:
        ss_samples = np.load(loaddir +  '{}_morris_traj.npy'.format(model_name))
        qoi = qoi_lambda(ss_samples)

    qoi[np.isinf(qoi)] = 0.0

    return param_samples, qoi, ss_samples

def plot_qoi_histogram(qoi, model_name, savedir, figsize=(4.0,2.0)):

    fig, ax = plt_func.get_sized_fig_ax(*figsize)
    
    # sns.kdeplot(qoi, ax=ax)
    if np.var(qoi)/np.mean(qoi) > 1e-10:
        g = sns.histplot(qoi, ax=ax, bins=20, kde=True, color='k', stat='density')
        if ax.get_legend() is not None:
            ax.get_legend().set_visible(False)

        ax.set_xlabel('steady-state activated MAPK')

    fig.savefig(savedir + '{}_qoi_hist.pdf'.format(model_name), bbox_inches='tight', transparent=True)
    plt.close()
    

def plot_trajectories(model_name, traj_samples, savedir, figsize=(7.0,2.0), n_traj=1000):

    fig, ax = plt_func.get_sized_fig_ax(*figsize)

    # plot a random subset of trajectories
    traj_idxs = np.random.choice(traj_samples.shape[0], n_traj, replace=False)

    for i in traj_idxs:
        ax.plot(traj_samples[i,:], c='k', alpha=0.25,linewidth=0.5)

    ax.set_xlabel('time index')
    ax.set_ylabel('activated MAPK')

    fig.savefig(savedir + '{}_qoi_traj.pdf'.format(model_name), bbox_inches='tight', transparent=True)

    plt.close()

def analyze_morris(model_name, params_to_analyze, qoi, param_samples, multiplier=0.1,lower=None,upper=None):
    """ Function to perform morris analysis on samples."""
    # try calling the model
    try:
        model = eval(model_name + '(transient=False)')
    except:
        print('Warning Model {} not found. Skipping this.'.format(model_name))

    # get parameter names and initial conditions
    pdict, plist = model.get_nominal_params()
    y0_dict, y0 = model.get_initial_conditions()

    # get the params to analyze
    param_idxs = jnp.array([list(pdict.keys()).index(p) for p in params_to_analyze])

    # get list of nominal vals for ID params
    analyze_nominal_params = jnp.array([pdict[p] for p in params_to_analyze])

    # define the bounds
    if lower is not None and upper is not None:
        bounds = [[p*lower, p*upper] for p in analyze_nominal_params]
        for i,p in enumerate(analyze_nominal_params):
            if p == 0.0:
                bounds[i] = [1e-4, 1.0]
    else:
        bounds = [[p*(1-multiplier), p*(1+multiplier)] for p in analyze_nominal_params]

    # set up the problem dictionary for SALib
    problem = {
        'num_vars': len(analyze_nominal_params),
        'names': params_to_analyze,
        'bounds': bounds,
    }

    morris_results = morris.analyze(problem, param_samples, qoi,)

    return morris_results

def analyze_HDMR(model_name, params_to_analyze, qoi, param_samples, multiplier=0.1,lower=None,upper=None):
    """ Function to perform morris analysis on samples."""
    # try calling the model
    try:
        model = eval(model_name + '(transient=False)')
    except:
        print('Warning Model {} not found. Skipping this.'.format(model_name))

    # get parameter names and initial conditions
    pdict, plist = model.get_nominal_params()
    y0_dict, y0 = model.get_initial_conditions()

    # get the params to analyze
    param_idxs = jnp.array([list(pdict.keys()).index(p) for p in params_to_analyze])

    # get list of nominal vals for ID params
    analyze_nominal_params = jnp.array([pdict[p] for p in params_to_analyze])

    # define the bounds
    if lower is not None and upper is not None:
        bounds = [[p*lower, p*upper] for p in analyze_nominal_params]
        for i,p in enumerate(analyze_nominal_params):
            if p == 0.0:
                bounds[i] = [1e-4, 1.0]
    else:
        bounds = [[p*(1-multiplier), p*(1+multiplier)] for p in analyze_nominal_params]

    # set up the problem dictionary for SALib
    problem = {
        'num_vars': len(analyze_nominal_params),
        'names': params_to_analyze,
        'bounds': bounds,
    }

    hdmr_results = hdmr.analyze(problem, param_samples, qoi,seed=1234)

    return hdmr_results

def write_sensitivity_results(model_name, sensitive_params, filename):

    with open(filename, 'a') as f:
        f.write('\n')
        f.write(model_name + ' : ')
        for p in sensitive_params:
            f.write(p + ', ')

def plot_morris_scatter(morris_results, param_names, model_name, savedir, threshold=0.1):

    fig, ax = plt_func.get_sized_fig_ax(2.0,2.0)

    sensitive_params = []
    annotations = []
    for i in range(len(morris_results['mu_star'])):
        if (morris_results['mu_star'][i]/np.max(morris_results['mu_star']) >= threshold) or \
            (morris_results['sigma'][i]/np.max(morris_results['sigma']) >= threshold):
            ax.scatter(morris_results['mu_star'][i], morris_results['sigma'][i], s=10, c='b')
            annotations.append(ax.annotate(param_names[i], (morris_results['mu_star'][i], morris_results['sigma'][i],), fontsize=8))
            sensitive_params.append(param_names[i])
        else:
            ax.scatter(morris_results['mu_star'][i], morris_results['sigma'][i], s=10, c='k')

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.plot([threshold*np.max(morris_results['mu_star']), threshold*np.max(morris_results['mu_star'])], ylim, 'r:', linewidth=1.0)
    ax.plot(xlim, [threshold*np.max(morris_results['sigma']), threshold*np.max(morris_results['sigma'])], 'r:', linewidth=1.0)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    ax.set_xlabel(r'$\mu^*$')
    ax.set_ylabel(r'$\sigma$')

    # move the annotations around so they don't overlap
    adjust_text(annotations, ax=ax)

    # save figure and and return fig and axes
    fig.savefig(savedir + '{}_morris_scatter.pdf'.format(model_name), bbox_inches='tight', transparent=True)
    plt.close()

    return sensitive_params

def plot_HDMR_bar(hdmr_results, param_names, model_name, savedir,threshold=0.01,figsize=(4.0,2.0)):

    # sort S
    S_idxs = np.flip(np.argsort(hdmr_results['S'][0:len(param_names)]))
    S_sorted = hdmr_results['S'][0:len(param_names)][S_idxs]
    S_conf_sorted = hdmr_results['S_conf'][0:len(param_names)][S_idxs]
    S_names = np.array(param_names)[S_idxs]

    # determine parameters that are above the threshold
    sensitive_param_idxs = np.where((S_sorted+S_conf_sorted)>=threshold)
    sensitive_params = S_names[sensitive_param_idxs]

    fig, ax = plt_func.get_sized_fig_ax(*figsize)
    ax.bar(S_names, S_sorted, yerr=S_conf_sorted, color='k')
    ax.bar(S_names[sensitive_param_idxs], S_sorted[sensitive_param_idxs], 
        yerr=S_conf_sorted[sensitive_param_idxs], color='b')
    # labels
    ax.set_ylabel('HDMR Sensitivity $S_i$', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    # line to show threshold
    xlim = ax.get_xlim()
    ax.plot(xlim, [threshold,threshold], '--', color='r')
    ax.set_xlim(xlim)

    # save figure and and return fig and axes
    fig.savefig(savedir + '{}_hdmr_S.pdf'.format(model_name), bbox_inches='tight', transparent=True)

    return sensitive_params

def run_analysis(model_name, qoi_lambda, params_to_analyze, loaddir, savedir, lower=1e-2, upper=1e2):
    # load and process samples
    param_samples, qoi, traj_samples = load_and_process_samples(model_name, qoi_lambda, loaddir, trajectory=True)
    
    # make plot of trajectories and histogram of the QoI samples
    cutoff = 2500
    if qoi.shape[0] > cutoff:
        n_traj = cutoff
    else:
        n_traj = qoi.shape[0]
    plot_trajectories(model_name, traj_samples, savedir, n_traj=n_traj)
    plot_qoi_histogram(qoi, model_name, savedir)
    
    # run Morris GSA and plot
    morris_results = analyze_morris(model_name, params_to_analyze, qoi, param_samples)
    sensitive_params = plot_morris_scatter(morris_results, params_to_analyze, model_name, savedir)
    write_sensitivity_results(model_name, sensitive_params,  './sensitive_params_Morris.txt')

    # run HDMR GSA and plot
    # hdmr_results = analyze_HDMR(model_name, params_to_analyze, qoi, param_samples, lower=lower,upper=upper)
    # sensitive_params = plot_HDMR_bar(hdmr_results, params_to_analyze, model_name, savedir,threshold=0.01,figsize=(4.0,2.0))
    # write_sensitivity_results(model_name, sensitive_params,  './sensitive_params_HDMR.txt')
    print('Completed {}'.format(model_name))
    

def main():
    """ main function to run analysis on all models.

    Note: params to analyze are the structurally locally ID params for each model.
    """
    savedir = '../../../results/MAPK/gsa/figs/'
    loaddir = '../../../results/MAPK/gsa/'

    # # HUANG FERRELL 1996
    model_name = 'huang_ferrell_1996'
    qoi_lambda = lambda x: x[:,-1]
    params_to_analyze = ['MKK_tot','a3','k7','a4','d3','d10','d2','a2','d6','a8','a1','E2_tot','a7','a9','k5','a6','d1','d9','d5','k8','d8','k4','k6','k9','k3','d7','a10','MAPK_tot','k2','d4','a5','MKKK_tot','k10','MKKPase_tot','k1']
    run_analysis(model_name, qoi_lambda, params_to_analyze, loaddir, savedir, lower=1e-2, upper=1e2)

    # # KHOLODENKO 2000
    model_name = 'kholodenko_2000'
    qoi_lambda = lambda x: x[:,-1]
    params_to_analyze = ['K8','v10','v9','K7','K9','KI','MAPK_total','K10']
    run_analysis(model_name, qoi_lambda, params_to_analyze, loaddir, savedir, lower=1e-2, upper=1e2)

    # LEVCHENKO 2000
    model_name = 'levchenko_2000'
    qoi_lambda = lambda x: x[:,-1]
    params_to_analyze = ['kOff1','a2','kOn1','k10','RAFPase','k4','k6','kOff4','kOn2','total_scaffold','d7','d9','d3','a3','a10','kOff3','k8','k5','a9','a8','a5','d8','d5','a6','k7','d2','a7','d10','a1','a4','k9','d6','k2','d1','k3','MEKPase','kOff2','MAPKPase','d4','k1']
    run_analysis(model_name, qoi_lambda, params_to_analyze, loaddir, savedir, lower=1e-2, upper=1e2)

    # HORNBERG 2005
    model_name = 'hornberg_2005'
    qoi_lambda = lambda x: x[:,-1]
    params_to_analyze = ['kd1','k2','kd2','k3','kd3','k4','kd4','k5','kd5','k6','kd6','k8','kd8','k10b','kd10','k13','kd13','k15','kd15','k16','kd63','k17','kd17','k18','kd18','k19','kd19','k20','kd20','k21','kd21','k22','kd22','k23','kd23','kd24','k25','kd25','k28','kd28','k29','kd29','k32','kd32','k33','kd33','k34','kd34','k35','kd35','k36','kd36','k37','kd37','k40','kd40','k41','kd41','k42','kd42','k43','kd43','k44','kd52','k45','kd45','k47','kd47','k48','kd48','k49','kd49','k50','kd50','k52','kd44','k53','kd53','k55','kd55','k56','kd56','k57','kd57','k58','kd58','k60','kd60','k61','kd61','k126','kd126','k127','kd127',]
    run_analysis(model_name, qoi_lambda, params_to_analyze, loaddir, savedir, lower=1e-2, upper=1e2)

    # ORTON 2009
    model_name = 'orton_2009'
    qoi_lambda = lambda x: x[:,-1]
    params_to_analyze = ['km_Erk_Activation','k1_C3G_Deactivation','km_Erk_Deactivation','k1_Akt_Deactivation','k1_P90Rsk_Deactivation','k1_PI3K_Deactivation','k1_Sos_Deactivation']
    run_analysis(model_name, qoi_lambda, params_to_analyze, loaddir, savedir, lower=1e-2, upper=1e2)

    # VON KRIEGSHEIM 2009
    model_name = 'vonKriegsheim_2009'
    qoi_lambda = lambda x: x[:,-1]
    params_to_analyze = ['k42','k37','k4','k27','k45','k30','k43','k48','k5','k14','k28','k39','k46','k63','k68','k55','k29','k41','k25','k7','k13','k2','k40','k6','k18','k56','k32','k38','k10','k34']
    run_analysis(model_name, qoi_lambda, params_to_analyze, loaddir, savedir, lower=1e-2, upper=1e2)

    # SHIN 2014
    model_name = 'shin_2014'
    qoi_lambda = lambda x: x[:,-1]
    params_to_analyze = ['kc47','kc43','kd39','kc45','ERK_tot','ki39','kc41']
    run_analysis(model_name, qoi_lambda, params_to_analyze, loaddir, savedir, lower=1e-2, upper=1e2)

    # RYU 2015
    model_name = 'ryu_2015'
    qoi_lambda = lambda x: x[:,-1]
    params_to_analyze = ['D2','T_dusp','K_dusp','K2','dusp_ind']
    run_analysis(model_name, qoi_lambda, params_to_analyze, loaddir, savedir, lower=1e-2, upper=1e2)

    # KOCHANCZYK 2017
    model_name = 'kochanczyk_2017'
    qoi_lambda = lambda x: x[:,-1]
    params_to_analyze = ['k3','q1','q3','q2','u3','d1','q6','u2b','u1a','d2','u2a','q5','u1b','q4']
    # run_analysis(model_name, qoi_lambda, params_to_analyze, loaddir, savedir, lower=1e-2, upper=1e2)

    # BIRTWISTLE 2007
    model_name = 'birtwistle_2007'
    qoi_lambda = lambda x: x[:,-1]
    params_to_analyze = ['koff67','Kmf52','koff57','EGF_off','koff29','koff31','koff89','kcat90','koff91','koff40','koff61','koff77','koff21','koff45','koff68','koff24','kf12','b98','HRGoff_4','VeVc','koff4','koff78','koff26','koff88','koff22','a98','koff8','koff28','kf14','koff76','koff25','koff73','koff95','koff59','koff66','koff65','Kmr52','koff33','kf15','koff6','kon91','kf13','kcat94','koff30','koff42','kon93','koff70','koff58','kf11','koff19','kcat96','koff36','kcat92','koff17','HRGoff_3','koff46','koff71','koff34','koff20','koff72','kcon49','kf63','kdeg','koff93','koff35','koff5','koff18','koff41','koff32','Vmaxr52','koff74','koff75','koff27','koff43','koff62','koff23','koff37','koff44','koff80','koff60','kf48','koff69','koff16','kf64','koff9','kon89','kf10','koff79','kon95','koff7']
    run_analysis(model_name, qoi_lambda, params_to_analyze, loaddir, savedir, lower=1e-2, upper=1e2)

if __name__ == '__main__':
    main()


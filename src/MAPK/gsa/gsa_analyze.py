import jax.numpy as jnp
import numpy as np
import sys
import seaborn as sns
from SALib.analyze import morris
import matplotlib.pyplot as plt

plt.style.use('~/.matplotlib/custom.mplstyle')

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


def plot_morris_scatter(morris_results, param_names, model_name, savedir, threshold=0.1):

    fig, ax = plt_func.get_sized_fig_ax(3.0,3.0)

    annotations = []
    for i in range(len(morris_results['mu_star'])):
        if morris_results['mu_star'][i]/np.max(morris_results['mu_star']) >= threshold:
            ax.scatter(morris_results['mu_star'][i]/np.max(morris_results['mu_star']), morris_results['sigma'][i]/np.max(morris_results['sigma']), s=10, c='b')
            annotations.append(ax.annotate(param_names[i], (morris_results['mu_star'][i]/np.max(morris_results['mu_star']), morris_results['sigma'][i]/np.max(morris_results['sigma']),), fontsize=8))
        else:
            ax.scatter(morris_results['mu_star'][i]/np.max(morris_results['mu_star']), morris_results['sigma'][i]/np.max(morris_results['sigma']), s=10, c='k')

    ylim = ax.get_ylim()
    ax.plot([threshold, threshold], ylim, 'r:', linewidth=1.0)
    ax.set_ylim(ylim)
    
    ax.set_xlabel(r'normalized $\mu^*$')
    ax.set_ylabel(r'normalized $\sigma$')

    # move the annotations around so they don't overlap
    adjust_text(annotations, ax=ax)

    # save figure and and return fig and axes
    fig.savefig(savedir + '{}_morris_scatter.pdf'.format(model_name), bbox_inches='tight', transparent=True)
    return fig, ax

def plot_qoi_histogram(qoi, model_name, savedir):

    fig, ax = plt_func.get_sized_fig_ax(3.0,3.0)

    # sns.kdeplot(qoi, ax=ax)
    if np.var(qoi) > 1e-10:
        g = sns.histplot(qoi, ax=ax, bins=20, kde=True, color='k', stat='density')
        ax.get_legend().set_visible(False)

    ax.set_xlabel('steady-state activated MAPK')

    fig.savefig(savedir + '{}_qoi_hist.pdf'.format(model_name), bbox_inches='tight', transparent=True)

    return fig, ax

def analyze_morris(model_name, params_to_analyze, qoi, param_samples, multiplier=0.1):
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
    bounds = [[p*(1-multiplier), p*(1+multiplier)] for p in analyze_nominal_params]

    # set up the problem dictionary for SALib
    problem = {
        'num_vars': len(analyze_nominal_params),
        'names': params_to_analyze,
        'bounds': bounds,
    }

    morris_results = morris.analyze(problem, param_samples, qoi)

    return morris_results

def load_and_process_samples(model_name, qoi_lambda):

    # load stuff
    param_samples = np.load('./{}_morris_sample.npy'.format(model_name))
    ss_samples = np.load('./{}_morris_ss.npy'.format(model_name))

    # apply lambda to get qoi
    qoi = np.apply_along_axis(qoi_lambda, 1, ss_samples)

    return param_samples, qoi, ss_samples


def main():
    """ main function to run analysis on all models.
    """
    savedir = '../../../results/MAPK/gsa/figs/'
    # HUANG FERRELL 1996
    model_name = 'huang_ferrell_1996'
    qoi_lambda = lambda x: x[-1]+x[-2]
    params_to_analyze = ['MKK_tot','a3','k7','a4','d3','d10','d2','a2','d6','a8','a1','E2_tot','a7','a9','k5','a6','d1','d9','d5','k8','d8','k4','k6','k9','k3','d7','a10','MAPK_tot','k2','d4','a5','MKKK_tot','k10','MKKPase_tot','k1']
    param_samples, qoi, _ = load_and_process_samples(model_name, qoi_lambda)
    morris_results = analyze_morris(model_name, params_to_analyze, qoi, param_samples)
    plot_morris_scatter(morris_results, params_to_analyze, model_name, savedir)
    plot_qoi_histogram(qoi, model_name, savedir)

    # KHOLODENKO 2000
    model_name = 'kholodenko_2000'
    qoi_lambda = lambda x: x[-1]
    params_to_analyze = ['K8','v10','v9','K7','K9','KI','MAPK_total','K10']
    param_samples, qoi, _ = load_and_process_samples(model_name, qoi_lambda)
    morris_results = analyze_morris(model_name, params_to_analyze, qoi, param_samples)
    plot_morris_scatter(morris_results, params_to_analyze, model_name, savedir)
    plot_qoi_histogram(qoi, model_name, savedir)

    # LEVCHENKO 2000
    # model_name = 'levchenko_2000'
    # qoi_lambda = lambda x: x[16]
    # params_to_analyze = 
    # param_samples, qoi, _ = load_and_process_samples(model_name, qoi_lambda)
    # morris_results = analyze_morris(model_name, params_to_analyze, qoi, param_samples)
    # plot_morris_scatter(morris_results, params_to_analyze, model_name, savedir)
    # plot_qoi_histogram(qoi, model_name, savedir)

    # BRIGHTMAN FELL 2000
    model_name = 'brightman_fell_2000'
    qoi_lambda = lambda x: x[26]
    params_to_analyze = ['kn14','K_24','kn16','V_26','kn1','k3','V_24','kn12','k15','k_13','kn7','kn11','K_25','k2_4','K_26','k17','K_23','DT']
    param_samples, qoi, _ = load_and_process_samples(model_name, qoi_lambda)
    morris_results = analyze_morris(model_name, params_to_analyze, qoi, param_samples)
    plot_morris_scatter(morris_results, params_to_analyze, model_name, savedir)
    plot_qoi_histogram(qoi, model_name, savedir)

    # HATAKEYAMA 2003
    model_name = 'hatakeyama_2003'
    qoi_lambda = lambda x: x[32]
    params_to_analyze = ['k21','k20','kb29','kf9','kb24','kb3','kb23','kf25','kb1','kb2','kf24','kb7','kf8','k19','kf34','k22','kb5','kf6','kf3','kb6']
    param_samples, qoi, _ = load_and_process_samples(model_name, qoi_lambda)
    morris_results = analyze_morris(model_name, params_to_analyze, qoi, param_samples)
    plot_morris_scatter(morris_results, params_to_analyze, model_name, savedir)
    plot_qoi_histogram(qoi, model_name, savedir)

    # HORNBERG 2005
    model_name = 'hornberg_2005'
    qoi_lambda = lambda x: x[58]+x[82]
    params_to_analyze = ['k42','k28','k52','kd50','k20','k6','kd45','k3','k18','k17','k25','k48','kd48','kd127','kd3','kd10','kd40','k61','kd5','k33','k16','kd22','kd4','kd34','kd44','k15','kd32','k10b','kd49','kd57','kd20','k21','k40','kd52','kd58','kd1','k8','kd53','kd35','k37','kd56','kd42','kd6','kd126','k35','kd23','kd33','kd47','kd55','kd25','kd18','kd19','k32','kd28','kd37','k44','kd8','kd17','k2','k19','k50','k41','k13','k34','kd21','kd41','k60','k126','k23','k29','kd29','kd2','k4','k58','k22','kd63','kd24','k56','k36']
    param_samples, qoi, _ = load_and_process_samples(model_name, qoi_lambda)
    morris_results = analyze_morris(model_name, params_to_analyze, qoi, param_samples)
    plot_morris_scatter(morris_results, params_to_analyze, model_name, savedir)
    plot_qoi_histogram(qoi, model_name, savedir)

    # BIRTWISTLE 2007
    # model_name = 'birtwistle_2007'
    # qoi_lambda = lambda x: x[75]+x[115]
    # params_to_analyze = 
    # param_samples, qoi, _ = load_and_process_samples(model_name, qoi_lambda)
    # morris_results = analyze_morris(model_name, params_to_analyze, qoi, param_samples)
    # plot_morris_scatter(morris_results, params_to_analyze, model_name, savedir)
    # plot_qoi_histogram(qoi, model_name, savedir)

    # ORTON 2009
    model_name = 'orton_2009'
    qoi_lambda = lambda x: x[15]
    params_to_analyze = ['km_Erk_Activation','k1_C3G_Deactivation','km_Erk_Deactivation','k1_Akt_Deactivation','k1_P90Rsk_Deactivation','k1_PI3K_Deactivation','k1_Sos_Deactivation']
    param_samples, qoi, _ = load_and_process_samples(model_name, qoi_lambda)
    morris_results = analyze_morris(model_name, params_to_analyze, qoi, param_samples)
    plot_morris_scatter(morris_results, params_to_analyze, model_name, savedir)
    plot_qoi_histogram(qoi, model_name, savedir)

    # VON KRIEGSHEIM 2009
    # model_name = 'vonKriegsheim_2009'
    # qoi_lambda = lambda x: x[26]+x[28]+x[29]
    # params_to_analyze = ['k42','k37','k4','k27','k45','k30','k43','k48','k5','k14','k28','k39','k46','k63','k68','k55','k29','k41','k25','k7','k13','k2','k40','k6','k18','k56','k32','k38','k10','k34']
    # param_samples, qoi, _ = load_and_process_samples(model_name, qoi_lambda)
    # morris_results = analyze_morris(model_name, params_to_analyze, qoi, param_samples)
    # plot_morris_scatter(morris_results, params_to_analyze, model_name, savedir)
    # plot_qoi_histogram(qoi, model_name, savedir)

    # SHIN 2014
    model_name = 'shin_2014'
    qoi_lambda = lambda x: x[-1]
    params_to_analyze = ['kc47','kc43','kd39','kc45','ERK_tot','ki39','kc41']
    param_samples, qoi, _ = load_and_process_samples(model_name, qoi_lambda)
    morris_results = analyze_morris(model_name, params_to_analyze, qoi, param_samples)
    plot_morris_scatter(morris_results, params_to_analyze, model_name, savedir)
    plot_qoi_histogram(qoi, model_name, savedir)

    # RYU 2015
    model_name = 'ryu_2015'
    qoi_lambda = lambda x: x[10]
    params_to_analyze = ['D2','T_dusp','K_dusp','K2','dusp_ind']
    param_samples, qoi, ss_samples = load_and_process_samples(model_name, qoi_lambda)
    morris_results = analyze_morris(model_name, params_to_analyze, qoi, param_samples)
    plot_morris_scatter(morris_results, params_to_analyze, model_name, savedir)
    plot_qoi_histogram(qoi, model_name, savedir)

    # KOCHANCZYK 2017
    model_name = 'kochanczyk_2017'
    qoi_lambda = lambda x: x[24]
    params_to_analyze = ['k3','q1','q3','q2','u3','d1','q6','u2b','u1a','d2','u2a','q5','u1b','q4']
    param_samples, qoi, ss_samples = load_and_process_samples(model_name, qoi_lambda)
    morris_results = analyze_morris(model_name, params_to_analyze, qoi, param_samples)
    plot_morris_scatter(morris_results, params_to_analyze, model_name, savedir)
    plot_qoi_histogram(qoi, model_name, savedir)



if __name__ == '__main__':
    main()


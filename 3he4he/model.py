'''
    model
    - 5/2+ excited state width
    + SONIK energy shift
'''

import os
import sys
import numpy as np
from scipy import stats
import constants as const

pwd = os.getcwd()
i = pwd.find('7Be')

# Import pyazr classes.
from brick.azr import AZR

def name():
    return __name__


input_filename = name() + '.azr'
output_filenames = [
    'AZUREOut_aa=1_R=2.out',
    'AZUREOut_aa=1_R=3.out',
    'AZUREOut_aa=1_TOTAL_CAPTURE.out',
    'AZUREOut_aa=1_R=1.out'
]

# data files to which we will apply a normalization factor
# (not really, we apply it to the theory calculations)
capture_data_files = [
    'Seattle_XS.dat',
    'Weizmann_XS.dat',
    'LUNA_XS.dat',
    'ERNA_XS.dat',
    'ND_XS.dat',
    'ATOMKI_XS.dat',
]

scatter_data_files = [
    'cross_general_apr_lab_AZURE2_1820.dat',
    'cross_general_apr_lab_AZURE2_1441.dat',
    'cross_general_apr_lab_AZURE2_1196.dat',
    'cross_general_apr_lab_AZURE2_873_2.dat',
    'cross_general_apr_lab_AZURE2_711.dat',
    'cross_general_apr_lab_AZURE2_586.dat',
    'cross_general_apr_lab_AZURE2_432.dat',
    'cross_general_apr_lab_AZURE2_291.dat',
    'cross_general_apr_lab_AZURE2_239.dat',
    'cross_general_apr_lab_AZURE2_873_1.dat'
]

ns_capture = [np.size(np.loadtxt('data/'+f)[:, 0]) for f in capture_data_files]
ns_scatter = [np.size(np.loadtxt('data/'+f)[:, 0]) for f in scatter_data_files]
ncapture = sum(ns_capture)
nscatter = sum(ns_scatter)
ns_data = ns_capture + [nscatter]
n_data = sum(ns_capture) + nscatter

use_brune = True
observable_only = False

xs2_data = np.loadtxt('output/'+output_filenames[0]) # capture XS to GS
xs3_data = np.loadtxt('output/'+output_filenames[1]) # capture XS to ES
xs_data = np.loadtxt('output/'+output_filenames[2]) # total capture
xs1_data = np.loadtxt('output/'+output_filenames[3]) # scattering data

scatter = xs1_data[:, 5]
# FIXED RELATIVE EXTRINSIC UNCERTAINTY ADDED IN QUADRATURE BELOW
scatter_err = np.sqrt(xs1_data[:, 6]**2 + (0.018*scatter)**2)
xs2 = xs2_data[:, 5]
xs2_err = xs2_data[:, 6]
xs3 = xs3_data[:, 5]
xs3_err = xs3_data[:, 6]

br = xs2
br_err = xs3_err

x = np.hstack((xs2_data[:, 0], xs_data[:, 0], xs1_data[:, 0]))
y = np.hstack((br, xs_data[:, 5], scatter))
dy = np.hstack((br_err, xs_data[:, 6], scatter_err))

nbr = xs2_data.shape[0]
nxs = xs_data.shape[0]
nscatter1 = xs1_data.shape[0]

assert nscatter == nscatter1, f'''
Number of scattering ({nscatter}, {nscatter1}) data points is inconsistent.
'''

azr = AZR(input_filename)
azr.ext_capture_file = 'output/intEC.dat'
azr.root_directory = '/tmp/'

rpar_labels = [p.label for p in azr.parameters]

f_labels = [
    '$f_{Seattle}$', '$f_{Weizmann}$', '$f_{Luna}$', '$f_{Erna}$', '$f_{ND}$', '$f_{Atomki}$',
    '$f_{1820}$', '$f_{1441}$', '$f_{1196}$', '$f_{873-2}$', '$f_{711}$',
    '$f_{586}$', '$f_{432}$', '$f_{291}$', '$f_{239}$', '$f_{873-1}$',
    r'$\Delta_{\rm{E, Paneru}}$'
]

labels = rpar_labels + f_labels

nrpar = len(rpar_labels)
nf_capture = len(capture_data_files)
nf_scatter = len(scatter_data_files)
ndim = nrpar + nf_capture + nf_scatter + 1

assert ndim == len(labels), f'Number of sampled parameters does not match the \
number of labels: {ndim} ≠ {len(labels)}'

def map_uncertainty(theta, ns):
    c = np.array([])
    for (theta_i, n) in zip(theta, ns):
        c = np.hstack((c, theta_i*np.ones(n)))
    return c


paneru_indices = np.arange(14, 24)

def calculate(theta):
    fj_capture = map_uncertainty(
        theta[nrpar:nrpar+nf_capture], ns_capture
    )
    fj_scatter = map_uncertainty(
        theta[nrpar+nf_capture:nrpar+nf_capture+nf_scatter], ns_scatter
    )
    paneru_shift = theta[-1]

    shifts = paneru_shift * np.ones(10)
    shifted_data = [(i, azr.config.data.segments[i].shift_energies(shift)) for
        (i, shift) in zip(paneru_indices, shifts)]

    mu = azr.predict(theta, mod_data=shifted_data)
    paneru, capture_gs, capture_es, capture_tot = mu

    bratio = capture_es.xs_com_fit/capture_gs.xs_com_fit
    sigma_tot = capture_tot.xs_com_fit
    scatter_dxs = paneru.xs_com_fit
    data_norm = np.hstack((fj_capture*sigma_tot, fj_scatter*scatter_dxs))
    return np.hstack((bratio, data_norm))


# starting positions
initial_capture_norms = np.ones(nf_capture)
initial_scatter_norms = np.ones(nf_scatter)
initial_values = azr.config.get_input_values().copy()

starting_positions = np.hstack(
    (initial_values, initial_capture_norms, initial_scatter_norms, 1e-3)
)

assert ndim == np.size(starting_positions), f'Number of sampled parameters \
does not match the number of starting parameters: \
{ndim} ≠ {np.size(starting_positions)}'

# starting position distributions
p0_dist = [stats.norm(sp, np.abs(sp)/100) for sp in starting_positions]

def my_truncnorm(mu, sigma, lower, upper):
    return stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)


# priors
anc429_prior = stats.uniform(1, 4)
Ga_1hm_prior = stats.uniform(-200e6, 400e6)

Ga_1hp_prior = stats.uniform(0, 100e6)
Gg0_1hp_prior = stats.uniform(0, 10e6)
Gg429_1hp_prior = stats.uniform(-10e3, 20e3)

anc0_prior = stats.uniform(1, 4)
Ga_3hm_prior = stats.uniform(-100e6, 200e6)

Ga_3hp_prior = stats.uniform(0, 100e6)
Gg0_3hp_prior = stats.uniform(-10e3, 20e3)
Gg429_3hp_prior = stats.uniform(-3e3, 6e3)

Ga_5hm_prior = stats.uniform(0, 100e6)
Ga_5hp_prior = stats.uniform(0, 100e6)
Gg0_5hp_prior = stats.uniform(-100e6, 200e6)

Ex_7hm_prior = stats.uniform(1, 9)
Ga_7hm_prior = stats.uniform(0, 10e6)
Gg0_7hm_prior = stats.uniform(0, 1e3)

f_seattle_prior = my_truncnorm(1, 0.03, 0, np.inf)
f_weizmann_prior = my_truncnorm(1, 0.037, 0, np.inf)
f_luna_prior = my_truncnorm(1, 0.032, 0, np.inf)
f_erna_prior = my_truncnorm(1, 0.05, 0, np.inf)
f_nd_prior = my_truncnorm(1, 0.08, 0, np.inf)
f_atomki_prior = my_truncnorm(1, 0.06, 0, np.inf)

Delta_E_paneru_prior = stats.norm(0, 3e-3)

# Add 1% (in quadrature) to systematic uncertainty to account for beam-position
# uncertainty.
som_syst = np.sqrt(const.som_syst**2 + 0.01**2)
som_priors = [my_truncnorm(1, syst, 0, np.inf) for syst in som_syst]

priors = [
    anc429_prior,
    Ga_1hm_prior,
    Ga_1hp_prior,
    Gg0_1hp_prior,
    Gg429_1hp_prior,
    anc0_prior,
    Ga_3hm_prior,
    Ga_3hp_prior,
    Gg0_3hp_prior,
    Gg429_3hp_prior,
    Ga_5hm_prior,
    Ga_5hp_prior,
    Gg0_5hp_prior,
    Ex_7hm_prior,
    Ga_7hm_prior,
    Gg0_7hm_prior,
    f_seattle_prior,
    f_weizmann_prior,
    f_luna_prior,
    f_erna_prior,
    f_nd_prior,
    f_atomki_prior
] + som_priors + [Delta_E_paneru_prior]

assert len(priors) == ndim, 'Number of priors does not match number of sampled parameters.'

def logpi(theta):
    return np.sum([prior.logpdf(theta_i) for (prior, theta_i) in zip(priors, theta)])


def logl(mu):
    return np.sum(-np.log(const.M_SQRT2PI*dy) - 0.5*((y-mu)/dy)**2)


'''
    log(Posterior)
'''

# Make sure this matches with what log_posterior returns.
blobs_dtype = [('loglikelihood', float)]

def log_posterior(theta):
    logprior = logpi(theta)
    if logprior == -np.inf:
        return -np.inf, -np.inf

    mu = calculate(theta)
    loglikelihood = logl(mu)
    if np.isnan(loglikelihood):
        return -np.inf, -np.inf

    fj_capture = theta[nrpar:nrpar+nf_capture]
    fj_scatter = theta[nrpar+nf_capture:]
    
    return loglikelihood + logprior, loglikelihood

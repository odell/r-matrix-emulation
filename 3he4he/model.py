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

input_filename = __name__ + '.azr'
azr = AZR(input_filename)
azr.ext_capture_file = 'output/intEC.dat'
azr.root_directory = '/tmp/'

output_filenames = [
    'AZUREOut_aa=1_R=1.out',
    'AZUREOut_aa=1_R=2.out',
    'AZUREOut_aa=1_R=3.out',
    'AZUREOut_aa=1_TOTAL_CAPTURE.out'
]

# Capture Data
# We will apply normalization factors to the capture data. That data is returned
# in a single column, so we need to know what order they are in and how many
# points are in each set.
capture_data_files = [
    'Seattle_XS.dat',
    'Weizmann_XS.dat',
    'LUNA_XS.dat',
    'ERNA_XS.dat',
    'ND_XS.dat',
    'ATOMKI_XS.dat',
]
num_pts_capture = [
    np.size(np.loadtxt('data/'+f)[:, 0]) for f in capture_data_files
]
num_pts_total_capture = sum(num_pts_capture)

# Scattering Data
scatter_data_files = [
    'cross_general_apr_lab_AZURE2_239.dat',
    'cross_general_apr_lab_AZURE2_291.dat',
    'cross_general_apr_lab_AZURE2_432.dat',
    'cross_general_apr_lab_AZURE2_586.dat',
    'cross_general_apr_lab_AZURE2_711.dat',
    'cross_general_apr_lab_AZURE2_873_1.dat',
    'cross_general_apr_lab_AZURE2_873_2.dat',
    'cross_general_apr_lab_AZURE2_1196.dat',
    'cross_general_apr_lab_AZURE2_1441.dat',
    'cross_general_apr_lab_AZURE2_1820.dat'
]
num_pts_scatter = [np.size(np.loadtxt('data/'+f)[:, 0]) for f in scatter_data_files]
num_pts_total_scatter = sum(num_pts_scatter)

# Center-of-Mass (COM) Data
xs1_data = np.loadtxt('output/'+output_filenames[0]) # scattering data
xs2_data = np.loadtxt('output/'+output_filenames[1]) # capture XS to GS
xs3_data = np.loadtxt('output/'+output_filenames[2]) # capture XS to ES
xs_data = np.loadtxt('output/'+output_filenames[3]) # total capture
## COM Scattering Data
scatter = xs1_data[:, 5]
# FIXED RELATIVE EXTRINSIC UNCERTAINTY ADDED IN QUADRATURE BELOW
scatter_err = np.sqrt(xs1_data[:, 6]**2 + (0.018*scatter)**2)
# COM Branching Ratio Data
# James sent me this setup. He used a modified version of AZURE2 that computes
# the branching ratio. So the data values in xs2_data and xs3_data both
# correspond to the measured data. No need to divide xs3 by xs2.
branching_ratio = xs2_data[:, 5]
branching_ratio_err = xs2_data[:, 6]

x = np.hstack((xs2_data[:, 0], xs_data[:, 0], xs1_data[:, 0])) # all energies
y = np.hstack((branching_ratio, xs_data[:, 5], scatter)) # all observables
dy = np.hstack((branching_ratio_err, xs_data[:, 6], scatter_err)) # all uncertainties

nbr = xs2_data.shape[0]
nxs = xs_data.shape[0]

assert num_pts_total_scatter == xs1_data.shape[0], f'''
Number of scattering ({num_pts_total_scatter}, {xs1_data.shape[0]}) data points is inconsistent.
'''

nrpar = len(azr.parameters)
nf_capture = len(capture_data_files)
nf_scatter = len(scatter_data_files)
ndim = nrpar + nf_capture + nf_scatter

def map_uncertainty(theta, ns):
    return np.hstack([theta_i*np.ones(n) for (theta_i, n) in zip(theta, ns)])


# Calculate the branching ratios, total capture cross sections, and scattering
# differential cross sections at point theta.
def calculate(theta):
    paneru, capture_gs, capture_es, capture_tot = azr.predict(theta)
    bratio = capture_es.xs_com_fit/capture_gs.xs_com_fit
    sigma_tot = capture_tot.xs_com_fit
    scatter_dxs = paneru.xs_com_fit
    return np.hstack((bratio, sigma_tot, scatter_dxs))


# starting position distributions
p0_dist = [stats.norm(sp, np.abs(sp)/100) for sp in
        azr.config.get_input_values()]

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
] + som_priors

assert len(priors) == ndim, f'''
Number of priors, ({len(priors)}), does not match number of sampled parameters,
({ndim}).
'''

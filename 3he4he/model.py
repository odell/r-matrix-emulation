'''
    R-matrix model
    Analyzing 3He(alpha, gamma) data
        * capture
        * scattering (SONIK)
'''

import numpy as np
from scipy import stats
from brick.azr import AZR

import constants as const

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
    'sonik_inflated_239.dat',
    'sonik_inflated_291.dat',
    'sonik_inflated_432.dat',
    'sonik_inflated_586.dat',
    'sonik_inflated_711.dat',
    'sonik_inflated_873_1.dat',
    'sonik_inflated_873_2.dat',
    'sonik_inflated_1196.dat',
    'sonik_inflated_1441.dat',
    'sonik_inflated_1820.dat'
]
num_pts_scatter = [np.size(np.loadtxt('data/'+f)[:, 0]) for f in scatter_data_files]
num_pts_total_scatter = sum(num_pts_scatter)

# Center-of-Mass (COM) Data
xs1_data = np.loadtxt('output/'+output_filenames[0]) # scattering data
xs2_data = np.loadtxt('output/'+output_filenames[1]) # capture XS to GS
xs3_data = np.loadtxt('output/'+output_filenames[2]) # capture XS to ES
xs_data = np.loadtxt('output/'+output_filenames[3]) # total capture

# COM Branching Ratio Data
# James sent me this setup. He used a modified version of AZURE2 that computes
# the branching ratio. So the data values in xs2_data and xs3_data both
# correspond to the measured data. No need to divide xs3 by xs2.
branching_ratio = xs2_data[:, 5]
branching_ratio_err = xs2_data[:, 6]

## COM Scattering 
def rutherford(energy, angle):
    '''
    Rutherford differential cross section
    '''
    return (4*const.ALPHA*const.HBARC / (4*energy*np.sin(angle/2*np.pi/180)**2))**2 / 100


energies_scatter = xs1_data[:, 0]
angles_scatter = xs1_data[:, 2]
scatter_rutherford = np.array(
    [rutherford(ei, ai) for (ei, ai) in zip(energies_scatter, angles_scatter)]
)
scatter = xs1_data[:, 5] / scatter_rutherford
# FIXED RELATIVE EXTRINSIC UNCERTAINTY ADDED IN "sonik_inflated_*.dat" FILES
scatter_err = xs1_data[:, 6]**2 / scatter_rutherford

# All of the energies (COM).
x = np.hstack((xs2_data[:, 0], xs_data[:, 0], xs1_data[:, 0]))
# All of the observables: branching ratios, total capture S factor, differential
# cross sections.
y = np.hstack((branching_ratio, xs_data[:, 7], scatter))
# All of the associated uncertainties reported with the data.
dy = np.hstack((branching_ratio_err, xs_data[:, 8], scatter_err))

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
    S_tot = capture_tot.sf_com_fit
    scatter_dxs = paneru.xs_com_fit / scatter_rutherford
    return np.hstack((bratio, S_tot, scatter_dxs))


def calculate_norm(theta):
    '''
    Apply normalization factors to theory predictions (total capture and
    scattering).
    '''
    fj_capture = map_uncertainty(
        theta[nrpar:nrpar+nf_capture], num_pts_capture
    )
    fj_scatter = map_uncertainty(
        theta[nrpar+nf_capture:nrpar+nf_capture+nf_scatter], num_pts_scatter
    )

    mu = azr.predict(theta)
    paneru, capture_gs, capture_es, capture_tot = mu

    bratio = capture_es.xs_com_fit/capture_gs.xs_com_fit
    S_tot = capture_tot.sf_com_fit
    scatter_dxs = paneru.xs_com_fit /scatter_rutherford
    data_norm = np.hstack((fj_capture*S_tot, fj_scatter*scatter_dxs))
    return np.hstack((bratio, data_norm))

# starting position distributions
p0_dist = [stats.norm(sp, np.abs(sp)/100) for sp in
        azr.config.get_input_values()]


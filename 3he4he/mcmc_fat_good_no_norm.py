'''
Conducts MCMC analysis using a previously trained PCGPwM emulator.
Does not sample normalization factors. Includes the inverse of the median
normalization factors from the CS analysis.
'''

import sys

import numpy as np
from scipy import stats
import emcee
import h5py
import dill as pickle

import model
import priors
from bayes import Model4

emu_filename = sys.argv[1]

n1 = model.nbr
n2 = model.nxs

with open(emu_filename, 'rb') as o:
    emu = pickle.load(o)


bayes_model = Model4(emu)

theta_star = np.load('datfiles/theta_star.npy')[:16]
nd = theta_star.size
nw = 2*nd

p0 = np.array(
    [stats.norm(theta_star, 0.01*np.abs(theta_star)).rvs() for _ in range(nw)]
)

f = emu_filename
i = f.find('/') + 1
j = f.find('.pkl')
backend_filename = f[:i] + 'backends/' + f[i:j] + '_no_norm.h5'
print(backend_filename)
backend = emcee.backends.HDFBackend(backend_filename)
backend.reset(nw, nd)
moves = [(emcee.moves.DEMove(), 0.2), (emcee.moves.DESnookerMove(), 0.8)]
sampler = emcee.EnsembleSampler(nw, nd, bayes_model.ln_posterior, moves=moves,
                                backend=backend)

state = sampler.run_mcmc(p0, 1000, thin_by=50, tune=True, progress=True)

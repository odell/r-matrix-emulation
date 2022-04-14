'''
Conducts MCMC analysis using a previously trained PCGPwM emulator.
'''

import sys

import numpy as np
from scipy import stats
import emcee
import h5py
import dill as pickle

import model
import priors
from bayes import Model

epsilon = float(sys.argv[1])
ntrain = int(sys.argv[2])
fat_frac = float(sys.argv[3])

fat_id = f'fat_{fat_frac:.1f}'
emu_id = f'good_eps_{epsilon:.4e}_ntrain_{ntrain}' + '_' + fat_id

n1 = model.nbr
n2 = model.nxs

with open('emulators/emu_' + emu_id + '.pkl', 'rb') as o:
    emu = pickle.load(o)


def ln_prior(theta):
    return np.sum([p.logpdf(t) for (p, t) in zip(priors.priors, theta)])


bayes_model = Model(emu)

theta_star = np.load('datfiles/theta_star.npy')[:-1]
nd = theta_star.size
nw = 2*nd

p0 = np.array(
    [stats.norm(theta_star, 0.01*np.abs(theta_star)).rvs() for _ in range(nw)]
)

backend = emcee.backends.HDFBackend('emulators/backends/' + emu_id + '.h5')
backend.reset(nw, nd)
moves = [(emcee.moves.DEMove(), 0.2), (emcee.moves.DESnookerMove(), 0.8)]
sampler = emcee.EnsembleSampler(nw, nd, bayes_model.ln_posterior, moves=moves,
                                backend=backend, args=[False])

state = sampler.run_mcmc(p0, 1000, thin_by=50, tune=True, progress=True)

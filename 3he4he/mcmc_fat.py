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

epsilon = float(sys.argv[1])
ntrain = int(sys.argv[2])
fat_frac = float(sys.argv[3])

fat_id = f'fat_{fat_frac:.1f}'
emu_id = f'eps_{epsilon:.4e}_ntrain_{ntrain}' + '_' + fat_id

n1 = model.nbr
n2 = model.nxs

with open('emulators/emu_' + emu_id + '.pkl', 'rb') as o:
    emu = pickle.load(o)


def ln_prior(theta):
    return np.sum([p.logpdf(t) for (p, t) in zip(priors.priors, theta)])


class Model:
    def __init__(self, emu):
        self.emu = emu
    
    
    def gp_predict(self, theta):
        p = self.emu.predict(theta=theta)
        mu = p.mean().T[0]
        var = p.var().T[0]
        return mu, var
    
    
    def ln_likelihood(self, theta):
        theta_R = theta[:model.nrpar]
        theta_f_capture = theta[model.nrpar:model.nrpar+model.nf_capture]
        theta_f_scatter = theta[model.nrpar+model.nf_capture:]
        
        f_capture = model.map_uncertainty(theta_f_capture, model.num_pts_capture)
        f_scatter = model.map_uncertainty(theta_f_scatter, model.num_pts_scatter)
        
        f = np.ones(model.y.size)
        f[n1:n1+n2] = f_capture
        f[n1+n2:] = f_scatter
        
        mu, var = self.gp_predict(theta_R)
        var_tot = (f*model.dy)**2 + var
        var_tot_no_norm = model.dy**2 + var

        return np.sum(-np.log(np.sqrt(2*np.pi*var_tot_no_norm)) - \
            0.5*(f*model.y - mu)**2 / var_tot) 


    def ln_posterior(self, theta):
        lnpi = ln_prior(theta)
        if lnpi == -np.inf:
            return -np.inf
        return lnpi + self.ln_likelihood(theta)


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
                                backend=backend)

state = sampler.run_mcmc(p0, 1000, thin_by=50, tune=True, progress=True)

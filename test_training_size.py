#!/usr/bin/env python
# coding: utf-8

# # MCMC Comparison
# 
# ## surmise
# 
# 1. Train a GP (using surmise) to emulate the $S$-factor. 
# 2. Use that GP to calculate the $R$-matrix parameter posteriors.
# 3. Calculate the $R$-matrix parameter posteriors using the $R$-matrix prediction *directly*.
# 
# * range of energies
# * use parameter samples near the true values *exclusively*
# * use increasingly wider distributions to generate training samples
#     * If we use flat distributions, the results are crap. With narrow (10%) Gaussians, the results are significantly better. As the Gaussians get wider, we expect to recover our original bad results. Where does this happen?
# * use fewer training points and see how things change

import os
import sys
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import seaborn as sns
from scipy import stats
from surmise.emulation import emulator
import emcee
import h5py
from corner import corner

from dt import cross_section, s_factor
import constants as const

rel_unc = float(sys.argv[1])
nt = int(sys.argv[2])
ntrain = 250

plt.style.use('science')
gr = (1 + np.sqrt(5)) / 2
h = 3
plt.rcParams['figure.dpi'] = 300


# Our physics model.
def f(energy, theta):
    er, gd2, gn2 = theta
    return s_factor(energy, er, er, gd2, gn2, const.AD, const.AN, const.UE, const.A)

parameter_labels = [r'$E_r$', r'$\gamma_d^2$', r'$\gamma_n^2$']

train = np.loadtxt(f'datfiles/better_training_data_{rel_unc:.2f}.txt')
test = np.loadtxt(f'datfiles/better_testing_data_{rel_unc:.2f}.txt')

ns, nd = train.shape
nk = np.unique(train[:, 0]).size # number of momenta

x = train[::ntrain, 0].reshape(-1, 1) # input/location/momentum
w = train[:nt, 1:4] # parameters that we want to emulate

y = train[:, -1].reshape(nk, ntrain) # output/cross section
y = y[:, :nt]


# Set up the surmise emulator.
args = {'epsilon' : 0.0, 'warnings' : True}
# args = {'epsilon': 0.1, ‘hypregmean’: -8}
emu = emulator(x=x, theta=w, f=y, method='PCGPwM', args=args)

bounds = np.copy(const.BOUNDS)
priors = [stats.uniform(b[0], b[1]-b[0]) for b in bounds]

energies, data, data_err = np.loadtxt('datfiles/data.txt', unpack=True)
theta_true = np.loadtxt('datfiles/theta_true.txt')[:3]

def ln_prior(theta):
    return np.sum([p.logpdf(t) for (p, t) in zip(priors, theta)])


class Model:
    def __init__(self, predict):
        self.predict_func = predict
    
    
    def ln_likelihood(self, theta):
        mu, var = self.predict_func(theta)
        var_tot = data_err**2 + var
        return np.sum(-np.log(np.sqrt(2*np.pi*var_tot)) - (data - mu)**2 / var_tot)


    def ln_posterior(self, theta):
        lnpi = ln_prior(theta)
        if lnpi == -np.inf:
            return -np.inf
        return lnpi + self.ln_likelihood(theta)


def gp_func(theta):
    p = emu.predict(theta=theta)
    mu = p.mean().T[0]
    var = p.var().T[0]
    return mu, var


def r_func(theta):
    return np.array([f(ei, theta) for ei in energies]), np.zeros(energies.size)


model_gp = Model(gp_func)
model_r = Model(r_func)


nd = 3
nw = 2*nd

p0 = np.array(
    [stats.norm(theta_true, 0.01*np.abs(theta_true)).rvs() for _ in range(nw)]
)

moves = [(emcee.moves.DEMove(), 0.2), (emcee.moves.DESnookerMove(), 0.8)]
sampler_gp = emcee.EnsembleSampler(nw, nd, model_gp.ln_posterior, moves=moves)
sampler_r = emcee.EnsembleSampler(nw, nd, model_r.ln_posterior, moves=moves)


state = sampler_gp.run_mcmc(p0, 1000, thin_by=10, tune=True, progress=True)
state = sampler_r.run_mcmc(p0, 1000, thin_by=10, tune=True, progress=True)

fig, ax = plt.subplots(ncols=2, figsize=(2*gr*h, h))
fig.patch.set_facecolor('white')

nb_gp = 100
nb_r = nb_gp

ax[0].plot(sampler_gp.get_log_prob(discard=nb_gp))
ax[1].plot(sampler_r.get_log_prob(discard=nb_r));


flat_chain_gp = sampler_gp.get_chain(flat=True, discard=nb_gp)
flat_chain_r = sampler_r.get_chain(flat=True, discard=nb_r)

# Corner plots

fig = corner(flat_chain_gp, labels=parameter_labels, show_titles=True, quantiles=[0.16, 0.5, 0.84], color='C0')
fig = corner(flat_chain_r, color='C1', fig=fig, truths=theta_true, truth_color='C3')
# fig = corner(w, color='C2', fig=fig)
fig.patch.set_facecolor('white')
plt.savefig(f'figures/corner_{rel_unc:.2f}_{nt}.pdf')


# Chains
fig, ax = plt.subplots(3, figsize=(gr*h, 3*h))
fig.patch.set_facecolor('white')

for i in range(3):
    ax[i].plot(flat_chain_gp[:, i], label='GP')
    ax[i].plot(flat_chain_r[:, i], label='R-matrix')
#     ax[i].plot(w[:, i], label='Train')
    low_68, high_68 = np.quantile(w[:, i], [0.16, 0.84])
    low_95, high_95 = np.quantile(w[:, i], [0.05, 0.95])
    ax[i].fill_between(np.arange(flat_chain_gp[:, i].size), low_68, high_68, color='C2', alpha=0.5)
    ax[i].fill_between(np.arange(flat_chain_gp[:, i].size), low_95, high_95, color='C2', alpha=0.25)
    ax[i].set_ylabel(parameter_labels[i])
    ax[i].axhline(theta_true[i], color='C3', label='True')
    ax[i].legend()

plt.savefig(f'figures/chains_{rel_unc:.2f}_{nt}.pdf')

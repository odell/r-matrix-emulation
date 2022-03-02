import sys
import os
import subprocess
from smt.sampling_methods import LHS

import numpy as np
from scipy import stats
from surmise.emulation import emulator

from dt import cross_section, s_factor

# Reduced mass in the deuteron channel.
MU_D = 1124.6473494927284

rel_unc = float(sys.argv[1])
SKEW = 1.25

indices = np.array([0, 1, 2])

energies = np.linspace(0.010, 0.200, 10)
momenta = np.sqrt(2*MU_D*energies)
n = 1 # output dimension
num_pars = indices.size

# Default values.
AD = 6.0
AN = 4.0
UE = 0.0
A = 0.0
GD2 = 3.0
GN2 = 0.5
ER = 0.070
DEFAULT_VALUES = np.array([ER, GD2, GN2, AD, AN, UE, A])

BOUNDS = np.array([[0.010, 0.120],
                   [1, 5],
                   [0.001, 0.5]])

NTRAIN = 250 # number of training points
NTEST = 50 # number of testing points

class RMatrixModel:
    def __init__(self, parameter_indices):
        self.parameter_indices = parameter_indices


    def s_factor(self, energy, theta):
        return s_factor(energy, theta[0], theta[0], *theta[1:])


    def evaluate(self, energy, theta):
        thetap = np.copy(DEFAULT_VALUES)
        thetap[self.parameter_indices] = theta
        return self.s_factor(energy, thetap)


model = RMatrixModel(indices)
bounds = BOUNDS[indices, :]

# Add more samples near the true values.
def my_truncnorm(mu, sigma, lower, upper):
    return stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)


def my_truncskewnorm(mu, sigma, lower, upper):
    return stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)


theta_true = np.loadtxt('datfiles/theta_true.txt')[:3]

def random_skew_norm_sample(i):
    mu = theta_true[i]
    sigma = rel_unc*mu
    upper_limit = bounds[i, -1]
    d = stats.skewnorm(SKEW, loc=mu, scale=sigma)
    sample = d.rvs()
    while sample < 0:
        sample = d.rvs()
    return sample

    
dists = [my_truncnorm(mu, rel_unc*mu, 0, bounds[i, -1]) for (i, mu) in enumerate(theta_true)]
skewed_samples = np.array([[random_skew_norm_sample(i) for i in range(num_pars)] for _ in range(NTRAIN)])

train_skewed = np.array([
    [k, *theta, model.evaluate(0.5*k**2/MU_D, theta)] for k in momenta for theta in skewed_samples
])

np.savetxt(rf'datfiles/skewed_training_data_{rel_unc:.2f}.txt', train_skewed, header='''
Momentum (MeV) | E_r (MeV) | gamma_d^2 (MeV) | gamma_n^2 (MeV) | S factor (MeV b)
''')

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

# 0 = er
# 1 = gamma_d^2
# 2 = gamma_n^2
indices = np.array(sys.argv[1:], dtype=np.int32)

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

NTRAIN = 500 # number of training points
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

# Set up Latin hypercube sampling to generate the training/testing space.
generator = LHS(xlimits=bounds)

# Convenience function for generating a matrix of data.
def generate_data(m):
    '''
    Generates a matrix of data. Organized according to:
    momentum | theta_0 | theta_1 | theta_2 | y
    '''
    theta_space = generator(m)
    return np.array([
        [k, *theta, model.evaluate(0.5*k**2/MU_D, theta)] for k in momenta for theta in theta_space
    ])


# Generating data.
train = generate_data(NTRAIN)
test = generate_data(NTEST)

np.savetxt('datfiles/training_data.txt', train, header='''
Momentum (MeV) | E_r (MeV) | gamma_d^2 (MeV) | gamma_n^2 (MeV) | S factor (MeV b)
''')

np.savetxt('datfiles/testing_data.txt', test, header='''
Momentum (MeV) | E_r (MeV) | gamma_d^2 (MeV) | gamma_n^2 (MeV) | S factor (MeV b)
''')

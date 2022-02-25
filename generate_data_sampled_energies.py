import sys
import os
import subprocess
from smt.sampling_methods import LHS

import numpy as np
from scipy import stats
from surmise.emulation import emulator

from dt import cross_section, s_factor

# Reduced mass in the deuteron channel.
MU_D = 1124.6473494927284 # MeV

indices = np.array([0, 1, 2])

n = 1 # output dimension

# Default values.
AD = 6.0
AN = 4.0
UE = 0.0
A = 0.0
GD2 = 3.0
GN2 = 0.5
ER = 0.070
DEFAULT_VALUES = np.array([ER, GD2, GN2, AD, AN, UE, A])
E_MIN = 0.010
E_MAX = 0.200
K_MIN = np.sqrt(2*MU_D*E_MIN)
K_MAX = np.sqrt(2*MU_D*E_MAX)

BOUNDS = np.array([[K_MIN, K_MAX],
                   [0.010, 0.120],
                   [1, 5],
                   [0.001, 0.5]])

NTRAIN = 500 # number of training points
NTEST = 50 # number of testing points

class RMatrixModel:
    def __init__(self, parameter_indices):
        self.parameter_indices = np.copy(parameter_indices)


    def s_factor(self, energy, theta):
        return s_factor(energy, theta[0], theta[0], *theta[1:])


    def evaluate(self, energy, theta):
        thetap = np.copy(DEFAULT_VALUES)
        thetap[self.parameter_indices] = theta
        return self.s_factor(energy, thetap)


model = RMatrixModel(indices)

# Set up Latin hypercube sampling to generate the training/testing space.
generator = LHS(xlimits=BOUNDS)

# Convenience function for generating a matrix of data.
def generate_data(m):
    '''
    Generates a matrix of data. Organized according to:
    momentum | theta_0 | theta_1 | theta_2 | y
    '''
    theta_space = generator(m)
    return np.array([
        [*theta, model.evaluate(0.5*theta[0]**2/MU_D, theta[1:])] for theta in theta_space
    ])


# Generating data.
train = generate_data(NTRAIN)
test = generate_data(NTEST)

np.savetxt('datfiles/training_data_sampled_energies.txt', train, header='''
Momentum (MeV) | E_r (MeV) | gamma_d^2 (MeV) | gamma_n^2 (MeV) | S factor (MeV b)
''')

np.savetxt('datfiles/testing_data_sampled_energies.txt', test, header='''
Momentum (MeV) | E_r (MeV) | gamma_d^2 (MeV) | gamma_n^2 (MeV) | S factor (MeV b)
''')

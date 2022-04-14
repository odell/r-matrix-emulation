'''
Trains and save a PCGPwM emulator based on the number of PCs (controlled via
epsilon) and number of training points (which are obtained via a greedy
algorithm).
'''

import sys

import numpy as np
import seaborn as sns
from surmise.emulation import emulator
import dill as pickle

import model

NRPAR = 16

# Command-line arguments
epsilon = float(sys.argv[1])
ntrain = int(sys.argv[2])
fat_frac = float(sys.argv[3])

fat_id = f'fat_{fat_frac:.1f}'
emu_id = f'eps_{epsilon:.4e}_ntrain_{ntrain}' + '_' + fat_id

DESIGN_POINTS = np.load('datfiles/' + fat_id + '_posterior_samples.npy')
INPUT_POINTS = np.load('datfiles/' + fat_id + '_posterior_chain.npy')

ntest = DESIGN_POINTS.shape[0] - ntrain

f = DESIGN_POINTS[:ntrain, :].T
w = INPUT_POINTS[:ntrain, :NRPAR]

# Set up the surmise emulator.
args = {'epsilon' : epsilon, 'warnings' : True}
emu = emulator(x=model.x, theta=w, f=f, method='PCGPwM', args=args)

with open('emulators/emu_' + emu_id + '.pkl', 'wb') as o:
    pickle.dump(emu, o, pickle.HIGHEST_PROTOCOL)


def residuals(y, mu, dy):
    return (y - mu) / dy


def relative_uncertainties(p):
    return (np.sqrt(p.var()) / p.mean())[0]


def absolute_uncertainties(p):
    return np.sqrt(p.var())[0]


ptrain = emu.predict()

fp = design_points[ntrain:, :]
wp = input_points[ntrain:, :NRPAR]

p = emu.predict(x=model.x, theta=wp)

def pop(array, i):
    '''
    Gets ith element from array.
    Removes ith element from array.
    Return ith element and the new array.
    '''
    x = array[i]
    array = np.delete(array, i)
    return x, array


class GreedyModel:
    def __init__(self, epsilon, ntrain, ntrain_0=10, j_size=5):
        self.epsilon = epsilon
        self.m = ntrain
        self.m0 = ntrain_0
        self.nj = j_size


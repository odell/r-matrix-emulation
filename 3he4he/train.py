'''
Trains and save a PCGPwM emulator based on the number of PCs (controlled via
epsilon) and number of training points.
'''

import sys
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surmise.emulation import emulator
import dill as pickle

import model

# Command-line arguments
epsilon = float(sys.argv[1])
ntrain = int(sys.argv[2])
emu_id = f'eps_{epsilon:.2f}_ntrain_{ntrain}'

plt.style.use('science')
gr = 4/3
h = 3

design_points = np.load('datfiles/posterior_samples.npy')
input_points = np.load('datfiles/posterior_chain.npy')

nrpar = 16
ntest = design_points.shape[0] - ntrain

f = design_points[:ntrain, :].T
w = input_points[:ntrain, :nrpar]

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
fig, ax = plt.subplots(ncols=3, figsize=(3*h, h), dpi=300)
fig.patch.set_facecolor('white')

sns.distplot(residuals(f, ptrain.mean(), np.sqrt(ptrain.var())).flatten(),
             bins=40, ax=ax[0], kde=True)
ax[0].set_xlabel(r'$(y-\mu)/\sigma$')
ax[0].set_title('Train')

sns.distplot(relative_uncertainties(ptrain).flatten(), bins=40, ax=ax[1], kde=True)
ax[1].set_xlabel(r'Relative 1-$\sigma$ Uncertainties (Train)')

sns.distplot(((f - ptrain.mean()) / f).flatten(), bins=40, ax=ax[2], kde=True)
ax[2].set_xlabel(r'$(y-\mu)/y$')

plt.tight_layout()
plt.savefig('emulators/figures/residuals_train_' + emu_id + '.pdf')

fp = design_points[ntrain:, :]
wp = input_points[ntrain:, :nrpar]

p = emu.predict(x=model.x, theta=wp)

fig, ax = plt.subplots(ncols=3, figsize=(3*h, h), dpi=300)
fig.patch.set_facecolor('white')

sns.distplot(residuals(fp.T, p.mean(), np.sqrt(p.var())).flatten(), bins=40, ax=ax[0], kde=True)
ax[0].set_xlabel(r'$(y-\mu)/\sigma$')
ax[0].set_title('Test')

sns.distplot(relative_uncertainties(p).flatten(), bins=40, ax=ax[1], kde=True)
ax[1].set_xlabel(r'Relative 1-$\sigma$ Uncertainties (Test)')

sns.distplot(((fp.T - p.mean()) / fp.T).flatten(), bins=40, ax=ax[2], kde=True)
ax[2].set_xlabel(r'$(y-\mu)/y$')

plt.tight_layout()
plt.savefig('emulators/figures/residuals_test_' + emu_id + '.pdf')

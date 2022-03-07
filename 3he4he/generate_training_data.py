import pickle
import sys
import numpy as np
from scipy import stats
import os
from multiprocessing import Pool

import model as model

sys.path.append('/home/odell/7Be')
import run

os.environ['OMP_NUM_THREADS'] = '1'

with open('/spare/odell/7Be/CP/samples/model_1_2021-08-06-02-55-37.pkl', 'rb') as f:
    run = pickle.load(f)

nrpar = model.azr.config.nd
theta_star = run.get_theta_star(max_likelihood=True)[:nrpar]
sigma = 0.2*np.abs(theta_star)

d = stats.norm(theta_star, sigma)
points = np.array([d.rvs() for _ in range(1000)])


with Pool(processes=16) as pool:
    train = pool.map(model.calculate, points)

np.save('datfiles/training_points.npy', points)
np.save('datfiles/training_data.npy', np.array(train))
np.save('datfiles/theta_star.npy', theta_star)

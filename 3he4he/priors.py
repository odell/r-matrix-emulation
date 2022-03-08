'''
Defines the prior distributions associated with the sampled R-matrix
parameters and normalization factors.
'''

import numpy as np
from scipy import stats

import constants as const

def my_truncnorm(mu, sigma, lower, upper):
    '''
    My version of a truncated normal distribution that actually makes sense.
    '''
    return stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)


# priors
anc429_prior = stats.uniform(1, 4)
Ga_1hm_prior = stats.uniform(-200e6, 400e6)

Ga_1hp_prior = stats.uniform(0, 100e6)
Gg0_1hp_prior = stats.uniform(0, 10e6)
Gg429_1hp_prior = stats.uniform(-10e3, 20e3)

anc0_prior = stats.uniform(1, 4)
Ga_3hm_prior = stats.uniform(-100e6, 200e6)

Ga_3hp_prior = stats.uniform(0, 100e6)
Gg0_3hp_prior = stats.uniform(-10e3, 20e3)
Gg429_3hp_prior = stats.uniform(-3e3, 6e3)

Ga_5hm_prior = stats.uniform(0, 100e6)
Ga_5hp_prior = stats.uniform(0, 100e6)
Gg0_5hp_prior = stats.uniform(-100e6, 200e6)

Ex_7hm_prior = stats.uniform(1, 9)
Ga_7hm_prior = stats.uniform(0, 10e6)
Gg0_7hm_prior = stats.uniform(0, 1e3)

f_seattle_prior = my_truncnorm(1, 0.03, 0, np.inf)
f_weizmann_prior = my_truncnorm(1, 0.037, 0, np.inf)
f_luna_prior = my_truncnorm(1, 0.032, 0, np.inf)
f_erna_prior = my_truncnorm(1, 0.05, 0, np.inf)
f_nd_prior = my_truncnorm(1, 0.08, 0, np.inf)
f_atomki_prior = my_truncnorm(1, 0.06, 0, np.inf)

# Add 1% (in quadrature) to systematic uncertainty to account for beam-position
# uncertainty.
som_syst = np.sqrt(const.som_syst**2 + 0.01**2)
som_priors = [my_truncnorm(1, syst, 0, np.inf) for syst in som_syst]

priors = [
    anc429_prior,
    Ga_1hm_prior,
    Ga_1hp_prior,
    Gg0_1hp_prior,
    Gg429_1hp_prior,
    anc0_prior,
    Ga_3hm_prior,
    Ga_3hp_prior,
    Gg0_3hp_prior,
    Gg429_3hp_prior,
    Ga_5hm_prior,
    Ga_5hp_prior,
    Gg0_5hp_prior,
    Ex_7hm_prior,
    Ga_7hm_prior,
    Gg0_7hm_prior,
    f_seattle_prior,
    f_weizmann_prior,
    f_luna_prior,
    f_erna_prior,
    f_nd_prior,
    f_atomki_prior
] + som_priors

import numpy as np
from scipy import stats

import model
import priors

n1 = model.nbr
n2 = model.nxs

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
    
    
    def normalization_factors(self, theta):
        theta_f_capture = theta[model.nrpar:model.nrpar+model.nf_capture]
        theta_f_scatter = theta[model.nrpar+model.nf_capture:]

        f_capture = model.map_uncertainty(theta_f_capture, model.num_pts_capture)
        f_scatter = model.map_uncertainty(theta_f_scatter, model.num_pts_scatter)

        f = np.ones(model.y.size)
        f[n1:n1+n2] = f_capture
        f[n1+n2:] = f_scatter
        
        return f


    def likelihood_normalization(self, var):
        return np.sum(-np.log(np.sqrt(2*np.pi*var)))


    def chi_squared(self, f, mu, var):
        return np.sum(-0.5*(f*model.y - mu)**2 / var)


    def ln_likelihood(self, theta, include_gp_var=True):
        theta_R = theta[:model.nrpar]

        f = self.normalization_factors(theta)
        
        mu, var = self.gp_predict(theta_R)

        factor = self.likelihood_normalization(
            model.dy**2 + (var if include_gp_var else 0)
        )
        chisq = self.chi_squared(f, mu,
            (f*model.dy)**2 + (var if include_gp_var else 0)
        )

        return factor + chisq


    def ln_posterior(self, theta, include_gp_var=True):
        lnpi = ln_prior(theta)
        if lnpi == -np.inf:
            return -np.inf
        return lnpi + self.ln_likelihood(theta, include_gp_var=include_gp_var)


class Model1(Model):
    '''
    Requires that the GP variance is smaller than the experimental uncertainty.
    '''
    def ln_likelihood(self, theta, include_gp_var=True):
        theta_R = theta[:model.nrpar]
        mu, var = self.gp_predict(theta_R)

        if np.sum(np.sqrt(var) - model.dy) > 0:
            return -np.inf

        f = self.normalization_factors(theta)

        factor = self.likelihood_normalization(
            model.dy**2 + (var if include_gp_var else 0)
        )
        chisq = self.chi_squared(f, mu,
            (f*model.dy)**2 + (var if include_gp_var else 0)
        )

        return factor + chisq


GP_SIGMA_PRIORS = [stats.norm(0, sigma) for sigma in model.dy]

def ln_prior_gp_var(sigma_gp):
    return np.sum([pi.logpdf(theta) for (pi, theta) in zip(GP_SIGMA_PRIORS,
        sigma_gp)])


class Model2(Model):
    '''
    Sets priors on the GP variances at each input location.
    '''
    def ln_likelihood(self, theta, include_gp_var=True):
        theta_R = theta[:model.nrpar]
        mu, var = self.gp_predict(theta_R)

        gp_var_prior = ln_prior_gp_var(np.sqrt(var))

        f = self.normalization_factors(theta)

        factor = self.likelihood_normalization(
            model.dy**2 + (var if include_gp_var else 0)
        )
        chisq = self.chi_squared(f, mu,
            (f*model.dy)**2 + (var if include_gp_var else 0)
        )

        return factor + chisq + gp_var_prior

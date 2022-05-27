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


class Model3(Model):
    '''
    Uses the full covariance matrix from the emulator, not just the diagonal.
    '''
    def gp_predict(self, theta):
        p = self.emu.predict(theta=theta)
        mu = p.mean().T[0]
        cov = p.covx()[:, 0, :]
        return mu, cov
    
    
    def likelihood_normalization(self, cov):
        return -np.log(np.sqrt(2*np.pi)) - 0.5*np.linalg.slogdet(cov)[1]


    def chi_squared(self, f, mu, cov):
        diff = f*model.y - mu
        return -0.5 * (diff @ np.linalg.inv(cov) @ diff)


    def ln_likelihood(self, theta, include_gp_cov=True):
        theta_R = theta[:model.nrpar]

        f = self.normalization_factors(theta)
        
        mu, cov = self.gp_predict(theta_R)

        factor = self.likelihood_normalization(
            np.diag(model.dy**2) + (cov if include_gp_cov else 0)
        )
        chisq = self.chi_squared(f, mu,
            np.diag((f*model.dy)**2) + (cov if include_gp_cov else 0)
        )

        return factor + chisq


    def ln_posterior(self, theta, include_gp_cov=True):
        lnpi = ln_prior(theta)
        if lnpi == -np.inf:
            return -np.inf
        return lnpi + self.ln_likelihood(theta, include_gp_cov=include_gp_cov)


import pickle

def chi_squared(y, mu, cov):
    diff = y - mu
    return -0.5 * (diff @ np.linalg.inv(cov) @ diff)


class Model4(Model3):
    '''
    Modifies the data by applying the inverse of the median normalization
    factors obtained from the CS analysis.
    Does *not* sample normalization factors.
    '''
    def __init__(self, emu):
        self.emu = emu
 
        with open('/spare/odell/7Be/CP/samples/model_1_2021-08-06-02-55-37.pkl', 'rb') as f:
            run = pickle.load(f)
        cs_flat_chain = run.get_flat_chain()
        f_cs = np.median(cs_flat_chain[:, 16:32], axis=0)
        inv_f = 1/self.normalization_factors(np.hstack((np.ones(16), f_cs)))
        self.y = inv_f * model.y
        self.dy = np.copy(model.dy)
        self.fdy = inv_f * model.dy


    def ln_likelihood(self, theta, include_gp_cov=True):
        mu, cov = self.gp_predict(theta)
        factor = self.likelihood_normalization(
            np.diag(self.dy**2) + (cov if include_gp_cov else 0)
        )
        chisq = chi_squared(self.y, mu,
            np.diag(self.fdy**2) + (cov if include_gp_cov else 0)
        )
        return factor + chisq


    def ln_posterior(self, theta, include_gp_cov=True):
        lnpi = ln_prior(theta)
        if lnpi == -np.inf:
            return -np.inf
        return lnpi + self.ln_likelihood(theta, include_gp_cov=include_gp_cov)


class Model5(Model4):
    '''
    Uses priors that represent the CS posterior on which the emulator was
    trained.
    '''
    design_chain = np.load('datfiles/fat_0.3_posterior_chain.npy')
    priors = [stats.gaussian_kde(x) for x in design_chain.T]

    def ln_prior(self, theta):
        return np.sum([pi.logpdf(x) for (pi, x) in zip(self.priors, theta)])


    def ln_posterior(self, theta):
        lnpi = self.ln_prior(theta)
        if np.isinf(lnpi):
            return -np.inf
        else:
            lnl = self.ln_likelihood(theta)
            if np.isnan(lnl) or np.isinf(lnl):
                return -np.inf
            else:
                return lnpi + lnl

import numpy as np

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


    def ln_posterior(self, theta):
        lnpi = ln_prior(theta)
        if lnpi == -np.inf:
            return -np.inf
        return lnpi + self.ln_likelihood(theta)
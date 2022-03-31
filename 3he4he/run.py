import emcee
import numpy as np
import pickle

class Run:
    '''
    Data structure to store relevant information about a run after it has been
    completed and "harvested".
    '''
    def __init__(self, h5_filename, h5_name, n_burnin, n_thin, good_walkers):
        self.h5_filename = h5_filename
        self.h5_name = h5_name
        self.nb = n_burnin
        self.nt = n_thin
        self.gw = good_walkers
        

    def get_chain(self):
        backend = emcee.backends.HDFBackend(
            self.h5_filename,
            name=self.h5_name
        )
        return backend.get_chain(discard=self.nb, thin=self.nt)[:, self.gw, :]


    def get_flat_chain(self):
        chain = self.get_chain()
        _, _, nd = chain.shape
        return chain.reshape(-1, nd)


    def get_theta_star(self, max_likelihood=True):
        backend = emcee.backends.HDFBackend(
            self.h5_filename,
            name=self.h5_name
        )
        if max_likelihood:
            return backend.get_chain(flat=True)[np.argmax(backend.get_blobs(flat=True)), :]
        else:
            return backend.get_chain(flat=True)[np.argmax(backend.get_log_prob(flat=True)), :]


    def save(self):
        filename = self.h5_filename[:-3] + '.pkl'
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

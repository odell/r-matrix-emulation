import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def rutherford(energy, angle):
    return (4*ALPHA*HBARC / (4*energy*np.sin(angle/2*np.pi/180)**2))**2 / 100


ALPHA = 0.0072973525693
HBARC = 197.32696310 # MeV•fm

SONIK_DATA_FILES = [
    'data/sonik_inflated_239.dat',
    'data/sonik_inflated_291.dat',
    'data/sonik_inflated_432.dat',
    'data/sonik_inflated_586.dat',
    'data/sonik_inflated_711.dat',
    'data/sonik_inflated_873_1.dat',
    'data/sonik_inflated_873_2.dat',
    'data/sonik_inflated_1196.dat',
    'data/sonik_inflated_1441.dat',
    'data/sonik_inflated_1820.dat'
]
NS_SONIK = [np.loadtxt(f).shape[0] for f in SONIK_DATA_FILES]
STOP = np.cumsum(NS_SONIK)
START = STOP - NS_SONIK

SONIK_LABELS = [
    r'$239$',
    r'$291$',
    r'$432$',
    r'$586$',
    r'$711$',
    r'$873$',
    r'$1196$',
    r'$1441$',
    r'$1820$'
]

SONIK_DATA = np.loadtxt('output/AZUREOut_aa=1_R=1.out')
ENERGIES = SONIK_DATA[:, 0]
UNIQUE_ENERGIES = np.unique(ENERGIES)
IE = np.array([np.where(ENERGIES == energy)[0] for energy in UNIQUE_ENERGIES])
IE = IE.reshape(-1, 3)
ANGLES = SONIK_DATA[:, 2]
DSDO = SONIK_DATA[:, 5]
DSDO_ERR = SONIK_DATA[:, 6]
SONIK_RUTHERFORD = np.array(
    [rutherford(en, theta) for (en, theta) in zip(ENERGIES, ANGLES)]
)

GR = 4/3
H = 3
def plot_sonik_data(rel_ruth=True, annotate=True):
    fig, ax = plt.subplots(3, ncols=3, figsize=(3*GR*H, 3*H))
    fig.patch.set_facecolor('white')

    for (i, energy_range) in enumerate(IE):
        for (j, k) in enumerate(energy_range):
            if rel_ruth:
                ruth = SONIK_RUTHERFORD[k]
            else:
                ruth = np.ones(len(k))
            ax[i//3, i%3].errorbar(ANGLES[k], DSDO[k]/ruth,
                                   yerr=DSDO_ERR[k]/ruth,
                                   capsize=2, linewidth=0.5, linestyle='',
                                   color='C6', marker='x', markersize=3)
            if annotate:
                ax[i//3, i%3].annotate(rf'{SONIK_LABELS[i]} keV/u', (0.1, 0.85),
                        xycoords='axes fraction')
    return fig


def plot_gp_prediction(dsdo, dsdo_err, fig=None, annotate=True):
    if fig is None:
        fig = plot_sonik_data(rel_ruth=True, annotate=annotate)
    ax = np.array(fig.get_axes()).reshape(3, 3)
    for (i, energy_range) in enumerate(IE):
        for (j, k) in enumerate(energy_range):
            x = ANGLES[k]
            ia = np.argsort(x)
            mu = dsdo[k]
            mu_err = dsdo_err[k]
            x = x[ia]
            mu = mu[ia]
            mu_err = mu_err[ia]
            # print(x, mu, mu_err, i, k, ia)
            ax[i//3, i%3].errorbar(x, mu, yerr=mu_err, linestyle='', capsize=2,
                    color='C2', marker='x', alpha=0.5)
            
def plot_azure2_prediction(dsdo, fig=None, annotate=True):
    if fig is None:
        fig = plot_sonik_data(rel_ruth=True, annotate=annotate)
    ax = np.array(fig.get_axes()).reshape(3, 3)
    for (i, energy_range) in enumerate(IE):
        for (j, k) in enumerate(energy_range):
            x = ANGLES[k]
            ia = np.argsort(x)
            mu = dsdo[k]
            x = x[ia]
            mu = mu[ia]
            # print(x, mu, mu_err, i, k, ia)
            ax[i//3, i%3].scatter(x, mu, color='C1', alpha=0.5)

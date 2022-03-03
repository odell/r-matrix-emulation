import numpy as np

MEV_AMU = 931.4940880 # MeV/u
MASS_HE3 = 3.01603*MEV_AMU # MeV
MASS_HE4 = 4.0026*MEV_AMU # MeV
MU = MASS_HE3*MASS_HE4 / (MASS_HE3 + MASS_HE4) # MeV
HBARC = 197.32696310 # MeV•fm
M_SQRT2PI = np.sqrt(2*np.pi)

capture_syst = [
    0.03,
    0.037,
    0.032,
    0.05,
    0.08,
    0.06
]

som_syst_1820 = 0.089
som_syst_1441 = 0.063
som_syst_1196 = 0.077
som_syst_873_2 = 0.041
som_syst_711 = 0.045
som_syst_586 = 0.057
som_syst_432 = 0.098
som_syst_291 = 0.076
som_syst_239 = 0.064
som_syst_873_1 = 0.062

som_syst = np.array([
    som_syst_1820,
    som_syst_1441,
    som_syst_1196,
    som_syst_873_2,
    som_syst_711,
    som_syst_586,
    som_syst_432,
    som_syst_291,
    som_syst_239,
    som_syst_873_1
])


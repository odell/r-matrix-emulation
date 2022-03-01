import numpy as np

# Other parameters that we are not trying to emulate yet.
AD = 6.0
AN = 4.0
UE = 0.0
A = 0.0
# Reduced mass in the deuteron channel.
MU_D = 1124.6473494927284 # MeV

BOUNDS = np.array([[0.010, 0.120],
                   [1, 5],
                   [0.001, 0.5]])

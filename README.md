# r-matrix-emulation
Attempts to emulate R-matrix calculations to speed up MCMC analyses.

## t(d,n)a

This reaction is "fairly simple".
There is a single, broad resonance at low energies which is readily described
with a single R-matrix level.
More importantly, I already have codes to calculate the cross section / S
factor.
These codes are written in C (`r-matrix-simple.c`), compiled to a shared library
(`libsfac.so`), and interfaced with Python (dt.py).
The `makefile` simplifies things, requiring only: `make libs`.
Access to the C functions is then available via `import dt`.

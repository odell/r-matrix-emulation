from ctypes import cdll, c_int, c_double, POINTER, byref

LIB_SFAC = cdll.LoadLibrary('libsfac.so')

s_factor = LIB_SFAC.S_factor
s_factor.argtypes = (c_double, c_double, c_double, c_double, c_double,\
        c_double, c_double, c_double, c_double)
s_factor.restype = c_double

cross_section = LIB_SFAC.cross_section
cross_section.argtypes = (c_double, c_double, c_double, c_double, c_double,\
        c_double, c_double, c_double, c_double)
cross_section.restype = c_double

S_factor_unitary_limit = LIB_SFAC.S_factor_unitary_limit
S_factor_unitary_limit.argtypes = (c_double, c_double)
S_factor_unitary_limit.restype = c_double

Sdn2 = LIB_SFAC.Sdn2
Sdn2.argtypes = (c_double, c_double, c_double, c_double, c_double, c_double,
        c_double, c_double, c_double, c_double, c_double, c_double)
Sdn2.restype = c_double

shift_factor = LIB_SFAC.shift_factor
shift_factor.argtypes = (c_int, c_double, c_double)
shift_factor.restype = c_double

sommerfeld = LIB_SFAC.sommerfeld
sommerfeld.argtypes = (c_double, c_double)
sommerfeld.restype = c_double

Gamma_c = LIB_SFAC.Gamma_c
Gamma_c.argtypes = (c_double, c_int, c_double, c_double)
Gamma_c.restype = c_double

Delta_c = LIB_SFAC.Delta_c
Delta_c.argtypes = (c_double, c_double, c_double, c_double, c_int, c_double)
Delta_c.restype = c_double

penetration_factor = LIB_SFAC.penetration_factor
penetration_factor.argtypes = (c_int, c_double, c_double)
penetration_factor.restype = c_double

shift_factor = LIB_SFAC.shift_factor
shift_factor.argtypes = (c_int, c_double, c_double)
shift_factor.restype = c_double

cf = LIB_SFAC.coulomb_functions
cf.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_double, c_double)
cf.restype = None

def coulomb_functions(l, eta, x):
    f = c_double()
    fp = c_double()
    g = c_double()
    gp = c_double()
    cf(byref(f), byref(fp), byref(g), byref(gp), c_int(l), c_double(eta), c_double(x))
    return (f.value, fp.value, g.value, gp.value)

import numpy as np

from .DSTfortran.fortranloops import dstloopf



def calc_ab_rungekutta4_flib( dx, L, q, zetas):
    adash = np.zeros(len(zetas), dtype=complex)
    bdash = np.zeros(len(zetas), dtype=complex)
    dstloopf( dx, L, q,  zetas,  len(q), len(zetas), adash, bdash)
    return adash, bdash

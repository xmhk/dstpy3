import numpy as np

from .DSTfortran import dstloopsfortran



def calc_ab_rungekutta4_flib( dx, L, q, zetas):
    adash = np.zeros(len(zetas), dtype=complex)
    bdash = np.zeros(len(zetas), dtype=complex)
    dstloopsfortran.dstloopsfort.loopf( dx, L, q,  zetas,  len(q), len(zetas), adash, bdash)
    return adash, bdash



def calc_abdiff_rungekutta4_flib( dx, L, q, zeta ):
    a=np.zeros(len(zeta), dtype=complex)
    b=np.zeros(len(zeta), dtype=complex)
    adiff=np.zeros(len(zeta), dtype=complex)
    bdiff=np.zeros(len(zeta), dtype=complex)
    dstloopsfortran.dstloopsfort.loopf_diff( dx, L, q, zeta, len(q), len(zeta), a, b, adiff, bdiff)
    return a, b, adiff, bdiff



def calc_ab_rungekutta4_flib_qp( dx, L, q, zetas):
    adash = np.zeros(len(zetas), dtype=complex)
    bdash = np.zeros(len(zetas), dtype=complex)
    dstloopsfortran.dstloopsfort.loopf_qp( dx, L, q,  zetas,  len(q), len(zetas), adash, bdash)
    return adash, bdash



def calc_abdiff_rungekutta4_flib_qp( dx, L, q, zeta ):
    a=np.zeros(len(zeta), dtype=complex)
    b=np.zeros(len(zeta), dtype=complex)
    adiff=np.zeros(len(zeta), dtype=complex)
    bdiff=np.zeros(len(zeta), dtype=complex)
    dstloopsfortran.dstloopsfort.loopf_diff_qp( dx, L, q, zeta, len(q), len(zeta), a, b, adiff, bdiff)
    return a, b, adiff, bdiff

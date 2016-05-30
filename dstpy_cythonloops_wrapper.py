import numpy as np
from DSTcython.cythonloops import DSTloopForward,\
								  DSTloopForwarddiff,\
								  DSTloopTransferMatrix,\
								  DSTloopTransferMatrixDIFF,\
								  DSTloopRK4


def calc_ab_rungekutta4_clib(dx, L, q, zeta):
	a, b  = DSTloopRK4( dx,
							len(zeta),
							len(q),
							q+0.0j,
							L,
							zeta)
	
	return a, b
								  
def calc_ab_forwarddisc_clib(dx, L, q, zeta):
	a, b  = DSTloopForward( dx,
							len(zeta),
							len(q),
							q+0.0j,
							L,
							zeta)
	
	return a, b


def calc_ab_diff_forwarddiff_clib(dx, L, q, zeta):
	a, b, adiff = DSTloopForwarddiff( dx,
							len(zeta),
							len(q),
							q+0.0j,
							L,
							zeta)
	bdiff = np.array([-1]) #not calculated (yet)
	return a, b, adiff, bdiff
	
	

def calc_ab_transfermatrix_clib(dx, L, q, zeta ):
	i =0
	qlength = np.shape(q)[0]
	zetalength = np.shape(zeta)[0]
	b = np.zeros([len(zeta),1], dtype =np.complex)
	a = np.zeros([len(zeta),1], dtype =np.complex)
	#
	# create zeta x q matrices for speedup
	# 
	zzet, qq = np.meshgrid(zeta,q)
	kksq =	-np.abs(qq)**2 - zzet**2+0.0j
	kk=np.sqrt(kksq)
	zzetsq = zzet**2
	coshkdxm = np.cosh(kk*dx)
	sinhkdxm = np.sinh(kk*dx) 
		
	UU00 = coshkdxm - 1.0j*zzet / kk * sinhkdxm
	UU01 = qq/kk * sinhkdxm
	UU10 = -1.0 * np.conj(qq) / kk * sinhkdxm
	UU11 = coshkdxm + 1.0j* zzet / kk * sinhkdxm
		
	U = np.zeros([2,2], dtype=np.complex)  
	S00, S10	= DSTloopTransferMatrix(len(zeta), len(q), UU00, UU01, UU10, UU11)  
	a = S00 * np.exp( 2.0j * zeta * L)
	b = S10
	return a,b		
	
	

#calc coefficents a and b and their derivatives	
def calc_ab_diff_transfermatrix_clib(dx, L, q, zeta ):
	i =0
	qlength = np.shape(q)[0]
	zetalength = np.shape(zeta)[0]
	a = np.zeros([len(zeta),1], dtype =np.complex)
	adiff = np.zeros([len(zeta),1], dtype =np.complex)
	b = np.zeros([len(zeta),1], dtype =np.complex)
	bdiff = np.zeros([len(zeta),1], dtype =np.complex)	
	
	#
	# create zeta x q matrices for speedup
	# 
	zzet, qq = np.meshgrid(zeta,q)
	
	kksq     =  -np.abs(qq)**2 - zzet**2+0.0j
	kk       = np.sqrt(kksq)	
	zzetsq   = zzet**2
	coshkdxm = np.cosh(kk*dx)
	sinhkdxm = np.sinh(kk*dx) 
	
	
	UU00 = coshkdxm - 1.0j * zzet / kk * sinhkdxm
	UU11 = coshkdxm + 1.0j * zzet / kk * sinhkdxm
	
	UU01 =                qq  / kk * sinhkdxm
	UU10 = -1.0 * np.conj(qq) / kk * sinhkdxm
	
	
	UDASH00 =  1.0j * dx * zzetsq / kksq * coshkdxm - (zzet * dx + 1.0j + 1.0j * zzetsq / kksq) * sinhkdxm / kk
	UDASH11 = -1.0j * dx * zzetsq / kksq * coshkdxm - (zzet * dx - 1.0j - 1.0j * zzetsq / kksq) * sinhkdxm / kk
	
	UDASH01 =        -qq  * zzet / kksq * (dx * coshkdxm - sinhkdxm/kk)
	UDASH10 = np.conj(qq )* zzet / kksq *( dx * coshkdxm - sinhkdxm/kk)
	
	
	S00, S10 , S20, S22, S30, S32= DSTloopTransferMatrixDIFF(len(zeta), len(q), 
											UU00, UU01, UU10, UU11,
											UDASH00, UDASH01, UDASH10, UDASH11)
  
	a = S00 * np.exp( 2.0j * zeta * L)
	adiff = (S20 +1.0j * L *(S00 + S22)) * np.exp(2.0j * zeta * L)
	b = S10
	bdiff = S30 + 1.0j * L * (S32 - S10)
	return a,b,adiff, bdiff	
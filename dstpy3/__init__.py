import numpy as np


import numpy.linalg as linalg
from scipy.linalg import toeplitz

#the DST object and the algorithms implemented here are heavily based by the following papers
#
# G. BOFFETTA AND A. R. OSBORNE	 Computation of the Direct Scattering Transform for the Nonlineare Schroedinger Equation
#								 Journal of Comp Phys 102, p 252-264, (1992)
#								 http://personalpages.to.infn.it/~boffetta/Papers/bo92.pdf
#
#
#
# Mansoor I. Yousefi, Frank R. Kschischang:	 Information Transmission using the Nonlinear Fourier Transform, Part I
#											 https://arxiv.org/abs/1202.3653
#
#											 Information Transmission using the Nonlinear Fourier Transform, Part II
#											 http://arxiv.org/abs/1204.0830
#
#
#											 Information Transmission using the Nonlinear Fourier Transform, Part III
#											 http://arxiv.org/abs/1302.2875
#


ab_methodsdict = {}
ab_methodnamesdict = {}


#
# import vanilla algos
#
from .dstpy_vanilla_algos import *	 # those should work when numpy + scipy is present


ab_methodsdict = {
					'TM'   : calc_ab_transfermatrix_vanilla,
					'CD'   : calc_ab_centraldifference_vanilla,
					'CN'   : calc_ab_cranknicolson_vanilla,
					'AL'   : calc_ab_ablowitzladik_vanilla,
					'AL2'  : calc_ab_ablowitzladik2_vanilla,
					'FD'   : calc_ab_forwarddisc_vanilla,
					'RK4'  : calc_ab_rungekutta4_vanilla,

									}

ab_methodnamesdict =	 {
					'TM'	 : 'Transfer Matrix (Python)',
					'CD'	 : 'Central Discretization (Python)',
					'CN'	 : 'Crank Nicolson (Python)',
					'AL'	 : 'Ablowitz Ladik (Python)',
					'AL2'	 : 'Ablowitz Ladik Norm (Python)',
					'FD'	 : 'Forward Discretization (Python)',
					'RK4'	 : 'Runge-Kutta 4 (Python)',
									}


abdiff_methodsdict = {'AL'	 : calc_ab_diff_ablowitzladik_vanilla}
abdiff_methodnamesdict = { 'AL'	 : 'Ablowitz Ladik (Python)'}
abdiff_methodsdict = {'AL2'	 : calc_ab_diff_ablowitzladik2_vanilla}
abdiff_methodnamesdict = { 'AL2'	 : 'Ablowitz Ladik2 (Python)'}
abdiff_methodsdict['RK4'] =	 calc_ab_diff_rk4_vanilla2
abdiff_methodnamesdict['RK4']= 'Runge-Kutta 4  (Python)'
abdiff_methodsdict['TM'] =	calc_abdiff_transfermatrix_vanilla
abdiff_methodnamesdict['TM']= 'Transfer Matrix (Python)'


#
# import arbitrary precision version. Import Errors occur when mpmath module is not installed
#
try:
	from .dstpy_arbprec_mpc import *
	ab_methodsdict['RK4_AP'] = calc_ab_rungekutta4_vanilla_ap
	ab_methodnamesdict['RK4_AP'] = "Runge Kutta 4 (Python, arbitray precision), Warning: SLOW!!!"
	abdiff_methodsdict['RK4_AP'] = calc_abdiff_rungekutta4_vanilla_ap
	abdiff_methodnamesdict['RK4_AP'] = "Runge Kutta 4 (Python, arbitray precision), Warning: SLOW!!!"


except ImportError:
	print("Warning: Problems with arbitray precision module ...")

# depricated
#
# import cython loops: possibly need to be compiled on every machine. Look in DSTcython folder
#
#try:
#	from .dstpy_cythonloops_wrapper import *
#	ab_methodsdict['TMC'] = calc_ab_transfermatrix_clib
#	ab_methodsdict['FDC'] = calc_ab_forwarddisc_clib
#	ab_methodsdict['RK4C'] =calc_ab_rungekutta4_clib
#	ab_methodnamesdict['TMC'] = 'Transfer Matrix (C)'
#	ab_methodnamesdict['FDC'] = 'Forward Discretization (C)'
#	ab_methodnamesdict['RK4C'] = 'Runge-Kutta 4 (C)'
#	abdiff_methodsdict['ALC'] = calc_al_diff_clib
#	abdiff_methodnamesdict['ALC'] = 'Ablowitz Ladik (C)'
#	abdiff_methodsdict['RK4C'] = calc_ab_diff_rungekutta4_clib
#	abdiff_methodnamesdict['RK4C'] = 'Runge Kutta 4 (C)'
#except ImportError:
#	print("Warning: Problems with cython loops module ...")



#
# import fortran loops: possibly need to be compiled on every machine. Look in DSTfortran folder
#
try:
	from .dstpy_fortran_wrapper import *
	ab_methodsdict['RK4F'] = calc_ab_rungekutta4_flib
	ab_methodnamesdict['RK4F'] = 'RK4 (Fortran)'
	abdiff_methodsdict['RK4F'] = calc_abdiff_rungekutta4_flib
	abdiff_methodnamesdict['RK4F'] = 'Runge-Kutta 4 (Fortran)'
	ab_methodsdict['RK4FQ'] = calc_ab_rungekutta4_flib_qp
	ab_methodnamesdict['RK4FQ'] = 'RK4 (Fortran, quad)'
	abdiff_methodsdict['RK4FQ'] = calc_abdiff_rungekutta4_flib_qp
	abdiff_methodnamesdict['RK4FQ'] = 'Runge-Kutta 4 (Fortran, quad)'

	ab_methodsdict['TMF'] = calc_ab_tm_flib
	ab_methodnamesdict['TMF'] = 'TM (Fortran)'
	abdiff_methodsdict['TMF'] = calc_abdiff_tm_flib
	abdiff_methodnamesdict['TMF'] = 'Runge-Kutta 4 (Fortran)'
	ab_methodsdict['TMFQ'] = calc_ab_tm_flib_qp
	ab_methodnamesdict['TMFQ'] = 'TM (Fortran, quad)'
	abdiff_methodsdict['TMFQ'] = calc_abdiff_tm_flib_qp
	abdiff_methodnamesdict['TMFQ'] = 'Runge-Kutta 4 (Fortran, quad)'

	
	abdiff_methodsdict['AL'] = calc_abdiff_al_flib
	abdiff_methodnamesdict['AL'] = 'Ablowitz-Ladik (Fortran)'
	abdiff_methodsdict['AL'] = calc_abdiff_al2_flib
	abdiff_methodnamesdict['AL'] = 'Ablowitz-Ladik2 (Fortran)'
	
  
except ImportError:
	print("Warning: Problems with fortran  loops module ...")

def dstinfo():
	print("calc_ab methods available:")
	#for k in ab_methodsdict.keys().sort():
	for k in sorted(ab_methodnamesdict.keys()):
		print("- %s : %s"%(k, ab_methodnamesdict[k]))

	print("\n\ncalc_abdiff methods available:")
	#for k in abdiff_methodsdict:
	for k in sorted(abdiff_methodnamesdict.keys()):
		print("- %s : %s"%(k, abdiff_methodnamesdict[k]))

	print("\neigenvalue calculation: use .calc_evals() / .calc_evals_spec() BUT this is slow and buggy.")


class DSTObj():
	def __init__(self, field, tvec, fiberbeta2, gamm, t0scaleext = None):
		self.b2 = fiberbeta2
		self.gamm = gamm
		dt = tvec[1]-tvec[0]
		npoints = len(tvec)
		if t0scaleext == None:
			t0scale = dt;
		else:
			t0scale = t0scaleext
		p0scale = np.sqrt( t0scale * t0scale * self.gamm / np.abs(self.b2))
		self.xvec = tvec / t0scale
		self.dx = self.xvec[1] - self.xvec[0]
		self.L	= np.max(self.xvec)
		self.q = field * p0scale
		self.unscaled_energy = np.sum( np.abs(field)**2 * dt)
		self.scaled_energy = np.sum( np.abs(self.q)**2 * self.dx)
		self.energyfactor = p0scale * np.sqrt(self.gamm/np.abs(self.b2))
		self.zetamax = self.scaled_energy / 4.
		self.scaled_dom = np.pi / (2 * self.L)
		self.ommax = np.floor( npoints / 2. ) * self.scaled_dom

		self.calc_ab_methodsdict = ab_methodsdict
		self.calc_ab_methodnamesdict =	ab_methodnamesdict
		self.calc_abdiff_methodsdict =abdiff_methodsdict
		self.calc_abdiff_methodnamesdict =abdiff_methodnamesdict


	def calc_ab(self,zetas, method = 'TMC'):
		if method not in self.calc_ab_methodsdict.keys():
			print("\n calc_ab method should be in ",self.calc_ab_methodsdict.keys())
			a = 0
			b = 0
		else:
			if np.isscalar(zetas):
				zetas = np.array([zetas])
			mfun = self.calc_ab_methodsdict[method]

			a, b = mfun(self.dx, self.L, self.q, zetas)
			if len(a) ==1:
				a = a[0]
				b = b[0]
		return a,b

	def calc_abdiff(self,zetas, method = 'ALC'):
		if method not in self.calc_abdiff_methodsdict.keys():
			print("\n calc_ab method should be in ",self.calc_ab_methodsdict.keys())
			a = 0
			b = 0
		else:
			if np.isscalar(zetas):
				zetas = np.array([zetas])
			mfun = self.calc_abdiff_methodsdict[method]
			a, b, ad, bd = mfun(self.dx, self.L, self.q, zetas)
			if len(a) ==1:
				#print(a,b,ad, bd)
				a = a[0]
				b = b[0]
				ad = ad[0]
				bd = bd[0]
		return a,b, ad, bd

	def calc_evals(self):
		#method based on central difference matrix
		len_q = len(self.q)
		cdm = np.zeros( [len_q,len_q], dtype=complex)
		for i in range(1, len_q):
			cdm[i,i-1] = -1
			cdm[i-1,i] =  1
		cdm[0, len_q-1] = -1
		cdm[len_q-1, 0] =  1
		cdm = cdm / 2. / self.dx

		MM = np.zeros([2*len_q, 2*len_q], dtype=complex)
		MM[	   0:len_q,		   0:len_q	] =	   cdm
		MM[	   0:len_q,	   len_q:2*len_q] = -1.0 * np.diag(self.q)
		MM[len_q:2*len_q,	   0:len_q	] = -1.0 * np.diag(np.conj(self.q))
		MM[len_q:2*len_q,  len_q:2*len_q] = -1.0 * cdm

		MM = 1.0j * MM
		evals, evecs = linalg.eig(MM)
		return evals#, evecs


	def calc_evals_spec(self):
		#matrix spectral method after Yousefi
		def matrixGAMMA(vGAMMA):
			LGAMMA2 = np.int(	len(vGAMMA)/2)
			GAMMA	= np.zeros( [len(vGAMMA), len(vGAMMA)], dtype=complex)
			for row in range( len(vGAMMA)):
				mstart = max( 0, row-LGAMMA2+1 )	#col index to start filling matrix GAMMA with elements
				mend   = min( row+LGAMMA2+1, 2*LGAMMA2 )#col index to stop ...
				gstart = min( row, LGAMMA2-1 )		#start index of vGAMMA to fill matrix with
				ii = mstart
				gammi = gstart
				while ii<mend:
				  GAMMA[row, ii] = vGAMMA[gammi]
				  ii+=1
				  gammi+=-1
			return	-1.0j * GAMMA

		qft	  = np.fft.ifft(self.q)
		M	  = len(self.q)
		M2	  = int( M/2.0)
		Gamma = matrixGAMMA(qft)
		Omega = -2*np.pi / 2/self.L * np.diag(np.arange(-M2,M2))

		matrixA = np.zeros([2*M,2*M], dtype=complex)
		matrixA[0:M, 0:M]	 = Omega
		matrixA[0:M, M:2*M]	 = Gamma
		matrixA[M:2*M, 0:M]	 = -1 * np.conj(np.transpose(Gamma))
		matrixA[M:2*M,M:2*M] = -1 * Omega
		evals, evecs = linalg.eig( matrixA)
		return evals#, evecs

	def calc_evals_spec2(self, muk = .0):
		#matrix spectral method after YANG
		# from scipy.linalg import toeplitz
		qft	  = np.fft.fftshift( np.fft.ifft(self.q) )
		M	  = len(self.q)
		M2	  = int( M/2.0)
		k = 2 * np.pi / 2/ self.L

		B1 = 1.0j * k *	 np.diag(np.arange( -M2, M2)) + muk * k * np.identity(M)

		toepcv = np.zeros( M , dtype = complex)
		toeprv = np.zeros( M , dtype = complex)
		toepcv[0:M2] = qft[M2:M]
		toeprv[0:M2] = qft[M2:0:-1]
		B2 = toeplitz( toepcv, toeprv)
		MM = np.zeros( [2 * M, 2*M], dtype=complex)
		MM[0:M, 0:M] = -B1
		MM[0:M, M:2*M] = B2
		MM[M:2*M, 0:M] = np.conj(np.transpose(B2))
		MM[M:2*M,M:2*M] = B1
		evals, evecs = np.linalg.eig(-1.0j* MM)
		return evals#,evecs

		"""
		def calc_evals(self):
			#method based on central difference matrix
			len_q = len(self.q)
			cdm = np.zeros( [len_q,len_q], dtype=complex)
			for i in range(1, len_q):
				cdm[i,i-1] = -1
				cdm[i-1,i] = 1
			cdm[0, len_q-1] = -1
			cdm[len_q-1, 0] = 1
			cdm = cdm / 2. / self.dx

			MM = np.zeros([2*len_q, 2*len_q], dtype=complex)
			MM[	   0:len_q,		   0:len_q	] = cdm
			MM[	   0:len_q,	   len_q:2*len_q] = -np.diag(self.q)
			MM[len_q:2*len_q,	   0:len_q	] = -np.diag(np.conj(self.q))
			MM[len_q:2*len_q,  len_q:2*len_q] = -cdm

			MM = 1.0j * MM
			evals, evecs = linalg.eig(MM)
			return evals


		def calc_evals_spec(self):
			#matrix spectral method after Yousefi
			def matrixGAMMA(vGAMMA):
				LGAMMA2 = np.int(	len(vGAMMA)/2)
				GAMMA	= np.zeros( [len(vGAMMA), len(vGAMMA)], dtype=complex)
				for row in range( len(vGAMMA)):
					mstart = max( 0, row-LGAMMA2+1 )	#col index to start filling matrix GAMMA with elements
					mend   = min( row+LGAMMA2+1, 2*LGAMMA2 )#col index to stop ...
					gstart = min( row, LGAMMA2-1 )		#start index of vGAMMA to fill matrix with
					ii = mstart
					gammi = gstart
					while ii<mend:
					  GAMMA[row, ii] = -1.0j * vGAMMA[gammi]
					  ii+=1
					  gammi+=-1
				return GAMMA

			qft	  = np.fft.ifft(self.q)
			M	  = len(self.q)
			M2	  = int( M/2.0)
			Gamma = matrixGAMMA(qft)
			Omega = -2*np.pi / 2/self.L * np.diag(np.arange(-M2,M2))

			matrixA = np.zeros([2*M,2*M], dtype=complex)
			matrixA[0:M, 0:M]	 = Omega
			matrixA[0:M, M:2*M]	 = Gamma
			matrixA[M:2*M, 0:M]	 = -1 * np.conj(np.transpose(Gamma))
			matrixA[M:2*M,M:2*M] = -1 * Omega
			evals, evecs = linalg.eig(matrixA)
			return evals
			"""

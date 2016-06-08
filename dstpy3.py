import numpy as np
from dstpy_cythonloops_wrapper import *
from dstpy_vanilla_algos import *
import numpy.linalg as linalg


#import ctypes


class DSTObj():
	def __init__(self, field, tvec, fiberbeta2, gamm):
		self.b2 = fiberbeta2
		self.gamm = gamm
		dt = tvec[1]-tvec[0]
		npoints = len(tvec)
		t0scale = dt;
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
		
		self.calc_ab_methodsdict = {'TMC': calc_ab_transfermatrix_clib,
					                'TM' : calc_ab_transfermatrix_vanilla,
					                'CD' : calc_ab_centraldifference_vanilla,
					                'CN' : calc_ab_cranknicolson_vanilla,
					                'AL' : calc_ab_ablowitzladik_vanilla,
					                'AL2': calc_ab_ablowitzladik2_vanilla,
					                'FD' : calc_ab_forwarddisc_vanilla,
									'FDC' : calc_ab_forwarddisc_clib,
									'RK4' : calc_ab_rungekutta4_vanilla, 
									'RK4C' : calc_ab_rungekutta4_clib
									}       
									        
		self.calc_ab_methodnamesdict = 	 {'TMC': 'Transfer Matrix (C)',
										  'TM' : 'Transfer Matrix (Python)',
										  'CD' : 'Central Discretization (Python)',
										  'CN' : 'Crank Nicolson (Python)',
										  'AL' : 'Ablowitz Ladik (Python)',
										  'AL2': 'Ablowitz Ladik Norm (Python)',
										  'FD' : 'Forward Discretization (Python)',										  
										  'FDC' : 'Forward Discretization (C)', 
										  'RK4' : 'Runge-Kutta 4 (Python)',
										  'RK4C' : 'Runge-Kutta 4 (Python)'}
										  
		self.calc_abdiff_methodsdict = {#'TMC' : calc_ab_diff_transfermatrix_clib,
										#'FDC' : calc_ab_diff_forwarddiff_clib,
										#'RK4C' : calc_ab_diff_rk4_clib,
										#'RK4' :  calc_abdiff_rungekutta_vanilla,
										#'FD'  : calc_ab_diff_forwarddisc_vanilla,
										'AL'  : calc_ab_diff_ablowitzladik_vanilla,
										'ALC'  : calc_al_diff_clib}
										
		self.calc_abdiff_methodnamesdict = {#'TMC' : 'Transfer Matrix (C)',
										#'FDC' : 'Forward Discretization (C)',
										#'FD'  : 'Forward Discretization (Python)',
										'AL'  : 'Ablowitz Ladik (Python)',
										'ALC'  : 'Ablowitz Ladik (C)',
										#'RK4C' : 'Runge Kutta 4 (C)',
										#'RK4' : 'Runge Kutta 4 (Python)'
										}		

												
										  
										  
	def  help(self):
		print("calc_ab methods available:")		
		for k in self.calc_ab_methodsdict:
			print("                           %s : %s"%(k, self.calc_ab_methodnamesdict[k]))
			
		print("\n\ncalc_abdiff methods available:")	
		for k in self.calc_abdiff_methodsdict:
			print("                           %s : %s"%(k, self.calc_abdiff_methodnamesdict[k]))		

		print("\n eigenvalue calculation: use .calc_evals() BUT this is slow and gives N false (mirrored) evals.")
	
	
	def calc_ab(self,zetas, method = 'RK4C'):
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
		len_q = len(self.q)
		cdm = np.zeros( [len_q,len_q], dtype=complex)
		for i in range(1, len_q):
			cdm[i,i-1]=-1
			cdm[i-1,i]=1
			cdm[0, len_q-1] = -1
			cdm[len_q-1, 0] = 1
		cdm = cdm / 2. /self.dx
		MM = np.zeros([2*len_q, 2*len_q], dtype=complex)
		MM[0:len_q, 0:len_q]=cdm
		MM[len_q:2*len_q,len_q:2*len_q] = -cdm
		MM[len_q:2*len_q, 0:len_q] = -np.diag(np.conj(self.q))
		MM[0:len_q, len_q:2*len_q]=  -np.diag(self.q)
		MM = 1.0j * MM    
		evals, evecs = linalg.eig(MM)
		return evals
			
		
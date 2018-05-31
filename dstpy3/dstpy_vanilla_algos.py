import numpy as np

def calc_ab_rungekutta4_vanilla( dx, L, q, zeta ):
	def fpk(qii, zetai):
			return np.array( [[-1.0j * zetai,		   qii],
							  [-np.conj(qii), 1.0j * zetai]] )
	a=np.zeros(len(zeta), dtype=complex)
	b=np.zeros(len(zeta), dtype=complex)
	for i in range(len(zeta)):
		v = np.zeros([len(q),2],dtype=complex)		  
		v[0,:] = np.array([1,0]) * np.exp( -1.0j * zeta[i] * -L)
		for ii in range(0,len(q)-1):
			fpkqii =  fpk( q[ii], zeta[i])
			k1 = np.dot( fpkqii, v[ii])
			k2 = np.dot( fpkqii, v[ii] + dx/2.0 * k1)
			k3 = np.dot( fpkqii, v[ii] + dx/2.0 * k2)
			k4 = np.dot( fpkqii, v[ii] + dx *k3)
			v[ii+1,:] = v[ii,:] +  dx * 1./6 *( k1 + 2*k2 + 2*k3 + k4)
		a[i] = v[-1,0] * np.exp(1.0j * zeta[i] * L)
		b[i] = v[-1,1] * np.exp(-1.0j * zeta[i] * L)
	return a, b


def calc_ab_transfermatrix_vanilla (dx, L, q, zeta ):
	i =0
	qlength = np.shape(q)[0]
	zetalength = np.shape(zeta)[0]
	a=np.zeros(len(zeta), dtype=complex)
	b=np.zeros(len(zeta), dtype=complex)
	zzet, qq = np.meshgrid(zeta,q)
	kk=np.sqrt( -np.abs(qq)**2 - zzet**2+0.0j)
	coshkdxm = np.cosh(kk*dx)
	sinhkdxm = np.sinh(kk*dx)
	UU00 = coshkdxm - 1.0j*zzet / kk * sinhkdxm
	UU01 = qq/kk * sinhkdxm
	UU10 = -1.0 * np.conj(qq) / kk * sinhkdxm
	UU11 = coshkdxm + 1.0j* zzet / kk * sinhkdxm
	U = np.zeros([2,2], dtype=np.complex)
	while i<zetalength:
		S = np.zeros([2,2], dtype=np.complex) 
		S[0,0]=1.0
		S[1,1]=1.0
		ii = qlength-1
		while ii>=0:
			U[0,0] = UU00[ii,i]
			U[0,1] = UU01[ii,i]
			U[1,0] = UU10[ii,i]
			U[1,1] = UU11[ii,i]
			S = np.dot(S, U)
			ii=ii-1
		a[i] = S[0,0] * np.exp( 2.0j * zeta[i] * L)
		b[i] = S[1,0]
		i+=1
	return a,b
	

def calc_ab_centraldifference_vanilla( dx, L, q, zeta ):
	a=np.zeros(len(zeta), dtype=complex)
	b=np.zeros(len(zeta), dtype=complex)
	for i in range(len(zeta)):
		v = np.zeros([len(q),2],dtype=complex)
		#calculate the first two elements of v
		v[0,:] = np.array([1,0]) * np.exp( -1.0j * zeta[i] * -L)
		p0 = np.array( [[-1.0j * zeta[i], q[0]],[-np.conj(q[0]), 1.0j * zeta[i]]] )
		v[1,:] = v[0,:] + dx * np.dot(p0, v[0,:])
		#calculate v[2]...v[N-1]
		for k in range(1,len(q)-1):
			pk = np.array( [[-1.0j * zeta[i], q[k]],[-np.conj(q[k]), 1.0j * zeta[i]]] )
			v[k+1,:] = v[k-1,:] + 2 * dx * np.dot(pk, v[k,:])
		a[i] = v[-1,0] * np.exp(1.0j * zeta[i] * L)
		b[i] = v[-1,1] * np.exp(-1.0j * zeta[i] * L)
	return a, b


def calc_ab_cranknicolson_vanilla( dx, L, q, zeta ):
	a=np.zeros(len(zeta), dtype=complex)
	b=np.zeros(len(zeta), dtype=complex)
	imat = np.array([[1,0],[0,1]])
	for i in range(len(zeta)):
		v = np.zeros([len(q),2],dtype=complex)
		v[0,:] = np.array([1,0]) * np.exp( -1.0j * zeta[i] * -L)
		pk = np.array( [[-1.0j * zeta[i], q[0]],[-np.conj(q[0]), 1.0j * zeta[i]]] )
		for k in range(0,len(q)-1):
			pk_plus_1 = np.array( [[-1.0j * zeta[i], q[k+1]],[-np.conj(q[k+1]), 1.0j * zeta[i]]] )

			v[k+1,:] = np.dot( np.dot(np.linalg.inv( imat-dx/2. *pk_plus_1 ),
									  imat+dx/2. *pk)
							  , v[k])
			pk = pk_plus_1
		a[i] = v[-1,0] * np.exp(1.0j * zeta[i] * L)
		b[i] = v[-1,1] * np.exp(-1.0j * zeta[i] * L)
	return a, b

def calc_ab_ablowitzladik_vanilla( dx, L, q, zeta ):
	a=np.zeros(len(zeta), dtype=complex)
	b=np.zeros(len(zeta), dtype=complex)
	for i in range(len(zeta)):
		v = np.zeros([len(q),2],dtype=complex)
		v[0] = np.array([1,0])*np.exp(-1.0j *zeta[i]*-L)
		for k in range(0,len(q)-1):
			z = np.exp(-1.0j *zeta[i] * dx)
			Qk = q[k]*dx
			v[k+1]=np.dot ( np.array([ [z, Qk],[-np.conj(Qk), 1./z]]) , v[k])
		a[i] = v[-1,0] * np.exp( 1.0j * zeta[i] * L)
		b[i] = v[-1,1] * np.exp(-1.0j * zeta[i] * L)
	return a, b

def calc_ab_ablowitzladik2_vanilla(	 dx, L, q, zeta	 ):
	a=np.zeros(len(zeta), dtype=complex)
	b=np.zeros(len(zeta), dtype=complex)
	for i in range(len(zeta)):
		v = np.zeros([len(q),2],dtype=complex)
		v[0] = np.array([1,0])*np.exp(-1.0j *zeta[i]*-L)
		for k in range(0,len(q)-1):
			z = np.exp(-1.0j *zeta[i] * dx)
			Qk = q[k]*dx
			#R = np.array(	 [ [z, Qk],[-np.conj(Qk), 1./z]])
			v[k+1]=1./ np.sqrt( 1+ np.abs(Qk)**2  )*np.dot ( np.array([ [z, Qk],[-np.conj(Qk), 1./z]]) , v[k])
		a[i] = v[-1,0] * np.exp(1.0j * zeta[i]	* L)
		b[i] = v[-1,1] * np.exp(-1.0j * zeta[i] * L)
	return a, b


def calc_ab_forwarddisc_vanilla(  dx, L, q, zeta  ):
	a=np.zeros(len(zeta), dtype=complex)
	b=np.zeros(len(zeta), dtype=complex)
	for i in range(len(zeta)):
		v = np.zeros([len(q),2],dtype=complex)
		#calculate the first element of v
		v[0,:] = np.array([1,0]) * np.exp( -1.0j * zeta[i] * -L)
		#calculate v[1]...v[N-1]
		for k in range(0,len(q)-1):
			A = np.array( [[-1.0j * zeta[i], q[k]],[-np.conj(q[k]), 1.0j * zeta[i]]] )
			v[k+1,:] = v[k,:] + dx * np.dot(A, v[k,:])
		a[i] = v[-1,0] * np.exp(1.0j * zeta[i] * L)
		b[i] = v[-1,1] * np.exp(-1.0j * zeta[i] * L)
	return a, b



def calc_ab_diff_rk4_vanilla2( dx, L, q, zeta ):
	def fpk( qii, zetai): 
		return np.array( [[ -1.0j * zetai	   ,  qii				, 0				   ,	 0 ],
						  [1.0* -np.conj(qii)	,  1.0j * zetai		, 0				   ,	 0	],
						  [ -1.0j				 ,	 0				   ,  -1.0j * zetai ,	qii ] ,
						  [0					 ,	  1.0j			   ,  1.0* -np.conj(qii) ,	 1.0j * zetai]]
						 )
	a=np.zeros(len(zeta), dtype=complex)
	b=np.zeros(len(zeta), dtype=complex)
	adiff=np.zeros(len(zeta), dtype=complex)
	bdiff=np.zeros(len(zeta), dtype=complex)
	for i in range(len(zeta)):
		v = np.zeros([len(q),4],dtype=complex)		
		v[0,0] =			   np.exp( -1.0j * zeta[i] * -L)
		v[0,2] =  -1.0j * -L * np.exp(-1.0j * zeta[i] * -L)
		for ii in range(0,len(q)-1):
			fpkqii =  fpk( q[ii], zeta[i])
			k1 = np.dot( fpkqii, v[ii])
			k2 = np.dot( fpkqii, v[ii] + dx/2.0 * k1)
			k3 = np.dot( fpkqii, v[ii] + dx/2.0 * k2)
			k4 = np.dot( fpkqii, v[ii] + dx *k3)
			v[ii+1,:] = v[ii,:] +  dx * 1./6 *( k1 + 2*k2 + 2*k3 + k4)
		a[i] = v[-1,0] * np.exp(1.0j * zeta[i] * L)
		b[i] = v[-1,1] * np.exp(-1.0j * zeta[i] * L)
		adiff[i] = (v[-1,2] + 1.0j * L * v[-1,0]) * np.exp(1.0j * zeta[i] *L)
		bdiff[i] = np.array([0])
	return a, b, adiff, bdiff	 
	
def calc_abdiff_transfermatrix_vanilla( dx, L, q, zeta ):
	a=np.zeros(len(zeta), dtype=np.complex)
	b=np.zeros(len(zeta), dtype=np.complex)
	ad=np.zeros(len(zeta), dtype=np.complex)
	bd=np.zeros(len(zeta), dtype=np.complex)	
	T = np.zeros( [4,4], dtype = np.complex)  
	
	zzet, qq = np.meshgrid(zeta,q)
	kk=np.sqrt( -np.abs(qq)**2 - zzet**2+0.0j)
	coshkdxm = np.cosh(kk*dx)
	sinhkdxm = np.sinh(kk*dx)
	UU00 = coshkdxm - 1.0j*zzet / kk * sinhkdxm
	UU01 = qq/kk * sinhkdxm
	UU10 = -1.0 * np.conj(qq) / kk * sinhkdxm
	UU11 = coshkdxm + 1.0j* zzet / kk * sinhkdxm
	
	UD00 = 1.0j * dx * zzet**2 / kk**2 * coshkdxm \
		   -(zzet * dx + 1.0j + 1.0j * zzet**2/kk**2) * sinhkdxm/kk
	UD01 = -1 * qq * zzet / kk**2 * (dx * coshkdxm - sinhkdxm/kk)
	UD10 = -1 * -1 * np.conj(qq) * zzet / kk**2 *(dx * coshkdxm - sinhkdxm/kk)
	UD11 =	-1.0j * dx * zzet**2 / kk**2 * coshkdxm \
		   -(zzet * dx - 1.0j - 1.0j * zzet**2/kk**2) * sinhkdxm/kk
	i=0
	while i <len(zeta):		 
		S = np.zeros( [4,4], dtype = np.complex)
		S[0,0] = 1.0
		S[1,1] = 1.0
		S[2,2] = 1.0
		S[3,3] = 1.0
		ii = len(q)-1
		while ii>=0:  
			T[0,0] = UU00[ii,i]
			T[0,1] = UU01[ii,i]
			T[1,0] = UU10[ii,i]
			T[1,1] = UU11[ii,i]
			
			T[2,2] = UU00[ii,i]
			T[2,3] = UU01[ii,i]
			T[3,2] = UU10[ii,i]
			T[3,3] = UU11[ii,i]
			
			T[2,0] = UD00[ii,i]
			T[2,1] = UD01[ii,i]
			T[3,0] = UD10[ii,i]
			T[3,1] = UD11[ii,i]
			S = np.dot(S, T)
			ii=ii-1
		a[i]=S[0,0] * np.exp ( 2.0j * zeta[i] * L)
		b[i]=S[1,0]
		ad[i] = ( S[2,0] + 1.0j * L *(S[0,0] + S[2,2]) )* np.exp(2.0j * zeta[i] * L)
		bd[i] = S[3,0] + 1.0j * L * (S[3,2]-S[1,0])		  
		i=i+1		 
	return a, b, ad, bd	   
	

def calc_ab_diff_ablowitzladik_vanilla(	 dx, L, q, zeta	 ):
	a=np.zeros(len(zeta), dtype=complex)
	adiff=np.zeros(len(zeta), dtype=complex)
	b=np.zeros(len(zeta), dtype=complex)
	for i in range(len(zeta)):
		v	  = np.zeros([len(q),4],dtype=complex)
		v[0]	 = np.array([  1, 0,  -1.0j * -L , 0] ) *			  np.exp(-1.0j * zeta[i] * -L)
		for ii in range(0,len(q)-1):
			z = np.exp(-1.0j * zeta[i] * dx)
			A	  = np.array( [[z					  , q[ii] *dx	  ,				0.0			 ,		0.0		],
							   [-np.conj( q[ii] )*dx	  , 1.0	 / z	  ,				0.0			 ,		0.0		],
							   [ -1.0j * dx * z		  , 0.0			  ,						 z	 ,	   q[ii] *dx ],
							   [0.0					 , 1.0j* dx / z	 ,		-np.conj( q[ii] )*dx ,	   1.0/z   ]] )
			v[ii+1]	   = np.dot ( A, v[ii] )
		a[i] =	v[-1,0] * np.exp(1.0j * zeta[i] * L)
		adiff[i] = (v[-1,2] + 1.0j * L * v[-1,0]) * np.exp(1.0j * zeta[i] *L)
		b[i] = v[-1,1] * np.exp(-1.0j * zeta[i] * L)
		bdiff = np.array([0])
	return a, b	, adiff, bdiff


def calc_ab_diff_ablowitzladik2_vanilla(  dx, L, q, zeta  ):
	a=np.zeros(len(zeta), dtype=complex)
	adiff=np.zeros(len(zeta), dtype=complex)
	b=np.zeros(len(zeta), dtype=complex)
	for i in range(len(zeta)):
		v	  = np.zeros([len(q),4],dtype=complex)
		v[0]	 = np.array([  1, 0,  -1.0j * -L , 0] ) *			  np.exp(-1.0j * zeta[i] * -L)
		for ii in range(0,len(q)-1):
			z = np.exp(-1.0j * zeta[i] * dx)
			A	  = np.array( [[z					  , q[ii] *dx	  ,				0.0			 ,		0.0		],
							   [-np.conj( q[ii] )*dx	  , 1.0	 / z	  ,				0.0			 ,		0.0		],
							   [ -1.0j * dx * z		  , 0.0			  ,						 z	 ,	   q[ii] *dx ],
							   [0.0					 , 1.0j* dx / z	 ,		-np.conj( q[ii] )*dx ,	   1.0/z   ]] )
			v[ii+1]	   = 1.0/ np.sqrt(1.0+ np.abs(q[ii]*dx)**2) * np.dot ( A, v[ii] )
		a[i] =	v[-1,0] * np.exp(1.0j * zeta[i] * L)
		adiff[i] = (v[-1,2] + 1.0j * L * v[-1,0]) * np.exp(1.0j * zeta[i] *L)
		b[i] = v[-1,1] * np.exp(-1.0j * zeta[i] * L)
		bdiff = np.array([0])
	return a, b	, adiff, bdiff	  
	


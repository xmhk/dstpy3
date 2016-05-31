import numpy as np
cimport numpy as np
DTYPE=np.complex128
ctypedef np.complex128_t DTYPE_t


cdef RK4PK( complex qii, complex zetai ):
	cdef complex pk00 = -1.0j * zetai
	cdef complex pk01 = qii
	cdef complex pk10 =  - np.conj(qii)
	cdef complex pk11 = 1.0j * zetai
	return pk00, pk01, pk10, pk11
	
cdef MatDotVec(complex A00, complex A01, complex A10, complex A11, complex v1, complex v2):
	cdef complex r1 = A00 * v1 + A01 * v2
	cdef complex r2 = A10 * v1 + A11 * v2
	return r1, r2
	
# calc a, b Runge Kutta 4th order
	
def DSTloopRK4(double dx,
			  int zetalength,			  
			  int qlength,
			  np.ndarray[DTYPE_t, ndim=1] q,			
			  complex L,
			  np.ndarray[DTYPE_t, ndim=1] zetas):
			  
	cdef np.ndarray[DTYPE_t,ndim=1] a= np.zeros([zetalength], dtype=DTYPE)	
	cdef np.ndarray[DTYPE_t,ndim=1] b= np.zeros([zetalength], dtype=DTYPE)	
	cdef np.ndarray[DTYPE_t,ndim=1] v1= np.zeros([qlength], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=1] v2= np.zeros([qlength], dtype=DTYPE)	
	cdef complex pk00
	cdef complex pk01
	cdef complex pk10
	cdef complex pk11
	cdef complex k11 
	cdef complex k12 
	cdef complex k21 
	cdef complex k22 
	cdef complex k31
	cdef complex k32
	cdef complex k41 
	cdef complex k42 
	cdef int i=0
	cdef int ii=0
	cdef int k=0


	while i<zetalength:
		k=0		
		while k < qlength:
			v1[k]=0
			v2[k]=0
			k = k+1
		
		# set first two elements of v
		v1[0] = np.exp( -1.0j * zetas[i] *  -L)
		pk00, pk01, pk10, pk11 = RK4PK( q[0], zetas[i])
		k11, k12 = MatDotVec( pk00, pk01, pk10, pk11, v1[0], v2[0])		
		v1[1] = v1[0] + dx * k11
		v2[1] = v2[0] + dx * k12
		
		ii=0				
		while ii < qlength-2:		
			pk00, pk01, pk10, pk11 = RK4PK( q[ii], zetas[i] )
			k11, k12 = MatDotVec(pk00, pk01, pk10, pk11, v1[ii], v2[ii] )
			
			
			pk00, pk01, pk10, pk11 = RK4PK(q[ii+1], zetas[i])
			k21, k22 = MatDotVec(pk00, pk01, pk10, pk11,
														v1[ii] + dx * k11, 
														v2[ii] + dx * k12 )
			
			k31, k32 = MatDotVec(pk00, pk01, pk10, pk11, 
														v1[ii] + dx * k21, 
														v2[ii] + dx * k22 )
			
			pk00, pk01, pk10, pk11 = RK4PK(q[ii+2], zetas[i])
			k41, k42 = MatDotVec(pk00, pk01, pk10, pk11, 
														v1[ii] + 2* dx * k31, 
														v2[ii] + 2 * dx * k32 )
														
			v1[ii+2]= v1[ii] + 2./6. * dx * (k11 + 2 * k21 + 2 * k31 + k41 )
			v2[ii+2]= v2[ii] + 2./6. * dx * (k12 + 2 * k22 + 2 * k32 + k42 )
			ii = ii + 1
		a[i] = v1[qlength-1] * np.exp( 1.0j * zetas[i] * L )
		b[i] = v2[qlength-1] * np.exp( -1.0j * zetas[i] * L )	
		i = i + 1
	return a, b



# calc a, b and derivaties (augmented) Runge Kutta 4th order
	
def DSTloopRK4diff(double dx,
			  int zetalength,			  
			  int qlength,
			  np.ndarray[DTYPE_t, ndim=1] q,			
			  complex L,
			  np.ndarray[DTYPE_t, ndim=1] zetas):
			  
	cdef np.ndarray[DTYPE_t,ndim=1] a= np.zeros([zetalength], dtype=DTYPE)	
	cdef np.ndarray[DTYPE_t,ndim=1] b= np.zeros([zetalength], dtype=DTYPE)	
			  
	cdef np.ndarray[DTYPE_t,ndim=1] adiff= np.zeros([zetalength], dtype=DTYPE)	
	cdef np.ndarray[DTYPE_t,ndim=1] bdiff= np.zeros([zetalength], dtype=DTYPE)	
	
	cdef np.ndarray[DTYPE_t,ndim=1] v1= np.zeros([qlength], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=1] v2= np.zeros([qlength], dtype=DTYPE)	
	cdef np.ndarray[DTYPE_t,ndim=1] v1diff= np.zeros([qlength], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=1] v2diff= np.zeros([qlength], dtype=DTYPE)	
	
	cdef complex Adiff00 = -1.0j * dx
	cdef complex Adiff01 = 0.0j
	cdef complex Adiff10 = 0.0j
	cdef complex Adiff11 = 1.0j * dx
	cdef complex pk00
	cdef complex pk01
	cdef complex pk10
	cdef complex pk11
	cdef complex k11 
	cdef complex k12 
	cdef complex k21 
	cdef complex k22 
	cdef complex k31
	cdef complex k32
	cdef complex k41 
	cdef complex k42 
	cdef complex tmp11
	cdef complex tmp12
	cdef complex tmp21
	cdef complex tmp22
	cdef int i=0
	cdef int ii=0
	cdef int k=0


	while i<zetalength:
		k=0		
		while k < qlength:
			v1[k]=0
			v2[k]=0
			k = k+1
		
		# set first two elements of v
		v1[0]     =               np.exp( -1.0j * zetas[i] *  -L)
		v1diff[0] =  -1.0j * -L * np.exp( -1.0j * zetas[i] * -L)
		pk00, pk01, pk10, pk11 = RK4PK( q[0], zetas[i])
		k11, k12 = MatDotVec( pk00, pk01, pk10, pk11, v1[0], v2[0])		
		v1[1] = v1[0] + dx * k11
		v2[1] = v2[0] + dx * k12
		
		tmp1,tmp2 = MatDotVec( Adiff00, Adiff01,Adiff10,Adiff11, v1[0], v2[0])
		tmp3,tmp4 = MatDotVec( pk00, pk01, pk10, pk11, v1diff[0], v2diff[0])
		
		ii=0				
		while ii < qlength-2:		
			pk00, pk01, pk10, pk11 = RK4PK( q[ii], zetas[i] )
			k11, k12 = MatDotVec(pk00, pk01, pk10, pk11, v1[ii], v2[ii] )
			
			
			pk00, pk01, pk10, pk11 = RK4PK(q[ii+1], zetas[i])
			k21, k22 = MatDotVec(pk00, pk01, pk10, pk11,
														v1[ii] + dx * k11, 
														v2[ii] + dx * k12 )
			
			k31, k32 = MatDotVec(pk00, pk01, pk10, pk11, 
														v1[ii] + dx * k21, 
														v2[ii] + dx * k22 )
			
			#next line: needed for diff
			tmp3,tmp4 = MatDotVec( pk00, pk01, pk10, pk11, v1diff[ii+1], v2diff[ii+1])
			#
			pk00, pk01, pk10, pk11 = RK4PK(q[ii+2], zetas[i])
			k41, k42 = MatDotVec(pk00, pk01, pk10, pk11, 
														v1[ii] + 2* dx * k31, 
														v2[ii] + 2 * dx * k32 )
														
			v1[ii+2]= v1[ii] + 2./6. * dx * (k11 + 2 * k21 + 2 * k31 + k41 )
			v2[ii+2]= v2[ii] + 2./6. * dx * (k12 + 2 * k22 + 2 * k32 + k42 )
			
			pk00, pk01, pk10, pk11 = RK4PK(q[ii+1], zetas[i])
			tmp1,tmp2 = MatDotVec( Adiff00, Adiff01,Adiff10,Adiff11, v1[ii+1], v2[ii+1])
			v1diff[ii+2] = tmp1 + tmp3
			v2diff[ii+2] = tmp2 + tmp4
			
			ii = ii + 1
		a[i] = v1[qlength-1] * np.exp( 1.0j * zetas[i] * L )
		b[i] = v2[qlength-1] * np.exp( -1.0j * zetas[i] * L )	
		adiff[i] = (v1diff[qlength-1] + 1.0j * L * v1[qlength-1]) * np.exp(1.0j * zetas[i] *L)
		i = i + 1
	return a, b, adiff, bdiff






#calc coefficents a and b(forward difference method)

def DSTloopForward(double dx,
			  int zetalength,			  
			  int qlength,
			  np.ndarray[DTYPE_t, ndim=1] q,			
			  complex L,
			  np.ndarray[DTYPE_t, ndim=1] zetas):			  
	cdef np.ndarray[DTYPE_t,ndim=1] a= np.zeros([zetalength], dtype=DTYPE)	
	cdef np.ndarray[DTYPE_t,ndim=1] b= np.zeros([zetalength], dtype=DTYPE)	
	cdef np.ndarray[DTYPE_t,ndim=1] v1= np.zeros([qlength], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=1] v2= np.zeros([qlength], dtype=DTYPE)	
	cdef int i=0
	cdef int k=0
	cdef complex A00 = 0
	cdef complex A01 = 0
	cdef complex A10 = 0
	cdef complex A11 = 0	
	while i<zetalength:
		k=0		
		while k < qlength:
			v1[k]=0
			v2[k]=0
			k = k+1
		v1[0] = np.exp( -1.0j * zetas[i] *  -L)
		k=0
		while k < qlength-1:		
			A00 = -1.0j * zetas[i]
			A01 = q[k] 
			A10 = -np.conj(q[k])
			A11 = 1.0j * zetas[i]			
			v1[k+1]= v1[k] + dx * A00* v1[k] + A01 * v2[k]
			v2[k+1]= v2[k] + dx * A10* v1[k] + A11 * v2[k]
			k = k + 1
		a[i] = v1[qlength-1] * np.exp( 1.0j * zetas[i] * L )
		b[i] = v2[qlength-1] * np.exp( -1.0j * zetas[i] * L )	
		i = i + 1
	return a, b
			


#calc coefficents a and b and their derivatives	(forward difference method)

def DSTloopForwarddiff(double dx,
			  int zetalength,			  
			  int qlength,
			  np.ndarray[DTYPE_t, ndim=1] q,			
			  complex L,
			  np.ndarray[DTYPE_t, ndim=1] zetas):
			  
	cdef np.ndarray[DTYPE_t,ndim=1] a= np.zeros([zetalength], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=1] adiff= np.zeros([zetalength], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=1] b= np.zeros([zetalength], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=1] bdiff= np.zeros([zetalength], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=1] v1= np.zeros([qlength], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=1] v2= np.zeros([qlength], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=1] vdiff1= np.zeros([qlength], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=1] vdiff2= np.zeros([qlength], dtype=DTYPE)	
	cdef int i=0
	cdef int k=0
	cdef complex A00 = 0
	cdef complex A01 = 0
	cdef complex A10 = 0
	cdef complex A11 = 0
	cdef complex Adiff00 = -1.0j * dx
	#cdef complex Adiff01 = 0
	#cdef complex Adiff10 = 0
	cdef complex Adiff11 = 1.0j * dx
	
	
	while i<zetalength:
		k=0
		
		while k < qlength:
			v1[k]=0
			v2[k]=0
			vdiff1[k] = 0
			vdiff2[k] = 0
			k = k+1
		v1[0] = np.exp( -1.0j * zetas[i] *  -L)
		vdiff1[0] = -1.0j * -L * np.exp(-1.0j * zetas[i] * -L)
		k=0
		while k < qlength-1:
		
			A00 = -1.0j * zetas[i]
			A01 = q[k] 
			A10 = -np.conj(q[k])
			A11 = 1.0j * zetas[i]
			
			v1[k+1]= v1[k] + dx * A00* v1[k] + A01 * v2[k]
			v2[k+1]= v2[k] + dx * A10* v1[k] + A11 * v2[k]
			
			vdiff1[k+1] = Adiff00* v1[k]    + A00* vdiff1[k] + A01 * vdiff2[k]
			vdiff2[k+1] = Adiff11* v2[k]    + A10* vdiff1[k] + A11 * vdiff2[k]
			k = k + 1
		a[i] = v1[qlength-1] * np.exp( 1.0j * zetas[i] * L )
		b[i] = v2[qlength-1] * np.exp( -1.0j * zetas[i] * L )
		adiff[i] = ( vdiff1[qlength-1] + 1.0j * L * v1[qlength-1] ) * np.exp(1.0j * zetas[i] * L)
		i = i + 1
	return a, b, adiff
			
			

		

		
		
		
#calc coefficents a and b  (tranfer matrix method)
def DSTloopTransferMatrix(int zetalength,			  
			int qlength,
			np.ndarray[DTYPE_t, ndim=2] UU00,
			np.ndarray[DTYPE_t, ndim=2] UU01,
			np.ndarray[DTYPE_t, ndim=2] UU10,
			np.ndarray[DTYPE_t, ndim=2] UU11):
	cdef int i=0	
	cdef int ii=0
	cdef np.ndarray[DTYPE_t,ndim=2] S= np.zeros([2,2], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=2] tmpS= np.zeros([2,2], dtype=DTYPE)	  
	cdef np.ndarray[DTYPE_t,ndim=1] S00= np.zeros([zetalength], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=1] S10= np.zeros([zetalength], dtype=DTYPE)
	while i<zetalength:
		S[0,0]=1.0
		S[0,1]=0.0
		S[1,0]=0.0
		S[1,1]=1.0		  
		ii = qlength-1
		while ii>=0:		   
			tmpS[0,0] = S[0,0] * UU00[ii,i] + S[0,1] * UU10[ii,i]
			tmpS[0,1] = S[0,0] * UU01[ii,i] + S[0,1] * UU11[ii,i]
			tmpS[1,0] = S[1,0] * UU00[ii,i] + S[1,1] * UU10[ii,i]
			tmpS[1,1] = S[1,0] * UU01[ii,i] + S[1,1] * UU11[ii,i]
			S[0,0]=tmpS[0,0]
			S[0,1]=tmpS[0,1]
			S[1,0]=tmpS[1,0]
			S[1,1]=tmpS[1,1]
			ii=ii-1
		S00[i]=S[0,0]
		S10[i]=S[1,0]
		i = i+1
	return S00, S10
	
#calc coefficents a and b and their derivatives	
def DSTloopTransferMatrixDIFF(int zetalength,		   
			int qlength,
			np.ndarray[DTYPE_t, ndim=2] UU00,
			np.ndarray[DTYPE_t, ndim=2] UU01,
			np.ndarray[DTYPE_t, ndim=2] UU10,
			np.ndarray[DTYPE_t, ndim=2] UU11,			
			np.ndarray[DTYPE_t, ndim=2] UUDASH00,
			np.ndarray[DTYPE_t, ndim=2] UUDASH01,
			np.ndarray[DTYPE_t, ndim=2] UUDASH10,
			np.ndarray[DTYPE_t, ndim=2] UUDASH11):
	cdef int i = 0	
	cdef int ii= 0
	cdef int n = 0
	cdef int m = 0
	cdef np.ndarray[DTYPE_t,ndim=2]    S = np.zeros([4,4], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=2] tmpS = np.zeros([4,4], dtype=DTYPE)	 
	
	cdef np.ndarray[DTYPE_t,ndim=1] S00 = np.zeros([zetalength], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=1] S10 = np.zeros([zetalength], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=1] S20 = np.zeros([zetalength], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=1] S22 = np.zeros([zetalength], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=1] S30 = np.zeros([zetalength], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=1] S32 = np.zeros([zetalength], dtype=DTYPE)
	
	while i<zetalength:		
		# we have to reset S for every zeta[i] ...
		n=0
		while n<4:
			m=0
			while m<4:
				if m!=n:
					S[m,n] = 0.0
				else:
					S[m,n] = 1.0
				m=m+1
			n=n+1			
		#start at the right qvec, move left
		ii = qlength-1
		while ii>=0:  
			#column 1
			tmpS[0,0] =	 S[0,1] * UU10[ii,i]     + S[0,0] * UU00[ii,i]
			tmpS[1,0] =	 S[1,1] * UU10[ii,i]     + S[1,0] * UU00[ii,i]
			tmpS[2,0] =	 S[2,3] * UUDASH10[ii,i] + S[2,2] * UUDASH00[ii,i] + S[2,1] * UU10[ii,i] + S[2,0] * UU00[ii,i]
			tmpS[3,0] =	 S[3,3] * UUDASH10[ii,i] + S[3,2] * UUDASH00[ii,i] + S[3,1] * UU10[ii,i] + S[3,0] * UU00[ii,i]
			
			#column 2
			tmpS[0,1] =	 S[0,1] * UU11[ii,i]     + S[0,0] * UU01[ii,i]
			tmpS[1,1] =	 S[1,1] * UU11[ii,i]     + S[1,0] * UU01[ii,i]
			tmpS[2,1] =	 S[2,3] * UUDASH11[ii,i] + S[2,2] * UUDASH01[ii,i] + S[2,1] * UU11[ii,i] + S[2,0] * UU01[ii,i]
			tmpS[3,1] =	 S[3,3] * UUDASH11[ii,i] + S[3,2] * UUDASH01[ii,i] + S[3,1] * UU11[ii,i] + S[3,0] * UU01[ii,i]
			
			#column 3
			tmpS[0,2] = 0.0
			tmpS[1,2] = 0.0
			tmpS[2,2] = S[2,3] * UU10[ii,i] + S[2,2] * UU00[ii,i]
			tmpS[3,2] = S[3,3] * UU10[ii,i] + S[3,2] * UU00[ii,i]
			
			#column 4
			tmpS[0,3] = 0.0
			tmpS[1,3] = 0.0
			tmpS[2,3] = S[2,3] * UU11[ii,i] + S[2,2] * UU01[ii,i]
			tmpS[3,3] = S[3,3] * UU11[ii,i] + S[3,2] * UU01[ii,i]
			
			# assign tmp values to S
			n=0
			while n<2:
				m=0
				while m<2:				
					S[m,n] = tmpS[m,n]
					m=m+1
				n=n+1			
			ii=ii-1
			
		#assign results to return vecs
		S00[i]=S[0,0]
		S10[i]=S[1,0]
		S20[i]=S[2,0]
		S22[i]=S[2,2]
		S30[i]=S[3,0]
		S32[i]=S[3,2]
		i = i+1
	return S00, S10, S20, S22, S30, S32
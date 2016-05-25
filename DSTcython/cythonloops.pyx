import numpy as np
cimport numpy as np
DTYPE=np.complex128
ctypedef np.complex128_t DTYPE_t


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
		b[i] = v1[qlength-1] * np.exp( -1.0j * zetas[i] * L )	
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
		b[i] = v1[qlength-1] * np.exp( -1.0j * zetas[i] * L )
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
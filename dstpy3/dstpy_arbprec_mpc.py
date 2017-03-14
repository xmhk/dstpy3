import numpy as np
from mpmath import fsum, fprod, fsub,fadd
from mpmath import conj as fconj
from mpmath import exp as fexp
from mpmath import re as mpmre
from mpmath import im as mpmim
from mpmath import mpc
from mpmath import mp

def calc_ab_rungekutta4_vanilla_ap( dx, L, q, zeta, digits=20 ):  
  
    mp.dps=digits
    
    a = []    
    b = []
    lzeta = len(zeta)
    lq    = len(q)
    for i in range(lzeta):        
        v1 = []
        v2 = []
        #calculate the first two elements of v        
        v1.append( fexp( fprod( [-1.0j , zeta[i],  -L]) ))
        v2.append( mpc( 0 + 0.0j) )             
        p0 = [mpc( -1.0j * zeta[i]) ,mpc( q[0]) ,mpc( -fconj(q[0])),mpc( 1.0j * zeta[i])]
        
        v1.append( fsum( [  v1[0] , fadd (  fprod([dx, p0[0] , v1[0]]) , fprod([dx, p0[1] , v2[0]])  ) ]))
        v2.append( fsum(  [ v2[0] , fadd (  fprod([dx, p0[2] , v1[0]]) , fprod([dx, p0[3] , v2[0]])  ) ] ))        
        
        for ii in range(0,lq-2):            
            fpii = [-1.0j * zeta[i], q[ii]    ,-np.conj(q[ii]),   1.0j * zeta[i]]
            fpii1 =[-1.0j * zeta[i], q[ii+1]  ,-np.conj(q[ii+1]), 1.0j * zeta[i]]
            fpii2 =[-1.0j * zeta[i], q[ii+2]  ,-np.conj(q[ii+2]), 1.0j * zeta[i]]
            
            k11 = fadd( fprod( [ fpii[0] , v1[ii]])  , fprod([ fpii[1] , v2[ii] ]))
            k12 = fadd( fprod( [ fpii[2] , v1[ii]])  , fprod([ fpii[3] , v2[ii] ]))
           
            k21 = fadd( fprod([fpii1[0] , fadd(v1[ii] , fprod( [dx,k11])) ])  ,
                        fprod([fpii1[1] , fadd(v2[ii] , fprod( [dx,k12]))]))
            k22 = fadd( fprod([fpii1[2] , fadd(v1[ii] , fprod( [dx,k11])) ])  ,
                        fprod([fpii1[3] , fadd(v2[ii] , fprod( [dx,k12]))]))
            
            
            k31 = fadd( fprod([fpii1[0] , fadd(v1[ii] , fprod( [dx,k21])) ])  ,
                        fprod([fpii1[1] , fadd(v2[ii] , fprod( [dx,k22]))]))
            k32 = fadd( fprod([fpii1[2] , fadd(v1[ii] , fprod( [dx,k21])) ])  ,
                        fprod([fpii1[3] , fadd(v2[ii] , fprod( [dx,k22]))]))
            k41 = fadd( fprod([fpii2[0] , fadd(v1[ii] , fprod( [2,dx,k31])) ])  ,
                        fprod([fpii2[1] , fadd(v2[ii] , fprod( [2,dx,k32]))]))
            k42 = fadd( fprod([fpii2[2] , fadd(v1[ii] , fprod( [2,dx,k31])) ])  ,
                        fprod([fpii2[3] , fadd(v2[ii] , fprod( [2,dx,k32]))]))
            v1.append( fadd(v1[ii] , 
                            fprod( [ 2 , dx , 1./6 , fsum([k11 , fprod([2,k21]) , fprod([2,k31]) , k41])])))
            v2.append( fadd(v2[ii] , 
                            fprod( [ 2 , dx , 1./6 , fsum([k12 , fprod([2,k22]) , fprod([2,k32]) , k42])])))
       
        a.append( fprod( [ v1[-1] , fexp( fprod([  1.0j , zeta[i] , L]))]))
        b.append( fprod( [ v2[-1] , fexp( fprod([ -1.0j , zeta[i] , L]))]))        
        ar = [float( "%.15e"%mpmre(x)) for x in a]
        ai = [float( "%.15e"%mpmim(x)) for x in a]
        
    return np.array( ar) + 1.0j * np.array(ai) , np.array( b)
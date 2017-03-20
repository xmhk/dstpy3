import numpy as np
from mpmath import fsum, fprod, fsub,fadd,fdiv
from mpmath import conj as fconj
from mpmath import exp as fexp
from mpmath import re as mpmre
from mpmath import im as mpmim
from mpmath import mpc
from mpmath import mp

def calc_ab_rungekutta4_vanilla_ap( dx, L, q, zeta, digits=30 ):    
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
                            fprod( [  dx , fdiv(1.0, 3.0) , fsum([k11 , fprod([2,k21]) , fprod([2,k31]) , k41])])))
            v2.append( fadd(v2[ii] , 
                            fprod( [  dx , fdiv(1.0, 3.0), fsum([k12 , fprod([2,k22]) , fprod([2,k32]) , k42])])))
       
        a.append( fprod( [ v1[-1] , fexp( fprod([  1.0j , zeta[i] , L]))]))
        b.append( fprod( [ v2[-1] , fexp( fprod([ -1.0j , zeta[i] , L]))]))        
        ar = [float( "%.15e"%mpmre(x)) for x in a]
        ai = [float( "%.15e"%mpmim(x)) for x in a]
        
        br = [float( "%.15e"%mpmre(x)) for x in b]
        bi = [float( "%.15e"%mpmim(x)) for x in b]
        
    return np.array( ar) + 1.0j * np.array(ai) , np.array( br) + 1.0j * np.array(bi)
    
    
def calc_abdiff_rungekutta4_vanilla_ap( dx, L, q, zeta, digits=30 ):    
    mp.dps=digits 
    # helper functions
    def fpk2(qii, zetai ):
        return  [ mpc( -1.0j * zetai)      , mpc( qii  )           , mpc( 0.0 )             ,     mpc( 0.0 ) ,
                  mpc( -1.0* np.conj(qii)),  mpc(1.0j * zetai )    , mpc( 0 )             ,     mpc( 0 ) ,
                          mpc( -1.0j)       , mpc( 0 )              , mpc(-1.0j * zetai)   ,   mpc(qii)  ,
                          mpc( 0 )          , mpc(1.0j)             , mpc( -1.0*np.conj(qii)) ,  mpc( 1.0j * zetai)]
        
    def matxvec(mat, v):
        res = []        
        for k in range(4):   

            res.append( fsum( [ fprod([mat[0+k*4], v[0]]), fprod([mat[1+k*4], v[1]]),
                                fprod([mat[2+k*4], v[2]]), fprod([mat[3+k*4], v[3]])]))        
        return res[0], res[1], res[2], res[3]
    
    # start integration
    a = [];  b = []; ad = []
    lzeta = len(zeta)
    lq    = len(q)
    for i in range(lzeta):        
        v1 = [];        v2 = [];        v3 = [];        v4 = []        
        v1.append( fexp( fprod( [-1.0j , zeta[i],  -L]) ))
        v2.append( mpc( 0 + 0.0j) )     
        v3.append( fprod( [ 1.0j ,L, fexp( fprod( [1.0j , zeta[i],  L])) ]))
        v4.append( mpc( 0 + 0.0j) )    
        p0 = fpk2( q[0], zeta[i])
        k11, k12, k13, k14 = matxvec(  p0, [v1[0],v2[0],v3[0],v4[0]] )     
        v1.append( fsum( [  v1[0] ,   fprod([dx, k11])]))
        v2.append( fsum( [  v2[0] ,   fprod([dx, k12])]))
        v3.append( fsum( [  v3[0] ,   fprod([dx, k13])]))
        v4.append( fsum( [  v4[0] ,   fprod([dx, k14])]))
        for ii in range(0,lq-2):            
            fpii =  fpk2(q[ii], zeta[i])
            fpii1 = fpk2(q[ii+1], zeta[i])
            fpii2 = fpk2(q[ii+2], zeta[i])            
            
            k11, k12, k13, k14 = matxvec(  fpii, [v1[ii],v2[ii],v3[ii],v4[ii]] )           
       
            k21, k22, k23, k24 = matxvec(  fpii1,[                
                            fadd(v1[ii] , fprod( [dx,k11])),
                            fadd(v2[ii] , fprod( [dx,k12])),
                            fadd(v3[ii] , fprod( [dx,k13])),
                            fadd(v4[ii] , fprod( [dx,k14]))]   )
            
            k31, k32, k33, k34 = matxvec(  fpii1,[                
                            fadd(v1[ii] , fprod( [dx,k21])),
                            fadd(v2[ii] , fprod( [dx,k22])),
                            fadd(v3[ii] , fprod( [dx,k23])),
                            fadd(v4[ii] , fprod( [dx,k24]))]   )            
                        
            k41, k42, k43, k44 = matxvec(  fpii2,[                
                            fadd(v1[ii] , fprod( [2,dx,k31])),
                            fadd(v2[ii] , fprod( [2,dx,k32])),
                            fadd(v3[ii] , fprod( [2,dx,k33])),
                            fadd(v4[ii] , fprod( [2,dx,k34]))]   )
            v1.append( fadd(v1[ii] , 
                            fprod( [  dx , fdiv(1.,3.) , fsum([k11 , fprod([2,k21]) , fprod([2,k31]) , k41])])))
            v2.append( fadd(v2[ii] , 
                            fprod( [  dx , fdiv(1.,3.) , fsum([k12 , fprod([2,k22]) , fprod([2,k32]) , k42])])))
            
            v3.append( fadd(v3[ii] , 
                            fprod( [  dx , fdiv(1.,3.) , fsum([k13 , fprod([2,k23]) , fprod([2,k33]) , k43])])))
            v4.append( fadd(v4[ii] , 
                            fprod( [  dx , fdiv(1.,3.) , fsum([k14 , fprod([2,k24]) , fprod([2,k34]) , k44])])))
          
        a.append( fprod( [ v1[-1] , fexp( fprod([  1.0j , zeta[i] , L]))]))
        b.append( fprod( [ v2[-1] , fexp( fprod([ -1.0j , zeta[i] , L]))]))  
        
        #ad.append( )
        
        ad.append( fprod([ fsum( [v3[-1], fprod([1.0j , L , v1[-1]])]) , fexp( fprod( [1.0j , zeta[i] ,L]))]))
        
        ar = [float( "%.15e"%mpmre(x)) for x in a]
        ai = [float( "%.15e"%mpmim(x)) for x in a]
        br = [float( "%.15e"%mpmre(x)) for x in b]
        bi = [float( "%.15e"%mpmim(x)) for x in b]
        
        adr = [float( "%.15e"%mpmre(x)) for x in ad]
        adi = [float( "%.15e"%mpmim(x)) for x in ad]
        
        
    return np.array( ar) + 1.0j * np.array(ai) ,\
            np.array( br) + 1.0j * np.array(bi),\
            np.array(adr) + 1.0j * np.array(adi),\
            np.zeros(np.shape(np.array(adi))) #bdiff not calculated yet    
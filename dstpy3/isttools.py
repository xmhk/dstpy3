import numpy as np 
from time import time
from dstpy3 import *

def nsol_darboux(zetasl, taus , phasespi, xvec) :
    """        
    N-Soliton darboux transform  
        
    for background, see 
    
    Akhmediev, Nail N. and Nina V. Mitzkevich. 
    Extremely high degree of N-soliton pulse compression in an optical fiber.
    IEEE Journal of Quantum Electronics 27.3 (1991)
    
    M. I. Yousefi, Mansoor I., and F. R. Kschischang.
    Information transmission using the nonlinear
    Fourier transform, Part III: Spectrum modulation. 
    IEEE Transactions on Information Theory 60 (2014)
      
    Nomenclature:
    for auxillary functions / auxillary fields
    $$   V_{i,k} = V(t,\lambda_i,q_k)$$
    $$   q_k = q(t, \lambda_k) $$  
      
    """
    def vnorm(v):
        return  np.abs(v[:,0])**2 + np.abs(v[:,1])**2 

    def vupdate( qk, lamkp1, lamj, vkp1k,vjk):    
        v1 =  (      (lamj - lamkp1)          * np.abs(vkp1k[:,0])**2  \
                   + (lamj - np.conj(lamkp1)) * np.abs(vkp1k[:,1])**2)          *vjk[:,0]\
                + (np.conj(lamkp1)-lamkp1)   * vkp1k[:,0] * np.conj(vkp1k[:,1]) * vjk[:,1]

        v2 =  (    (lamj - np.conj(lamkp1))   * np.abs(vkp1k[:,0])**2 \
                + (lamj - lamkp1)             * np.abs(vkp1k[:,1])**2)  *vjk[:,1]\
                + (np.conj(lamkp1)-lamkp1) * np.conj( vkp1k[:,0]) *  vkp1k[:,1] * vjk[:,0]        
        return np.transpose(np.array([v1/vnorm(vkp1k) ,v2 /vnorm(vkp1k)]))
    
    lams= [z /2.0 for z in zetasl] #lambda scaling 
    N = len(lams)
    md = {}
    q0 = np.zeros( len(xvec)) + 0.0j
    md['q0'] = q0
    # initialize v(lam _i , 0)
    for k in range(1,N+1): 
        # taus introduce center positions
        # phasespi the relative phase (given in units of pi)
        kidx = k-1
        tmpv = np.transpose( np.array([ np.exp(-1.0j * lams[kidx] * (xvec-taus[kidx])  -1.0j *np.pi* phasespi[kidx] ), 
                                        np.exp(+1.0j * lams[kidx] * (xvec-taus[kidx])  +1.0j *np.pi* phasespi[kidx] )]) ) 
        s = "v_%d_%d"%(k,0)        
        md[s] = tmpv
    # iterative dressing 
    for k in range(0,N-1):
        kidx = k-1
        tmpvkp1k = md['v_%d_%d'%(k+1,k)]          
        md["q%d"%(k+1)] = md['q%d'%k] +  2.0j * (2*np.imag(lams[kidx+1])) \
                             *tmpvkp1k[:,0] * np.conjugate(tmpvkp1k[:,1]) /vnorm (tmpvkp1k)
        # eigenvector update with k+1
        for j in range(k+2, N+1): 
            jidx = j-1            
            md['v_%d_%d'%(j, k+1)] =  vupdate( md['q%d'%(k+1)], lams[kidx+1], lams[jidx], \
                                               md["v_%d_%d"%(k+1,k)],md["v_%d_%d"%(j,k)])     
    tmpvkp1k = md['v_%d_%d'%(k+2,k+1)]  
    md['q%d'%(k+2)] = md['q%d'%(k+1)] +  2.0j * (2*np.imag(lams[kidx+2])) \
                             *tmpvkp1k[:,0] * np.conjugate(tmpvkp1k[:,1]) /vnorm (tmpvkp1k)    
    return md['q%d'%(k+2)]


#
#  search algos   ... very experimental ...
#   

def evsearch2(dob, 
            maxite = 30, 
            solitemax = 30,
            methodspec = 'TMF',
            methodite = 'RK4F',            
            absamin = 1e-14 , 
            solreldist = 1e-4,
            reletol = 0.05, 
            solrelom = 0.1,  
            guesses = [], 
            specoffs = 0.5,
            parallel = False,
            verbose=False ):

    def cplxrpr(z):  #shortcut for printing a complex number
        if np.isfinite(z):
            return "%.3e + %.3ej"%(np.real(z), np.imag(z))
        else:
            return "NAN"
            
    def check_newcandidate(CNzm, CNknownlist, CNsolreldist, CNdob):
        CNrv = True 
        for CNlam in CNknownlist:
            if cdistance(CNlam, CNzm, CNdob)<CNsolreldist:
                CNrv = False
        return CNrv

    
    class logobj():    #conbined logging + printing (if verbose)
        def __init__(self, verbose):
            self.log=[]
            self.verbose=verbose
        def appnd(self, strng):
            self.log.append(strng)
            if self.verbose:
                print(strng)
                
    def cdistance(z1, z2, CDDdob ): # return normalized distance
        return np.abs(np.real(z1) - np.real(z2)) / CDDdob.ommax +np.abs(np.imag(z1) - np.imag(z2)) / CDDdob.zetamax    
        
        
    def check_soliton_iteration_result(CSI_zm, CSI_a, CSI_ad, CSI_dob, CSI_solrelom, CSI_solreldist, CSI_actzetamax, CSI_absamin, CSI_knownlist):
        # check the iteration result for soliton candidates (called from inner loop)
        if (not np.isfinite(CSI_a)) or (not np.isfinite(CSI_ad)) or (not np.isfinite(CSI_zm)):
            return 1
        if np.abs(np.real(CSI_zm))>CSI_dob.ommax * CSI_solrelom * 1.05:
            return 2
        if np.imag(CSI_zm)>CSI_actzetamax * 1.05:
            return 3
        if np.imag(CSI_zm)< 0:
            return 4
        if np.abs(CSI_a)>CSI_absamin:
            return 5
        if not check_newcandidate(CSI_zm, CSI_knownlist, CSI_solreldist, CSI_dob):
            return 6
        ### none of the above:
        return 0
  
    LogForThisRun = logobj(verbose)
  
    #
    # Calc spectrum
    #       
    #specoffs: shift ov values a bit -> a pure 0... sometimes gives NaN result    
    ov = ( np.arange(-len(dob.xvec)/2, len(dob.xvec)/2 ) + specoffs) * dob.scaled_dom
    if parallel:
        # in console: ipcluster start -n 4    <--- zum starten notwendig
        import ipyparallel as ipp
        from ipyparallel import interactive      
        c = ipp.Client()
        dview = c[:]           
        ov = ( np.arange(-len(dob.xvec)/2, len(dob.xvec)/2 ) + 0.5) * dob.scaled_dom
        dview.push(  dict( xvec=dob.tvec, feld=dob.field, t0scalepushd=dob.t0scale, methodspecpushd =methodspec))    
        @interactive    # when imported, there is a mess with the namespaces which is prevented by interactive        
        def pcalc(oms):
            import numpy as np         # dirty way for import, but faster than #dview.execute('import numpy as np') above the function ...
            from dstpy3 import DSTObj
            dob = DSTObj(feld, xvec, -1, 1, t0scaleext = t0scalepushd)
            a,b = dob.calc_ab(oms, method = methodspecpushd) 
            return a        
        aspec = dview.map( pcalc , ov, block=True )        
        c.close()
        #LogForThisRun.appnd("%d/%d use parallel for spec"%(0,maxite))
    else:                
        aspec,tmp = dob.calc_ab(ov+0.0j, method=methodspec)  
    
    specissane = True
    #
    # check whether aspec is sane -- not Infs, no NaNs
    #
    if len(np.nonzero(np.invert( np.isfinite(aspec) ))[0])!=0:
        specissane = False    
    spectrum =  np.abs(-1/np.pi * np.log(np.abs(aspec)))    
    spece = np.trapz(spectrum , x=ov) / 2.0 # 2.0: scaling may differ depending on normalization...
    actzetamax = dob.zetamax - spece    
    #
    # Loop over iterations (search for solitons)
    #
    aktite = 0
    solitonsfound = []    
    solitonsnumber = 0    
    #allow guesses to be an numpy-array: re-creating as a list
    guesses = [x for x in guesses]
    #
    # try up to maxitetimes to find solitons, starting from guess or random number
    # 
    while (aktite<maxite) and np.abs(actzetamax) > reletol * dob.zetamax and specissane :   
        aktite += 1
        if len(guesses) == 0:                        
            zm = 2* (np.random.rand() -0.5) * solrelom * dob.ommax  + 1.0j * np.random.rand() * np.abs( actzetamax ) 
            LogForThisRun.appnd("%d/%d  starting with random value: %s"%(aktite, maxite, cplxrpr(zm)))
        else:            
            zm = guesses.pop()+ 0.0j                                  
            LogForThisRun.appnd("%d/%d  checking guess: %s"%(aktite, maxite, cplxrpr(zm)))        
        #
        # try solitemax times to iterate until error or soliton found
        #
        solite = 0; a = 0.0; ad = 1.0 
        while (solite < solitemax):
            solite +=1             
            zm = zm - a/ad       
            a,b,ad,bd = dob.calc_abdiff(zm, method = methodite)
            solitecheck = check_soliton_iteration_result(zm, a, ad, dob, solrelom, solreldist, actzetamax, absamin, solitonsfound)
            if solitecheck == 1:
                LogForThisRun.appnd("     sub %d/%d: numerical problem (a = %s, ad = %s, zm = %s)"%(solite, solitemax,repr(a), repr(ad), repr(zm)))                               
                solite = solitemax                
            if solitecheck == 2:
                solite = solitemax
                LogForThisRun.appnd("     sub %d/%d: real part too big: zm = %s "%(solite, solitemax, cplxrpr(zm)))
            if solitecheck == 3:
                solite = solitemax
                LogForThisRun.appnd("     sub %d/%d: imag part too big: zm = %s "%(solite, solitemax, cplxrpr(zm)))    
            if solitecheck == 4:
                solite = solitemax
                LogForThisRun.appnd("     sub %d/%d: imag part negative: zm = %s "%(solite, solitemax, cplxrpr(zm)))    
            if solitecheck == 6:
                solite = solitemax
                LogForThisRun.appnd("     sub %d/%d: zm = %s already known"%(solite, solitemax, cplxrpr(zm)))
            #
            # keep iterating or soliton found  
            #
            if solitecheck == 5:                
                LogForThisRun.appnd("     sub %d/%d: iterating: zm = %s  |a| = %.1e"%(solite, solitemax, cplxrpr(zm),np.abs(a)))
            if solitecheck == 0:                
                LogForThisRun.appnd("     sub %d/%d: SOLITON FOUND: zm = %s  |a| = %.1e"%(solite, solitemax, cplxrpr(zm),np.abs(a)))
                solitonsfound.append(zm)
                actzetamax = dob.zetamax - spece - np.sum(np.imag(solitonsfound))
                solite = solitemax # exit soliton iteration loop next iteration
                solitonsnumber+=1
                ## symmetry? -> check whether mirrored eigenvalue already known
                zmrconj = -1 * np.real(zm) + 1.0j * np.imag(zm)
                if check_newcandidate(zmrconj, solitonsfound, solreldist, dob):
                    LogForThisRun.appnd("%d/%d      -> adding candidate to list: %s"%( aktite, maxite, cplxrpr(zmrconj)))
                    guesses.append(-1* np.real(zm) + 1.0j * np.imag(zm)) 
    #
    # build return dict
    #
    rd = {}
    rd['aspec'] = aspec     
    rd['spectrum'] = spectrum
    rd['ov'] = ov
    rd['E_spec'] = spece
    rd['E_max'] = dob.zetamax
    rd['E_sol'] = np.sum(np.imag(solitonsfound))
    rd['solitons_number'] = solitonsnumber
    rd['E_diff'] = dob.zetamax - rd['E_sol'] - rd['E_spec']   
    rd['paramliste'] = [['maxite',maxite],
                        ['solitemax',solitemax], 
                        ['absamin', absamin], 
                        ['solreldist', solreldist],
                        ['solrelom',solrelom],
                       ['reletol', reletol], 
                       ['methodite',methodite], 
                       ['methodspec',methodspec],
                       ['specoffs',specoffs],
                       #['guesses',guesses],    #including guesses can give some crazy errors
                       #                         with sio.savemat                                              
                       ['parallel',parallel]   ] 
    #
    # converged or not?
    #
    if specissane:
        if aktite< maxite:        
            rd['converged'] = True        
            LogForThisRun.appnd("CONVERGED ... E_spec=%.1e E_sol=%.1e E_max=%.1e E_diff=%.1e (%.1e rel)"%(rd['E_spec'], rd['E_sol'], 
                                            rd['E_max'], rd['E_diff'], rd['E_diff']/rd['E_max']))
        else:
            rd['converged'] = False        
            LogForThisRun.appnd("NOT CONVERGED ... E_spec=%.1e E_sol=%.1e E_max=%.1e E_diff=%.1e (%.1e rel)\n (aktite = %d)"%(rd['E_spec'], rd['E_sol'], 
                                            rd['E_max'], rd['E_diff'], rd['E_diff']/rd['E_max'], aktite))
    else:
        rd['converged'] = False         
        LogForThisRun.appnd("NOT CONVERGED -- a-Spectrum contains NaN or INF")
    #
    #sort solitons found by imag evals, descending        
    #    
    rd['evals'] = sorted( solitonsfound, key=lambda x:  np.imag(x))   
    rd['log'] = LogForThisRun.log 
    guesses = []
    return rd 
 
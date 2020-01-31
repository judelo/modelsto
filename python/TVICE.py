import numpy as np
import matplotlib.pyplot as plt
import scipy.special

def logerfc(t):
#
# out = logerfc(t): compute an accurate a estimate of log(erfc(t))
#

    out = np.zeros(t.shape) 
    id = t < 20.
    out[id] = np.log(scipy.special.erfc(t[id])) 
    
    c = np.cumprod(np.arange(1,16,2)/2) 
    t2n0 = t[id==0]**2
    t2n = np.copy(t2n0) 
    S = np.ones((t2n.size,)) 
    
    p = -1
    for n in range(8):
        S = S + (p * c[n]) / t2n 
        t2n = t2n*t2n0 
        p = -p 
    out[id==0] = -t2n0 + np.log(S/(t[id==0]*np.sqrt(np.pi))) 
    return out


def logerf2(a,b):
    #
    # usage: out = logerf2(a,b) with a < b
    # computes an accurate estimate of log(erf(b)-erf(a))

    a0 = np.copy(a)
    id = (b < 0)
    a[id] = -b[id]
    b[id] = -a0[id]

    out = np.zeros(a.shape) 
    id1 = (b-a)/(np.abs(a)+np.abs(b)) < 1e-14 
    out[id1] = np.log(2*(b[id1]-a[id1])/np.sqrt(np.pi)) - b[id1]**2 

    id2 = (id1==0) & (a<1) 
    out[id2] = np.log(scipy.special.erf(b[id2])-scipy.special.erf(a[id2])) 

    id3 = (id1==0) & (id2==0) 
    m = logerfc(b[id3]) 
    out[id3] = m + np.log(np.expm1(logerfc(a[id3])-m)) 
    
    return out


##################################    
##### WRITE THE FOLLOWING FUNCTION...
##################################


def tvice(u0,sig,lambd,niter): 
# usage: out = tvice(u0,sigma,lambda,niter) 
# TV-ICE denoising algorithm (vectorized version)

    u = np.copy(u0)
    
    return u

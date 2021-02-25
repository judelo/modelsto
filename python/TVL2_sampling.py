import numpy as np
import matplotlib.pyplot as plt

######
## Sampling TV
######


def deltaTV_pointwise(x,x2,epsilon):
    return  np.abs(np.roll(x2,1,axis=0)- x2) + np.abs(np.roll(x2,1,axis=1) - x2) + np.abs(np.roll(x2,-1,axis=0)- x2) + np.abs(np.roll(x2,-1,axis=1) - x2)  -np.abs(np.roll(x,1,axis=0)- x) -np.abs(np.roll(x,1,axis=1) - x) - np.abs(np.roll(x,-1,axis=0)- x) -np.abs(np.roll(x,-1,axis=1) - x) + epsilon*(x2**2-x**2)

def transition_all_TV(x,alpha,beta,epsilon): 
    nr,nc = x.shape
    # draw which grid of 1 out of 4 independent pixels we will use  
    i=np.random.randint(2) 
    j=np.random.randint(2)
    # compute the perturbation delta everywhere but apply it only on the grid 
    delta = 2*np.random.rand(nr,nc)-1  
    x2 = np.copy(x)
    x2[i:nr:2,j:nc:2] = x[i:nr:2,j:nc:2] + alpha*delta[i:nr:2,j:nc:2]
     # decide for each point if we apply the transition or not
    diff = np.exp(-beta*deltaTV_pointwise(x,x2,epsilon) ) - np.random.rand(nr,nc)
    x2[diff<0] = x[diff<0]
    return x2

def metropolis_TV1(x,alpha,Npas,beta,epsilon):
    # Metropolis algorithm
    for t in range(Npas):
        x=transition_all_TV(x,alpha,beta,epsilon)
    return x



######
## Sampling TVL2
######

def deltaTVL2_pointwise(x,x2,ub,alpha,sigma,lambd):
    return  0.5*((x2-ub)**2- (x-ub)**2)/(sigma**2) + lambd*( np.abs(np.roll(x2,1,axis=0)- x2)+np.abs(np.roll(x2,1,axis=1) - x2)+np.abs(np.roll(x2,-1,axis=0)- x2)+np.abs(np.roll(x2,-1,axis=1) - x2) -np.abs(np.roll(x,1,axis=0)- x) -np.abs(np.roll(x,1,axis=1) - x) -np.abs(np.roll(x,-1,axis=0)- x) -np.abs(np.roll(x,-1,axis=1) - x)) 
 

def transition_all_TVL2(x,alpha,ub,sigma,lambd):
    nr,nc = x.shape
    # draw which grid of 1 out of 4 independent pixels we will use  
    i=np.random.randint(2) 
    j=np.random.randint(2)
    # compute the perturbation delta everywhere but apply it only on the grid 
    delta = 2*np.random.rand(nr,nc)-1
    x2 = np.copy(x)
    x2[i:nr:2,j:nc:2] = x[i:nr:2,j:nc:2] + alpha*delta[i:nr:2,j:nc:2]
    # decide for each point if we apply the transition or not
    diff = np.exp(-deltaTVL2_pointwise(x,x2,ub,alpha,sigma,lambd)) - np.random.rand(nr,nc)
    x2[diff<=0] = x[diff<=0]
    return x2

def metropolis_TVL2(x,alpha,Npas,ub,sigma,lambd):
    t_burnin = int(Npas/10)
    nr,nc = x.shape
    xmean = np.zeros((nr,nc))
    x2    = np.zeros((nr,nc))
    # Metropolis algorithm
    for t in range(Npas):
        x=transition_all_TVL2(x,alpha,ub,sigma,lambd)
        # update the mean
        if t >= t_burnin:
            tb = t - t_burnin
            xmean = tb/(tb+1)*xmean + 1/(tb+1)*x
            x2    = tb/(tb+1)*x2 + 1/(tb+1)*x**2
    stdx = np.sqrt(x2 - xmean**2)
    return x,xmean,stdx


######
## Sampling TV1L2 with diagonal A 
######


def deltaTVL2A_pointwise(mask,x,x2,ub,alpha,sigma,lambd):
    return  0.5*((mask*x2-ub)**2- (mask*x-ub)**2)/(sigma**2) + lambd*(
        np.abs(np.roll(x2,1,axis=0)- x2)+np.abs(np.roll(x2,1,axis=1) - x2)+np.abs(np.roll(x2,-1,axis=0)- x2)+np.abs(np.roll(x2,-1,axis=1) - x2) -np.abs(np.roll(x,1,axis=0)- x) -np.abs(np.roll(x,1,axis=1) - x) -np.abs(np.roll(x,-1,axis=0)- x) -np.abs(np.roll(x,-1,axis=1) - x)) 
 

def transition_all_TVL2A(x,mask,alpha,ub,sigma,lambd):
    nr,nc = x.shape
    # draw which grid of 1 out of 4 independent pixels we will use  
    i=np.random.randint(2) 
    j=np.random.randint(2)
    # compute the perturbation delta everywhere but apply it only on the grid 
    delta = 2*np.random.rand(nr,nc)-1
    x2 = np.copy(x)
    x2[i:nr:2,j:nc:2] = x[i:nr:2,j:nc:2] + alpha*delta[i:nr:2,j:nc:2]
    # decide for each point if we apply the transition or not
    diff = np.exp(-deltaTVL2A_pointwise(mask,x,x2,ub,alpha,sigma,lambd)) - np.random.rand(nr,nc)
    x2[diff<=0] = x[diff<=0]
    return x2

def metropolis_TVL2A(x,mask,alpha,Npas,ub,sigma,lambd):
    t_burnin = int(Npas/10)
    nr,nc = x.shape
    xmean = np.zeros((nr,nc))
    x2    = np.zeros((nr,nc))
    # Metropolis algorithm
    for t in range(Npas):
        x=transition_all_TVL2A(x,mask,alpha,ub,sigma,lambd)
        # update the mean
        if t >= t_burnin:
            tb = t - t_burnin
            xmean = tb/(tb+1)*xmean + 1/(tb+1)*x
            x2    = tb/(tb+1)*x2 + 1/(tb+1)*x**2
    stdx = np.sqrt(x2 - xmean**2)
    return x,xmean,stdx

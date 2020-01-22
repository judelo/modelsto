import numpy as np
import matplotlib.pyplot as plt

def div(cx,cy):
    #cy and cy are coordonates of a vector field.
    #the function computes the discrete divergence of this vector field

    nr,nc=cx.shape

    ddx=np.zeros((nr,nc))
    ddy=np.zeros((nr,nc))

    ddx[:,1:-1]=cx[:,1:-1]-cx[:,0:-2]
    ddx[:,0]=cx[:,0]
    ddx[:,-1]=-cx[:,-2]
  
    ddy[1:-1,:]=cy[1:-1,:]-cy[0:-2,:]
    ddy[0,:]=cy[0,:]
    ddy[-1,:]=-cy[-2,:]
 
    d=ddx+ddy

    return d

def grad(im):
    #computes the gradient of the image 'im'
    # image size 
    nr,nc=im.shape
  
    gx = im[:,1:]-im[:,0:-1]
    gx = np.block([gx,np.zeros((nr,1))])

    gy =im[1:,:]-im[0:-1,:]
    gy=np.block([[gy],[np.zeros((1,nc))]])
    return gx,gy



def chambolle_pock_prox_TV(ub,lambd,niter):
    # the function solves the problem
    # argmin_u   1/2 \| u - ub\|^2 + \lambda TV(u)
    # with TV(u) = \sum_i \|\nabla u (i) \|_2
    # uses niter iterations of Chambolle-Pock
    
    nr,nc = ub.shape
    ut = np.copy(ub)
    ubar = np.copy(ut)
    p = np.zeros((nr,nc,2))
    tau   = 0.9/np.sqrt(8*lambd**2)
    sigma = 0.9/np.sqrt(8*lambd**2) 
    theta = 1
    
    for k in range(niter):
        # calcul de proxF
        ux,uy  = grad(ubar)
        p = p + sigma*lambd*np.stack((ux,uy),axis=2)
        normep = np.sqrt(p[:,:,0]**2+p[:,:,1]**2)
        normep = normep*(normep>1) + (normep<=1)
        p[:,:,0] = p[:,:,0]/normep
        p[:,:,1] = p[:,:,1]/normep

        # calcul de proxG
        d=div(p[:,:,0],p[:,:,1])
        #TVL2
        unew = 1/(1+tau)*(ut+tau*lambd*d+tau*ub) 
        
        #extragradient step
        ubar = unew+theta*(unew-ut)
        ut = unew
    return ut

def chambolle_pock_prox_TV1(ub,lambd,niter):
    # the function solves the problem
    # argmin_u   1/2 \| u - ub\|^2 + \lambda TV(u)
    # with TV(u) = \sum_i \|\nabla u (i) \|_1
    # uses niter iterations of Chambolle-Pock
    
    nr,nc = ub.shape
    ut = np.copy(ub)
    ubar = np.copy(ut)
    p = np.zeros((nr,nc,2))
    tau   = 0.9/np.sqrt(8*lambd**2)
    sigma = 0.9/np.sqrt(8*lambd**2) 
    theta = 1
    
    for k in range(niter):
        # calcul de proxF
        ux,uy  = grad(ubar)
        p = p + sigma*lambd*np.stack((ux,uy),axis=2)
        normep = np.abs(p[:,:,0])+np.abs(p[:,:,1])
        normep = normep*(normep>1) + (normep<=1)
        p[:,:,0] = p[:,:,0]/normep
        p[:,:,1] = p[:,:,1]/normep

        # calcul de proxG
        d=div(p[:,:,0],p[:,:,1])
        #TVL2
        unew = 1/(1+tau)*(ut+tau*lambd*d+tau*ub) 
        
        #extragradient step
        ubar = unew+theta*(unew-ut)
        ut = unew
    return ut



def convol(a,b):
    return np.real(np.fft.ifft2(np.fft.fft2(a)*np.fft.fft2(b))) 

def IdplustauATA_inv(x,tau,h): 
    return np.real(np.fft.ifft2(np.fft.fft2(x)/(1+tau*np.abs(np.fft.fft2(h))**2)))

def chambolle_pock_deblurring_TV(ub,h,lambd,niter):
    # the function solves the problem
    # argmin_u   1/2 \| Au - ub\|^2 + \lambda TV(u)
    # with TV(u) = \sum_i \|\nabla u (i) \|_2
    # and A = blur given by a kernel h
    # uses niter iterations of Chambolle-Pock

    nr,nc = ub.shape
    ut = np.copy(ub)

    p = np.zeros((nr,nc,2))
    tau   = 0.9/np.sqrt(8*lambd**2)
    sigma = 0.9/np.sqrt(8*lambd**2) 
    theta = 1
    ubar = np.copy(ut)
    
    h_fft = np.fft.fft2(h)
    hc_fft = np.conj(h_fft)
    hc = np.fft.ifft2(hc_fft)


    for k in range(niter):
        
        # subgradient step on p 
        ux,uy  = grad(ubar)
        p = p + sigma*lambd*np.stack((ux,uy),axis=2)
        normep = np.sqrt(p[:,:,0]**2+p[:,:,1]**2)
        normep = normep*(normep>1) + (normep<=1)
        p[:,:,0] = p[:,:,0]/normep
        p[:,:,1] = p[:,:,1]/normep
        
    # subgradient step on u
        d=div(p[:,:,0],p[:,:,1])
        unew = (ut+tau*lambd*d+tau*convol(ub, hc)) 
        unew = IdplustauATA_inv(unew, tau,h)
    
    #extragradient step on u 
        ubar = unew+theta*(unew-ut)
        ut = unew
        
    return ut

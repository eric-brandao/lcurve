import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy import linalg # for svd
from scipy import optimize

def curvature(lambd, sig, beta, xi):
    '''
    computes the NEGATIVE of the curvature. Adapted from Per Christian Hansen, DTU Compute, October 27, 2010.
    '''
    # Initialization.
    phi = np.zeros(lambd.shape)
    dphi = np.zeros(lambd.shape)
    psi = np.zeros(lambd.shape)
    dpsi = np.zeros(lambd.shape)
    eta = np.zeros(lambd.shape)
    rho = np.zeros(lambd.shape)
    if len(beta) > len(sig): # A possible least squares residual.
        LS = True
        rhoLS2 = beta[-1] ** 2
        beta = beta[0:-2]
    else:
        LS = False
    # Compute some intermediate quantities.
    for jl, lam in enumerate(lambd):
        f  = np.divide((sig ** 2), (sig ** 2 + lam ** 2)) # ok
        cf = 1 - f # ok
        eta[jl] = np.linalg.norm(f * xi) # ok
        rho[jl] = np.linalg.norm(cf * beta)
        f1 = -2 * f * cf / lam 
        f2 = -f1 * (3 - 4*f)/lam
        phi[jl]  = np.sum(f*f1*np.abs(xi)**2) #ok
        psi[jl] = np.sum(cf*f1*np.abs(beta)**2)
        dphi[jl] = np.sum((f1**2 + f*f2)*np.abs(xi)**2)
        dpsi[jl] = np.sum((-f1**2 + cf*f2)*np.abs(beta)**2) #ok

    if LS: # Take care of a possible least squares residual.
        rho = np.sqrt(rho ** 2 + rhoLS2)

    # Now compute the first and second derivatives of eta and rho
    # with respect to lambda;
    deta  =  np.divide(phi, eta) #ok
    drho  = -np.divide(psi, rho)
    ddeta =  np.divide(dphi, eta) - deta * np.divide(deta, eta)
    ddrho = -np.divide(dpsi, rho) - drho * np.divide(drho, rho)

    # Convert to derivatives of log(eta) and log(rho).
    dlogeta  = np.divide(deta, eta)
    dlogrho  = np.divide(drho, rho)
    ddlogeta = np.divide(ddeta, eta) - (dlogeta)**2
    ddlogrho = np.divide(ddrho, rho) - (dlogrho)**2
    # curvature.
    curv = - np.divide((dlogrho * ddlogeta - ddlogrho * dlogeta),
        (dlogrho**2 + dlogeta**2)**(1.5))
    return curv

def l_corner(rho,eta,reg_param,u,sig,bm):
    '''
    computes the corner of the L-curve.
    Inputs:
        rho, eta, reg_param - computed in l_curve function
        u left side matrix computed from svd (size: Nm x Nm) - Nm is the number of measurement points
        sig is the singular value vector of A
        bm is the measured results
    A is of Nm x Nu, where Nm are the number of measurements and Nu the number of unknowns
    Adapted from Per Christian Hansen, DTU Compute, October 27, 2010.
    '''
    # Set threshold for skipping very small singular values in the
    # analysis of a discrete L-curve.
    s_thr = np.finfo(float).eps # Neglect singular values less than s_thr.
    # Set default parameters for treatment of discrete L-curve.
    deg   = 2  # Degree of local smooting polynomial.
    q     = 2  # Half-width of local smoothing interval.
    order = 4  # Order of fitting 2-D spline curve.
    # Initialization.
    if (len(rho) < order):
        print('I will fail. Too few data points for L-curve analysis')
    Nm, Nu = u.shape
    p = sig.shape
    # if (nargout > 0), locate = 1; else locate = 0; end
    beta = (np.conjugate(u)) @ bm
    beta = np.reshape(beta[0:int(p[0])], beta.shape[0])
    b0 = (bm - (beta.T @ u).T)#u @ beta
    # s = sig
    xi = np.divide(beta[0:int(p[0])], sig)
    # Call curvature calculator
    curv = curvature(reg_param, sig, beta, xi) # ok
    # Minimize 1
    # reg_c = optimize.fmin(curvature, 0.0, args = (sig, beta, xi), full_output=False, disp=False)
    # Minimize 1
    curv_id = np.argmin(curv)
    x1 = reg_param[int(np.amin([curv_id, len(curv)]))]
    x2 = reg_param[int(np.amax([curv_id, 0]))]
    reg_c = optimize.fminbound(curvature, x1, x2, args = (sig, beta, xi), full_output=False, disp=False)
    kappa_max = - curvature(reg_c, sig, beta, xi) # Maximum curvature.
    if kappa_max < 0:
        lr = len(rho)
        reg_c = reg_param[lr]
        rho_c = rho[lr]
        eta_c = eta[lr]
    else:
        f = np.divide((sig**2), (sig**2 + reg_c**2))
        eta_c = np.linalg.norm(f * xi)
        rho_c = np.linalg.norm((1-f) * beta[0:len(f)])
        if Nm > Nu:
            rho_c = np.sqrt(rho_c ** 2 + np.linalg.norm(b0)**2)
    return reg_c

# reg_param[int(np.amin([curv_id+1, len(curv)])])

    # reg_c = fminbnd('lcfun',...
    # reg_param(min(gi+1,length(g))),reg_param(max(gi-1,1)),...
    # optimset('Display','off'),s,beta,xi); % Minimizer.
    

def csvd(A):
    '''
    computes the svd based on the size of A.
    Input:
        A is of Nm x Nu, where Nm are the number of measurements and Nu the number of unknowns
    Adapted from Per Christian Hansen, DTU Compute, October 27, 2010.
    '''
    Nm, Nu = A.shape
    if Nm >= Nu: # more measurements than unknowns
        u, sig, v = linalg.svd(A, full_matrices=False)
    else:
        v, sig, u = linalg.svd(np.conjugate(A.T), full_matrices=False)
    return u, sig, v

def l_cuve(u, sig, bm, plotit = False):
    '''
    Plot the L-curve and find its "corner".
    Adapted from Per Christian Hansen, DTU Compute, October 27, 2010.
    Inputs:
        u: left side matrix computed from svd (size: Nm x Nm) - Nm is the number of measurement points
        sig: singular values computed from svd (size: Nm x 1)
        bm: your measurement vector (size: Nm x 1)
    '''
    # Set defaults.
    npoints = 200  # Number of points on the L-curve
    smin_ratio = 16*np.finfo(float).eps  # Smallest regularization parameter.
    # Initialization.
    Nm, Nu = u.shape
    p = sig.shape
    # if (nargout > 0), locate = 1; else locate = 0; end
    beta = np.conjugate(u) @ bm
    beta2 = np.linalg.norm(bm) ** 2 - np.linalg.norm(beta)**2
    # if ps == 1:
    s = sig
    beta = np.reshape(beta[0:int(p[0])], beta.shape[0])
    xi = np.divide(beta[0:int(p[0])],s)
    xi[np.isinf(xi)] = 0

    eta = np.zeros((npoints,1))
    # print('eta {}'.format(eta.shape))
    rho = np.zeros((npoints,1)) #eta
    reg_param = np.zeros((npoints,1))
    s2 = s ** 2
    reg_param[-1] = np.amax([s[-1], s[0]*smin_ratio])
    ratio = (s[0]/reg_param[-1]) ** (1/(npoints-1))
    # print('ratio {}'.format(ratio))
    for i in np.arange(start=npoints-2, step=-1, stop = -1):
        reg_param[i] = ratio*reg_param[i+1]
    for i in np.arange(start=0, step=1, stop = npoints):
        f = s2 / (s2 + reg_param[i] ** 2)
        eta[i] = np.linalg.norm(f * xi)
        rho[i] = np.linalg.norm((1-f) * beta[:int(p[0])])
    if (Nm > Nu and beta2 > 0):
        rho = np.sqrt(rho ** 2 + beta2)
    # want to plot the L curve?
    if plotit:
        plt.loglog(rho, eta)
        plt.xlabel('Residual norm ||Ax - b||')
        plt.ylabel('Solution norm ||x||')
        plt.show()
    # Compute the corner of the L-curve (optimal regularization parameter)
    lam_opt = l_corner(rho,eta,reg_param,u,sig,bm)
    return lam_opt
    # print('eta {}'.format(eta[0]))
    # print('xi: {}'.format(xi))
    # print('xi shape {}'.format(xi.shape))
    # print('s shape {}'.format(s.shape))

    
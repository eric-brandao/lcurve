# import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy import linalg # for svd
from lcurve_functions import csvd, l_cuve

#%%
mat = scio.loadmat('c0.mat')
c0 = np.array(mat['c0'])[0][0]

mat = scio.loadmat('freq.mat')
ind_f = 1
freq = np.array(mat['freq_m'])[0][ind_f]
print('Running test for freq: {}'.format(freq))
k0 = 2 * np.pi * freq / c0

mat = scio.loadmat('receivers.mat')
receivers = np.array(mat['receivers_m'])

mat = scio.loadmat('directions.mat')
directions = np.array(mat['dir_m'])
k_vec = k0 * directions

mat = scio.loadmat('pm.mat')
pm_all = np.array(mat['p_m'])
pm = pm_all[:,:,ind_f]
pm = pm.T

h_mtx = np.exp(1j*receivers @ k_vec.T)
print('the shape of H is {}'.format(h_mtx.shape))

##### %% SVD ####
u, sig, v = csvd(h_mtx)
# u, s, v = linalg.svd(h_mtx, full_matrices=False) #compute SVD without 0 singular values
lam_opt = l_cuve(u, sig, pm, plotit=False)
print('Optmal regu par is: {}'.format(lam_opt))

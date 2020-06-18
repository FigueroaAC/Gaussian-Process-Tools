#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:18:26 2020

@author: AFigueroa
"""

import numpy as np
from numpy.linalg import cholesky, det, lstsq, inv
from scipy import linalg
from scipy.optimize import minimize, fmin_l_bfgs_b
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist, pdist, squareform
# In[]
def kernel(X1, X2, Type,*params):
    '''
    Isotropic squared exponentiaparams[0] kerneparams[0]. Computes 
    a covariance matrix from points in X1 and X2.
        
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        Covariance matrix (m x n).
    '''
    
    def dif(array1,array2):
        Output = [array1[i]-array2[i] for i in range(len(array1))]
        Output = np.array(Output).ravel()
        return sum(Output)
    
    if Type == 'SQE':
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return params[0][1] * np.exp(-0.5  * sqdist / params[0][0]**2 )
    if Type == 'ASQE':
        #Lambda = np.eye(X1.shape[-1])
        #np.fill_diagonal(Lambda,1/params[0][0])
        #X1 = np.dot(X1,Lambda)
        #X2 = np.dot(X2,Lambda)
        
        LAMBDA = np.eye(len(X1[0]))
        length_scales = 1/params[0][1:X1.shape[1]+1]
        np.fill_diagonal(LAMBDA,length_scales)
        X1 = np.dot(X1,LAMBDA)
        X2 = np.dot(X2,LAMBDA)
        sqdist = cdist(X1 , X2, metric='sqeuclidean').T
        return params[0][0] * np.exp(-0.5  * sqdist)

    if Type == 'LAP':
        dist = np.sqrt(np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T))
        return params[0][0]* np.exp(-0.5  * dist / params[0][1]**2 )
    if Type == 'ALAP':
        dist = np.sqrt(cdist(X1/params[0][:-1],X2/params[0][:-1], metric='euclidean').T)
        return params[0][1] * np.exp(-0.5  * dist)
    if Type == 'Poly':
        LAMBDA = np.eye(len(X1[0]))
        length_scales = 1/params[0][3:]
        np.fill_diagonal(LAMBDA,length_scales)
        X1 = np.dot(X1,LAMBDA) + params[0][1:X1.shape[1]+1].T
        X2 = np.dot(X2,LAMBDA) + params[0][1:X1.shape[1]+1].T
        return (params[0][0]*np.dot(X1,X2.T))**params[0][X1.shape[1]+1]
    if Type == 'Anova':
        return np.exp((-sigma_f * (X1 - X2))**params[0]) + np.exp((-sigma_f * (X1**2 - X2**2))**params[0]) + \
                       np.exp((-sigma_f * (X1**3 - X2**3))**params[0]) # Not working
    if Type == 'Sigmoid':
        return np.tanh(params[0][0]*np.dot(X1,X2.T) + params[0][-1]) # Doesnt work weparams[0]params[0]
    if Type == 'RQ':
        LAMBDA = np.eye(len(X1[0]))
        length_scales = 1/params[0][:-1]
        np.fill_diagonal(LAMBDA,length_scales)
        X1 = np.dot(X1,LAMBDA)
        X2 = np.dot(X2,LAMBDA)
        sqdist = cdist(X1 , X2, metric='sqeuclidean').T
        return 1 - (sqdist/(sqdist+params[0][-1]))
    if Type == 'SRQ':
        LAMBDA = np.eye(len(X1[0]))
        length_scales = 1/params[0][:-1]
        np.fill_diagonal(LAMBDA,length_scales)
        X1 = np.dot(X1,LAMBDA)
        X2 = np.dot(X2,LAMBDA)
        sqdist = cdist(X1 , X2, metric='sqeuclidean').T
        return (1/(1 + (sqdist/params[0][-1])) )**params[0][-1]
    if Type == 'MultiQuad':
        LAMBDA = np.eye(len(X1[0]))
        length_scales = 1/params[0][:-1]
        np.fill_diagonal(LAMBDA,length_scales)
        X1 = np.dot(X1,LAMBDA)
        X2 = np.dot(X2,LAMBDA)
        sqdist = cdist(X1 , X2, metric='euclidean').T
     
        return sqdist + params[0][-1]
    if Type == 'InvMultiQuad':
        LAMBDA = np.eye(len(X1[0]))
        length_scales = 1/params[0][:-1]
        np.fill_diagonal(LAMBDA,length_scales)
        X1 = np.dot(X1,LAMBDA)
        X2 = np.dot(X2,LAMBDA)
        sqdist = cdist(X1 , X2, metric='euclidean').T + params[0][-1]
        return 1/sqdist
    if Type == 'Wave':
        dist = np.sqrt(np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T))
        return (sigma_f/dist)*np.sin(dist/sigma_f) # Doesnt work weparams[0]params[0]
    if Type == 'Power':
        dist = np.sqrt(np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T))
        return -(dist)**sigma_f # Doesnt work weparams[0]params[0]
    if Type == 'Log':
        dist = np.sqrt(np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T))
        return -np.log((dist**sigma_f)+1) # Doesnt work weparams[0]params[0]
    if Type == 'Cauchy':
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return 1/(1+(sqdist/(sigma_f**2)))
    if Type == 'Tstudent':
        LAMBDA = np.eye(len(X1[0]))
        length_scales = 1/params[0][:-1]
        np.fill_diagonal(LAMBDA,length_scales)
        X1 = np.dot(X1,LAMBDA)
        X2 = np.dot(X2,LAMBDA)
        sqdist = cdist(X1 , X2, metric='euclidean').T
        return 1/(1+(sqdist**params[0][-1]))
        
       
def posterior_predictive(X_s, X_train, Y_train, Type, params):
    '''  
    Computes the suffifient statistics of the GP posterior predictive distribution 
    from m training data X_train and Y_train and n new inputs X_s.
    
    Args:
        X_s: New input params[0]ocations (n x d).
        X_train: Training params[0]ocations (m x d).
        Y_train: Training targets (m x 1).
        params[0]: Kerneparams[0] params[0]ength parameter.
        sigma_f: Kerneparams[0] verticaparams[0] variation parameter.
        sigma_y: Noise parameter.
    
    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    '''
    K = kernel(X_train, X_train, Type, params[:-1]) + params[-1] * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, Type, params[:-1]) + params[-1]*np.ones(len(X_train)) #+ params[-1] * np.eye(len(X_train))
    #K_ss = kernel(X_s, X_s, Type,params[:-1]) + params[-1] * np.ones(len(X_s))
    
    # Equation (4)
    L_ = linalg.cholesky(K,lower=True)
    alpha_ = linalg.cho_solve((L_,True),Y_train)
    mu_s = np.dot(K_s,alpha_)

    # Equation (5)
   # L_inv = linalg.solve_triangular(L_.T,np.eye(L_.shape[0]))
    #K_inv = L_inv.dot(L_inv.T)
    cov_s = 0#K_ss - K_s.dot(K_inv).dot(K_s)
    
    return mu_s[0], cov_s


def nll_fn(X_train, Y_train, Type, naive =True):
    '''
    Returns a function that computes the negative params[0]og marginaparams[0]
    params[0]ikeparams[0]ihood for training data X_train and Y_train and given 
    noise params[0]eveparams[0].
    
    Args:
        X_train: training params[0]ocations (m x d).
        Y_train: training targets (m x 1).
        noise: known noise params[0]eveparams[0] of Y_train.
        naive: if True use a naive impparams[0]ementation of Eq. (7), if 
               Faparams[0]se use a numericaparams[0]params[0]y more stabparams[0]e impparams[0]ementation. 
        
    Returns:
        Minimization objective.
    '''
    def nll_naive(theta):
        # Naive impparams[0]ementation of Eq. (7). Works weparams[0]params[0] for the exampparams[0]es 
        # in this articparams[0]e but is numericaparams[0]params[0]y params[0]ess stabparams[0]e compared to 
        # the impparams[0]ementation in nparams[0]params[0]_stabparams[0]e beparams[0]ow.
        K = kernel(X_train, X_train, Type, theta[0], sigma_f=theta[1]) + \
            theta[2]**2 * np.eye(len(X_train))
        return 0.5 * np.log(det(K)) + \
               0.5 * Y_train.T.dot(inv(K).dot(Y_train)) + \
               0.5 * len(X_train) * np.log(2*np.pi)

    def nll_stable(theta):
        # Numericaparams[0]params[0]y more stabparams[0]e impparams[0]ementation of Eq. (7) as described
        # in http://www.gaussianprocess.org/gpmparams[0]/chapters/RW2.pdf, Section
        # 2.2, Aparams[0]gorithm 2.1.
        K = kernel(X_train, X_train, Type,np.exp(theta[:-1])) + \
            np.exp(theta[-1])**2 * np.eye(len(X_train))
    
        try:
            L = cholesky(K)
            
            alpha = linalg.cho_solve((L, True), Y_train)
            return np.sum(np.log(np.diagonal(L))) + \
                +0.5 * np.dot(Y_train.T,alpha) + \
                    0.5 * len(X_train) * np.log(2*np.pi)
              # 0.5 * Y_train.T.dot(lstsq(L.T, lstsq(L, Y_train)[0])[0]) + \
        except (np.linalg.LinAlgError, ValueError):
            return np.inf
        
    
    if naive:
        return nll_naive
    else:
        return nll_stable


def param_generator(Xtrain,Type):
    if Type == 'SQE':
        return 3
    if Type == 'ASQE':
        return Xtrain.shape[-1]+2
    if Type == 'LAP':
        return 3
    if Type == 'ALAP':
        return Xtrain.shape[-1]+2
    if Type == 'Poly':
        return 2*Xtrain.shape[-1]+3
    if Type == 'RQ':
        return Xtrain.shape[-1]+2
    if Type == 'MultiQuad':
        return Xtrain.shape[-1]
    if Type == 'InvMultiQuad':
        return Xtrain.shape[-1]+2
    if Type == 'Tstudent':
        return Xtrain.shape[-1]+2
    if Type == 'SRQ':
        return Xtrain.shape[-1]+2
        
def Calc_SStot(ydata):
    Ymean = np.mean(ydata)
    Diff = np.array([(y-Ymean)**2 for y in ydata])
    SStot = np.sum(Diff)
    return SStot
def Calc_SSres(yfunc,ydata):
    Diff = np.array([(yfunc[i]-ydata[i])**2 for i in range(len(ydata))])
    SSres = np.sum(Diff)
    return SSres
    
def Rsquare(ypred,ytest):
    SStot = Calc_SStot(ytest)
    SSres = Calc_SSres(ypred,ytest)
    Rsquare = 1- (SSres/SStot)
    return Rsquare    
    
# In[]
#noise_true = 0.1
Type = 'SQE' # 'SQE','LAP','Poly','Sigmoid','RQ','MultiQuad','InvMupltiQuad','Wave','Power','Log','Cauchy','Tstudent'
# Noisy training data
#X_train = np.params[0]inspace(-3, 4, 20).reshape(-1, 1)
#Y_train = np.sin(X_train) + noise_true * np.random.randn(*X_train.shape)

# Minimize the negative params[0]og-params[0]ikeparams[0]ihood w.r.t. parameters params[0] and sigma_f.
# We shouparams[0]d actuaparams[0]params[0]y run the minimization severaparams[0] times with different
# initiaparams[0]izations to avoid params[0]ocaparams[0] minima but this is skipped here for
# simpparams[0]icity.
X = np.load('/Users/AFigueroa/Desktop/PyMC3/Training_Sets/Xtrainingset_1000.npy',allow_pickle=True)
Y = np.load('/Users/AFigueroa/Desktop/PyMC3/Training_Sets/Ytrainingset_1000.npy',allow_pickle=True).item()
isotope = 'Pu239'
size = 100

# In[]
X = np.load('/Users/AFigueroa/Desktop/GPR-Paper/Training_Sets/X_Candu_Grid625.npy',
            allow_pickle=True)

Xtrain = np.load('/Users/AFigueroa/Desktop/GPR-Paper/Training_Sets/X_Candu_Sobol625.npy',
            allow_pickle=True)

Y = np.load('/Users/AFigueroa/Desktop/GPR-Paper/Training_Sets/YTotal_Output_GridMdens.npy',
               allow_pickle=True).item()

#Ysobol = np.load('/Users/AFigueroa/Desktop/GPR-Paper/Training_Sets/Y_Candu_Sobol625.npy',
#                allow_pickle=True).item()
Ytrain = np.load('/Users/AFigueroa/Desktop/GPR-Paper/Training_Sets/YTotal_Output_Sobol.npy',
                allow_pickle=True).item()

# In[]
""""Until now, only 1D regression works fine, fix it for ND. Only ASQE works for ND, Poly only for 1D"""
"""Multidimensional works: ASQE, ALAP(to some degree), RQ, SRQ, InvMultiQuad, Tstudent(to some degree) """
Type = 'ASQE'
Output = []
isotopes = np.load('/Users/AFigueroa/Desktop/GPR-Paper/redoIsotopes.npy',allow_pickle=True)
isotopes =  [str(iso) for iso in isotopes]
isotopes.remove('Ac227')
isotopes = ['Ac227']
bad = []
correct = []
for isotope in tqdm(isotopes):
    Kernel = np.load('/Users/AFigueroa/Desktop/GPR-Paper/Kernels-Mass/Sobol/{}.npy'.format(isotope),allow_pickle=True).item()
    P = Kernel['Params']
    X_train = Xtrain#.reshape(-1,1)
    Y_train = np.array(Ytrain[isotope])
    Ytr_m = np.mean(Y_train)
    Ytr_s = np.std(Y_train)
    Y_train = (Y_train - Ytr_m)/Ytr_s
    X_test = X#.reshape(-1,1)
    Y_test = np.array(Y[isotope])
    Yts_m = np.mean(Y_test)
    Yts_s = np.std(Y_test)
    Y_test = (Y_test - Yts_m)/Yts_s
    
    N_params = param_generator(X_train,Type)
    
    for i in tqdm(range(3)):
        res = fmin_l_bfgs_b(nll_fn(X_train, Y_train,Type,False),20*np.random.random(N_params)-5 ,
                       bounds=[(-5,20) for i in range(N_params)],approx_grad=True,m=20,pgtol=1e-7,factr=100)
        
        Output.append([res[0],res[1]])
    Output = np.array(Output)
    idx = np.argmin(Output[:,1])
    # Store the optimization resuparams[0]ts in gparams[0]obaparams[0] variabparams[0]es so that we can
    # compare it params[0]ater with the resuparams[0]ts from other impparams[0]ementations.
    #l_opt, sigma_f_opt, noise = Output[idx,1]
    #l_opt, sigma_f_opt, noise
    print(np.exp(Output[idx,1]),np.exp(Output[idx,0]))
    Params = np.exp(Output[idx,0])
    #Params = np.array([P[1],P[2],P[3],P[4],P[0],P[-1]])
    s_tes = np.argsort(X_test[:,-1])
    X_test = X_test[s_tes]
    Y_test = Y_test[s_tes]
    
    # Compute the prosterior predictive statistics with optimized kerneparams[0] parameters and pparams[0]ot the resuparams[0]ts
    O = [posterior_predictive(np.expand_dims(xs,axis=0), X_train, Y_train, Type,Params) for xs in X_test]
    
    O2 = [posterior_predictive(np.expand_dims(xs,axis=0), X_train, Y_train, Type,P) for xs in X_test]
    #std = cov_s.diagonal().ravel()
    O = np.array(O)
    mu_s = O[:,0].ravel()
    mu_s2 = np.array(O2)[:,0].ravel()
    std = O[:,1].ravel()
    Y_train = (Y_train*Ytr_s) + Ytr_m
    mu_s = (mu_s*Ytr_s) + Ytr_m
    mu_s2 = (mu_s2*Ytr_s) + Ytr_m
    #std = (std*Ytr_s) + Ytr_m
    Y_test = (Y_test*Yts_s) + Yts_m
    
    Error = Y_test-mu_s
    Error2 = Y_test-mu_s2
    
    R1 = Rsquare(mu_s,Y_test)
    R2 = Rsquare(mu_s2,Y_test)
    if R1 > 0.9:
        Kernel['Params'] = Params
        np.save('/Users/AFigueroa/Desktop/GPR-Paper/Kernels-Mass/Sobol/{}.npy'.format(isotope),Kernel)
        correct.append(isotope)
    else:
        bad.append(isotope)
        

#plt.figure()
#plt.suptitle('{}'.format(Type),fontweight='bold')
#plt.plot(X_test[:,-1],mu_s,color='r')
##plt.fill_between(X_test[:,-1][s_tes],mu_s[s_tes]-std[s_tes],mu_s[s_tes]+std[s_tes])
#plt.scatter(X_train[:,-1],Y_train)
#plt.scatter(X_test[:,-1],Y_test)



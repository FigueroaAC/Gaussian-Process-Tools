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

import warnings
warnings.filterwarnings('ignore')
# In[]
def kernel(X1, X2, Type,*params):
    
    def dif(array1,array2):
        Output = [array1[i]-array2[i] for i in range(len(array1))]
        Output = np.array(Output).ravel()
        return sum(Output)
    
    if Type == 'SQE':
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return params[0][1] * np.exp(-0.5  * sqdist / params[0][0]**2 )
    if Type == 'ASQE':
        LAMBDA = np.eye(X1.shape[1])
        length_scales = 1/params[0][:X1.shape[1]+1]
        np.fill_diagonal(LAMBDA,length_scales)
        X1 = np.dot(X1,LAMBDA)
        X2 = np.dot(X2,LAMBDA)
        sqdist = cdist(X1 , X2, metric='sqeuclidean').T
        return  np.exp(-0.5  * sqdist)

    if Type == 'LAP':
        dist = np.sqrt(np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T))
        return params[0][0]* np.exp(-0.5  * dist / params[0][1]**2 )
    if Type == 'ALAP':
        dist = np.sqrt(cdist(X1/params[0][:-1],X2/params[0][:-1], metric='euclidean').T)
        return params[0][1] * np.exp(-0.5  * dist)
    if Type == 'Poly':
        LAMBDA = np.eye(len(X1[0]))
        length_scales = 1/params[0][1:X1.shape[1]+1]
        np.fill_diagonal(LAMBDA,length_scales)
        X1 = np.dot(X1,LAMBDA) + params[0][X1.shape[1]:X1.shape[1]+1].T
        X2 = np.dot(X2,LAMBDA) + params[0][X1.shape[1]:X1.shape[1]+1].T
        return (params[0][0]*np.dot(X1,X2.T))**params[0][-1]
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
        return (1/(1 + (sqdist/params[0][-1])) )
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
    if Type == 'Tstudent':
        LAMBDA = np.eye(len(X1[0]))
        length_scales = 1/params[0][:-1]
        np.fill_diagonal(LAMBDA,length_scales)
        X1 = np.dot(X1,LAMBDA)
        X2 = np.dot(X2,LAMBDA)
        sqdist = cdist(X1 , X2, metric='euclidean').T
        return 1/(1+(sqdist**params[0][-1]))
        
       
def posterior_predictive(X_s, X_train, Y_train, Type, params,alpha_,ndims,K_inv):

    K_s = kernel(X_train, X_s, Type, params[:-int(ndims**2)-ndims])
    Kf = np.reshape(params[-int(ndims**2):],(ndims,ndims))
    Kf = np.tril(Kf) + np.tril(Kf,-1).T
    arr = np.eye(ndims)
    np.fill_diagonal(arr,params[-int(ndims**2)-ndims:-int(ndims**2)])
    K_ss = [np.kron(Kf[:,i],K_s) + arr[i][i]*np.ones(Y_train.shape[0]) for i in range(ndims)]
    
    KF = [np.kron(Kf[:,i],K_s) for i in range(ndims)]
    mu = [np.dot(KF[i],alpha_) for i in range(ndims)]
 
    cov_s = [np.ravel(K_ss[i].dot(K_ss[i].T) - np.dot(np.dot(KF[i], K_inv),KF[i].T)) \
             for i in range(ndims)]
    return mu, cov_s
  


def ll(X_train,Y_train,Type,ndims,theta):
    theta = np.exp(theta)
    # =========================================================================
    # Check we are not sampling the same values for the noises
    # =========================================================================
    for i in range(-int(ndims**2)-ndims,-int(ndims**2)):
        for j in range(i+1,-int(ndims**2)):
            if theta[i] == theta[j]:
                return np.inf
    Kx = kernel(X_train, X_train, Type,theta[:-int(ndims**2)-ndims]) 
    Kf = np.reshape(theta[-int(ndims**2):],(ndims,ndims))
    Kf = np.tril(Kf) + np.tril(Kf,-1).T
    arr = np.eye(ndims)
    np.fill_diagonal(arr,theta[-int(ndims**2)-ndims:-int(ndims**2)])
    Kself = np.kron(Kf,Kx) + np.kron(arr,np.eye(len(X_train))) 
    
    try:
        Lself = cholesky(Kself)

        alpha_self = linalg.cho_solve((Lself, True), Y_train)

        return 0.5 * len(X_train)* 2 * np.log(2*np.pi) +\
                0.5* np.sum(np.log(np.diagonal(Lself))) +\
                0.5 *1* np.dot(Y_train,alpha_self) #+\

    except (np.linalg.LinAlgError, ValueError):
        return np.inf
    

def nll_fn(X_train, Y_train, Type,ndims,naive =True):

    def nll_naive(theta):
        K = kernel(X_train, X_train, Type, theta[0], sigma_f=theta[1]) + \
            theta[2]**2 * np.eye(len(X_train))
        return 0.5 * np.log(det(K)) + \
               0.5 * Y_train.T.dot(inv(K).dot(Y_train)) + \
               0.5 * len(X_train) * np.log(2*np.pi)
    
    def nll_stable(theta):
        theta = np.exp(theta)
        
        for i in range(-int(ndims**2)-ndims,-int(ndims**2)):
            for j in range(i+1,-int(ndims**2)):
                if theta[i] == theta[j]:
                    return np.inf
                
        Kx = kernel(X_train, X_train, Type,theta[:-int(ndims**2)-ndims]) 
        Kf = np.reshape(theta[-int(ndims**2):],(ndims,ndims))
        Kf = np.tril(Kf) + np.tril(Kf,-1).T
        arr = np.eye(ndims)
        np.fill_diagonal(arr,theta[-int(ndims**2)-ndims:-int(ndims**2)])

        Kself = np.kron(Kf,Kx) + np.kron(arr,np.eye(len(X_train))) 
        
        try:

            Lself = cholesky(Kself)

            alpha_self = linalg.cho_solve((Lself, True), Y_train)

            return 0.5 * len(X_train)* 2 * np.log(2*np.pi) +\
                    0.5* np.sum(np.log(np.diagonal(Lself))) +\
                    0.5 *1* np.dot(Y_train,alpha_self) 
        except (np.linalg.LinAlgError, ValueError):
            return np.inf
        
    
    if naive:
        return nll_naive
    else:
        return nll_stable


def param_generator(Xtrain,Type,ndims):
    if Type == 'SQE':
        return Xtrain.shape[-1] + ndims + int(ndims**2)
    if Type == 'ASQE':
        return Xtrain.shape[-1] + ndims + int(ndims**2)
    if Type == 'LAP':
        return Xtrain.shape[-1] + ndims + int(ndims**2)
    if Type == 'ALAP':
        return Xtrain.shape[-1] + ndims + int(ndims**2)
    if Type == 'Poly':
        return Xtrain.shape[-1] + ndims + int(ndims**2)
    if Type == 'RQ':
        return Xtrain.shape[-1]+2
    if Type == 'MultiQuad':
        return Xtrain.shape[-1] + ndims + int(ndims**2)
    if Type == 'InvMultiQuad':
        return Xtrain.shape[-1] + ndims + int(ndims**2)
    if Type == 'Tstudent':
        return Xtrain.shape[-1]+2
    if Type == 'SRQ':
        return Xtrain.shape[-1] + ndims + int(ndims**2)
    
        
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
    

def Kernel_optimizer(Xtrain,Y,Xtest,Isotopes,size,Type,N_opt,N_repeat,approach,plot=False):
    
    ndims = len(Isotopes)
    # Prepare training data:
    Y_train = [Y[isotope][:size] for isotope in Isotopes]
    Y_train = np.array(Y_train).T
    X_train = Xtrain[:size]
    Y_m = np.mean(Y_train,axis=0)
    Y_sd = np.std(Y_train,axis=0)
    # Normalize Y's:
    Y_train = (Y_train - Y_m)/Y_sd
    # Flatten array:
    Ytrain = Y_train.T.ravel()
    # Number of parameters to fit:
    N_params = param_generator(X_train,Type,ndims)
    # =============================================================================
    # Space pre-exploration, here we select points close to a minimum as starting points
    # =============================================================================
    if approach == 'Traditional':
    
        loglike = []
        points = 10000#0
        Kernel_modifier = ['RQ','SRQ','MultiQuad','InvMultiQuad']
        if Type in Kernel_modifier:
            Type_mod = 1
            N_params+=1
        elif Type == 'Poly':
            Type_mod = 3
            N_params+=3
        else:
            Type_mod = 0
            
        # ======================================================
        # Initialize different bounds for the sampling of theta
        #           (This is of upmost importance)
        # ======================================================
        amp_param = [15 for i in range(X_train.shape[1]+Type_mod)]
        amp_sigma = [3 for i in range(ndims)] # we are looking for small noise
        amp_dims = [8 for i in range(int(ndims**2))] # we are looking for not too large elements of the matrix
        amp_theta = np.array(amp_param + amp_sigma + amp_dims)
        
        intersect_param = [5 for i in range(X_train.shape[1]+Type_mod)]
        intersect_sigma = [3 for i in range(ndims)] # we are looking for small noise
        intersect_dims = [1 for i in range(int(ndims**2))] # we are looking for not too large elements of the matrix 
        intersect_theta = np.array(intersect_param+intersect_sigma+intersect_dims)
    
        thetaset = [amp_theta*np.random.random(N_params)-intersect_theta for i in range(points)]
        for i in tqdm(range(points)):
            loglike.append(ll(X_train, Ytrain,Type,ndims,thetaset[i]))
        thetaset = np.array(thetaset)
        # =========================================================================
        # Sort the log likelihoods found:    
        # =========================================================================
        loglike = np.array(loglike)
        idx = np.argsort(loglike)
        thetaset = thetaset[idx]
        loglike = loglike[idx]
    # =========================================================================
    # Sometimes it works better to start from the valleys [i], sometimes from  
    # the mountains [-i]. here we try both approaches:
    # =========================================================================
    
        Output = []
        for j in tqdm(range(N_repeat)):
            for i in tqdm(range(N_opt)):
                res = fmin_l_bfgs_b(nll_fn(X_train, Ytrain,Type,ndims,False),thetaset[i] ,
                            bounds=[(-np.inf,np.inf) for i in range(N_params)],approx_grad=True,m=40,pgtol=1e-8,factr=100)
                Output.append([res[0],res[1]])
        for j in tqdm(range(N_repeat)):
            for i in tqdm(range(N_opt)):
                res = fmin_l_bfgs_b(nll_fn(X_train, Ytrain,Type,ndims,False),thetaset[-i] ,
                            bounds=[(-np.inf,np.inf) for i in range(N_params)],approx_grad=True,m=40,pgtol=1e-8,factr=100)
                Output.append([res[0],res[1]])
        Output = np.array(Output)
    elif approach == 'Experimental':
        loglike = []
        points = 100000
        Kernel_modifier = ['RQ','SRQ','MultiQuad','InvMultiQuad']
        if Type in Kernel_modifier:
            Type_mod = 1
            N_params+=1
        elif Type == 'Poly':
            Type_mod = 3
            N_params+=3
        else:
            Type_mod = 0
            
        # ======================================================
        # Initialize different bounds for the sampling of theta
        #           (This is of upmost importance)
        # ======================================================
        amp_param = [15 for i in range(X_train.shape[1]+Type_mod)]
        amp_sigma = [3 for i in range(ndims)] # we are looking for small noise
        amp_dims = [8 for i in range(int(ndims**2))] # we are looking for not too large elements of the matrix
        amp_theta = np.array(amp_param + amp_sigma + amp_dims)
        
        intersect_param = [5 for i in range(X_train.shape[1]+Type_mod)]
        intersect_sigma = [3 for i in range(ndims)] # we are looking for small noise
        intersect_dims = [1 for i in range(int(ndims**2))] # we are looking for not too large elements of the matrix 
        intersect_theta = np.array(intersect_param+intersect_sigma+intersect_dims)
    
        thetaset = [amp_theta*np.random.random(N_params)-intersect_theta for i in range(points)]
        for i in tqdm(range(points)):
            loglike.append(ll(X_train, Ytrain,Type,ndims,thetaset[i]))
        thetaset = np.array(thetaset)
        # =========================================================================
        # Sort the log likelihoods found:    
        # =========================================================================
        loglike = np.array(loglike)
        idx = np.argsort(loglike)
        thetaset = thetaset[idx]
        loglike = loglike[idx]
        Output = []
        for j in tqdm(range(N_repeat)):
            res = fmin_l_bfgs_b(nll_fn(X_train, Ytrain,Type,ndims,False),thetaset[0] ,
                            bounds=[(-np.inf,np.inf) for i in range(N_params)],approx_grad=True,m=40,pgtol=1e-8,factr=100)
            Output.append([res[0],res[1]])
        Output = np.array(Output)
    # =========================================================================
    # Optimal Parameters:
    # =========================================================================
    idx = np.argmin(Output[:,1])
    print(np.exp(Output[idx,1]),np.exp(Output[idx,0]))
    Params = np.exp(Output[idx,0])
    # =========================================================================
    # Precomute alpha:
    # =========================================================Par=============
    K = kernel(X_train, X_train, Type, Params[:-int(ndims**2)-ndims]) 
    Kf = np.reshape(Params[-int(ndims**2):],(ndims,ndims))
    Kf = np.tril(Kf) + np.tril(Kf,-1).T
    arr = np.eye(ndims)
    np.fill_diagonal(arr,Params[-int(ndims**2)-ndims:-int(ndims**2)])
    Kself = np.kron(Kf,K) + np.kron(arr,np.eye(len(X_train)))
    L_ = linalg.cholesky(Kself,lower=True)
    alpha_ = linalg.cho_solve((L_,True),Ytrain)
    L_inv = linalg.solve_triangular(L_.T,np.eye(L_.shape[0]))
    K_inv = L_inv.dot(L_inv.T)

    # =========================================================================
    # Compute predictions: (Variance computation is not yet implemented)
    # =========================================================================
    Predictions = [posterior_predictive(xs[:,np.newaxis].T, X_train, Ytrain, Type,Params,alpha_,ndims,K_inv) for xs in Xtest]
    Predictions = np.array(Predictions)

    Ypred = np.array(Predictions)[:,0,:,0]
    Ystd = np.array(Predictions)[:,1,:,0]
    # =============================================================================
    # Reverse Normalization: 
    # =============================================================================
    
    Ypred = (Ypred*Y_sd) + Y_m
    Y_train = (Y_train*Y_sd) + Y_m 
    Ystd = (Ystd*Y_sd) + Y_m
    
#    if plot:
#        for i,isotope in enumerate(Isotopes):
#            plt.figure()
#            plt.suptitle('Multitask GP Model {} - {} Kernel'.format(isotope,Type),fontweight='bold')
#            plt.scatter(X_train[:,1],Y_train[:,i],label='Train points')
#            plt.scatter(Xtest[:,1],Ypred[:,i],label='Predictions')
#            plt.xlabel('Burnup (MWd/KgHM)',fontweight='bold')
#            plt.ylabel('Mass (Kg)',fontweight='bold')
#            plt.legend()
#            plt.grid(True)
    
    if plot:
        # =====================================================================
        # Calculate number of rows and columns
        # =====================================================================
        c = 1
        r = 1
        run = True
        flag = 'c'
        while run:
            if ndims > r*c and flag == 'c':
                c+=1
                flag = 'r'
            elif ndims > r*c and flag == 'r':
                r+=1
                flag = 'c'
            else:
                run = False
        print(r,c)
        fig, axs = plt.subplots(r, c)
        counter = 0
        for i in range(r):
            for j in range(c):
                if counter < ndims:
                    axs[i,j].scatter(X_train[:,-1],Y_train[:,counter],label='Train points')
                    axs[i,j].scatter(Xtest[:,-1],Ypred[:,counter],label='Predictions')
                    axs[i,j].legend()
                    axs[i,j].set_title('Multitask GP Model {} - {} Kernel'.format(Isotopes[counter],Type),fontweight='bold')
                    axs[i,j].grid(True)
                    axs[i,j].set_xlabel('Burnup (MWd/KgHM)',fontweight='bold')
                    axs[i,j].set_ylabel('Mass (Kg)',fontweight='bold')
                    counter+=1
        fig.tight_layout()

    K_params = np.concatenate((Params[:-int(ndims**2)],Kf.ravel()))
    return K_params


# In[]
# =============================================================================
# Load Datasets
# =============================================================================
Xtrain = np.load('',allow_pickle=True)

Ytrain = np.load('',allow_pickle=True).item()

Xtest = np.load('',allow_pickle=True)

Ytest = np.load('',allow_pickle=True).item()

# In[]
# =============================================================================
# Compute Kernel Parameters
# =============================================================================
Isotopes = ['Pu239','Pu240','Cs137','Ba137']
size = 50 # Size of training sets. Multitasking training scales as size^2 * #_isotopes^2
Type = 'ASQE' #'SQE'|'ASQE'|'RQ'|'SRQ'|'InvMultiQuad'} Should work fine
N_opt = 1
N_repeat = 1
approach = 'Traditional'#'Traditional'|'Experimental'
plot = True
Parms = Kernel_optimizer(Xtrain,Ytrain,Xtest,Isotopes,size,Type,N_opt,N_repeat,approach,plot)

# In[]
## ============================================================================
## Test functions. with 2 functions it works well, for more it is not so reliable
## ============================================================================
#if __name__ == '__main__':
#    def f1(x):
#        return np.sqrt(x)
#    def f2(x):
#        return x**2
#    def f3(x):
#        return x
#    
#    x = np.linspace(0,1,20)
#    Y1 = f1(x)
#    Y2 = f2(x)
#    Y3 = f3(x)
#    
#    Y1m = np.mean(Y1)
#    Y1s = np.std(Y1)
#    Y2m = np.mean(Y2)
#    Y2s = np.std(Y2)
#    Y3m = np.mean(Y3)
#    Y3s = np.std(Y3)
#    
#    Y1 = (Y1-Y1m)/Y1s
#    Y2 = (Y2-Y2m)/Y2s
#    Y3 = (Y3-Y3m)/Y3s
#    
#    
#    Y = np.hstack([Y1,Y2])
#    Y = np.hstack([Y,Y3])
#    ndims = 3
#    N_params = param_generator(x[:,np.newaxis],Type,ndims)
#    
#    # =============================================================================
#    # Space pre-exploration, here we select points close to a minimum as starting points
#    # =============================================================================
#    amp_param = [15 for i in range(x[:,np.newaxis].shape[1])]
#    amp_sigma = [3 for i in range(ndims)] # we are looking for small noise
#    amp_dims = [8 for i in range(int(ndims**2))] # we are looking for not too large elements of the matrix
#    amp_theta = np.array(amp_param + amp_sigma + amp_dims)
#    
#    intersect_param = [5 for i in range(x[:,np.newaxis].shape[1])]
#    intersect_sigma = [3 for i in range(ndims)] # we are looking for small noise
#    intersect_dims = [1 for i in range(int(ndims**2))] # we are looking for not too large elements of the matrix 
#    intersect_theta = np.array(intersect_param+intersect_sigma+intersect_dims)
#    
#    loglike = []
#    points = 50000
#    
#    thetaset = [amp_theta*np.random.random(N_params)-intersect_theta for i in range(points)]
#    for i in tqdm(range(points)):
#        loglike.append(ll(x[:,np.newaxis], Y,Type,ndims,thetaset[i]))
#    thetaset = np.array(thetaset)
#    
#    # =============================================================================
#    # Optimize hyperparamters:
#    # =============================================================================
#    Output = []
#    N_repeats = 5
#    N_opts = 300
#    for j in tqdm(range(N_repeats)):
#        for i in range(N_opts):
#            res = fmin_l_bfgs_b(nll_fn(x[:,np.newaxis], Y,Type,ndims,False),thetaset[-i] ,
#                           bounds=[(-np.inf,np.inf) for i in range(N_params)],approx_grad=True,m=40,pgtol=1e-14,factr=10.0,epsilon=1e-10)
#            Output.append([res[0],res[1]])
#    Output = np.array(Output)
#    idx = np.argmin(Output[:,1])
#    print(np.exp(Output[idx,1]),np.exp(Output[idx,0]))
#    Params = np.exp(Output[idx,0])
#    
#    #Params = Paramsgood
#    K = kernel(x[:,np.newaxis], x[:,np.newaxis], Type, Params[:-int(ndims**2)-ndims]) 
#    Kf = np.reshape(Params[-int(ndims**2):],(ndims,ndims))
#    arr = np.eye(ndims)
#    np.fill_diagonal(arr,Params[-int(ndims**2)-ndims:-int(ndims**2)])
#    
#    Kself = np.kron(Kf,K) + np.kron(arr,np.eye(len(x)))
#    L_ = linalg.cholesky(Kself,lower=True)
#    alpha_ = linalg.cho_solve((L_,True),Y)
#    
#    xtest = np.linspace(0,1,100)[:,np.newaxis]
#    
#    # Compute the prosterior predictive statistics with optimized kerneparams[0] parameters and pparams[0]ot the resuparams[0]ts
#    O = [posterior_predictive(xs[:,np.newaxis], x[:,np.newaxis], Y, Type,Params,alpha_,ndims) for xs in xtest]
#    O = np.array(O)
#    Ypred1 = O[:,0,0]
#    Ypred2 = O[:,1,0]
#    Ypred3 = O[:,2,0]
#    
#    Ypred1 = (Ypred1*Y1s) + Y1m
#    Ypred2 = (Ypred2*Y2s) + Y2m
#    Ypred3 = (Ypred3*Y3s) + Y3m
#    
#    Y1 = (Y1*Y1s) + Y1m
#    Y2 = (Y2*Y2s) + Y2m
#    Y3 = (Y3*Y3s) + Y3m
#    
#    plt.figure()
#    plt.scatter(x,Y1)
#    plt.scatter(xtest,Ypred1)
#    
#    plt.figure()
#    plt.scatter(x,Y2)
#    plt.scatter(xtest,Ypred2)
#    
#    plt.figure()
#    plt.scatter(x,Y3)
#    plt.scatter(xtest,Ypred3)

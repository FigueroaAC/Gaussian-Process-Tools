#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:48:21 2020

@author: AFigueroa
"""

import numpy as np
from scipy import linalg
from scipy.spatial import distance
from tqdm import tqdm
# In[]:
def alpha_calculator(Params,X,Y):
    Cs = Params[0]
    P =  Params[1:-1]
    alpha = Params[-1]
    LAMBDA = np.eye(len(X[0]))
    length_scales = (1/P)
    np.fill_diagonal(LAMBDA,length_scales)
    Xtrain = np.dot(X,LAMBDA)
    distSself = distance.cdist(Xtrain , Xtrain, metric='sqeuclidean').T
    # Calculate Self Correlation:
    KSelf = Cs * np.exp(-.5 * distSself)
    Y_norm = (Y - np.mean(Y))/np.std(Y) 
    KSelf = KSelf + alpha*np.eye(len(KSelf))
    L_ = linalg.cholesky(KSelf,lower=True)
    L_inv = linalg.solve_triangular(L_.T,np.eye(L_.shape[0]))
    K_inv = L_inv.dot(L_inv.T)
    alpha_ = linalg.cho_solve((L_,True),Y_norm)
    return alpha_, K_inv

def GPR_MODEL_w_Std(Params,Lambda_Inv,Xtrain,Ytrain,alpha_,K_inv,Xtest):
    Constant = Params[0]
    sigma = Params[-1]
    Distance = Xtest - Xtrain
    Distance_sq = np.diag(np.dot(np.dot(Distance,Lambda_Inv),Distance.T))
    Ktrans = (Constant * np.exp(-0.5*Distance_sq)) + (sigma*np.ones(len(Xtrain)))
    Ktranstrans = Constant + sigma
    
    Y_pred = np.dot(Ktrans,alpha_)
    Y_pred = (Y_pred*np.std(Ytrain))+np.mean(Ytrain)

    Var = np.expand_dims(Ktranstrans,-1) -np.einsum(
            "ij,ij->i", np.dot(np.expand_dims(Ktrans,0), K_inv), np.expand_dims(Ktrans,0))
    Var = np.sqrt(np.abs(Var[0]))
    return Y_pred, Var

def Sensitivity_Analysis(Xtrain,Ytrain,kernel,XtestGrid,size):
    # =========================================================================
    # Generate Test points    
    # =========================================================================
    Parambounds = [[np.min(XtestGrid[:,0]),np.max(XtestGrid[:,0])], 
                    [np.min(XtestGrid[:,1]),np.max(XtestGrid[:,1])] ]
    
    # =============================================================================
    # Total Sensitivity Indexes:
    # =============================================================================

    Xtest = [ [(np.random.random()*Parambounds[0][1])+Parambounds[0][0],
               (np.random.random()*Parambounds[1][1])+Parambounds[1][0]] \
                for i in range(size)]
    Xtest = np.array(Xtest)
    
    Params = kernel['Params']
    Lambda = kernel['LAMBDA']
    
    LambdaT = Lambda[0][0]
    LambdaB = Lambda[1][1]
    
    XtrainT = Xtrain[:,0]
    XtrainB = Xtrain[:,1]
    
    XtestT = Xtest[:,0]
    XtestB = Xtest[:,1]
    
    alpha,kinv = alpha_calculator(Params,Xtrain,Ytrain)
    
    YT = [GPR_MODEL_w_Std(Params,LambdaT,XtrainT[:,np.newaxis],Ytrain,alpha,kinv,xt) \
          for xt in XtestT]
    YB = [GPR_MODEL_w_Std(Params,LambdaB,XtrainB[:,np.newaxis],Ytrain,alpha,kinv,xb) \
          for xb in XtestB]
    Yall = [GPR_MODEL_w_Std(Params,Lambda,Xtrain,Ytrain,alpha,kinv,x) \
          for x in Xtest]
    
    Yall = np.array(Yall)
    YT = np.array(YT)
    YB = np.array(YB)
    
    VarT = np.var(YT,axis=0)
    VarB = np.var(YB,axis=0)
    Varall = np.var(Yall,axis=0)
    
    StotTmu = VarT[0]/Varall[0]
    StotTsd = VarT[1]/Varall[1]
    
    StotBmu = VarB[0]/Varall[0]
    StotBsd = VarB[1]/Varall[1]
    
    # =========================================================================
    # 1st order sensitivity indexes:    
    # =========================================================================
    T = np.linspace(Parambounds[0][0],Parambounds[0][1],int(np.sqrt(size)))
    B = np.linspace(Parambounds[1][0],Parambounds[1][1],int(np.sqrt(size)))
    
    Tts,Bts = np.meshgrid(T,B)
    
    XtestGrid = np.array([Tts.ravel(),Bts.ravel()]).T
    
    Yall_1st = [GPR_MODEL_w_Std(Params,Lambda,Xtrain,Ytrain,alpha,kinv,x) \
          for x in XtestGrid]
    
    Yall_1st = np.array(Yall_1st)
    Yall_mu = Yall_1st[:,0].reshape(( int(np.sqrt(XtestGrid.shape[0])),
                      int(np.sqrt(XtestGrid.shape[0]))))
    Yall_sd = Yall_1st[:,1].reshape(( int(np.sqrt(XtestGrid.shape[0])),
                      int(np.sqrt(XtestGrid.shape[0]))))
    
    VarT1 = np.mean(np.var(Yall_mu,axis=1))
    VarB1 = np.mean(np.var(Yall_mu,axis=0))
    
    VarT1sd = np.var(np.var(Yall_sd,axis=1))
    VarB1sd = np.var(np.var(Yall_sd,axis=0))
    
    Varall = np.var(Yall_1st,axis=0)
    
 
    S1Tmu = VarT1/Varall[0]
    S1Tsd = VarT1sd/Varall[1]
    
    S1Bmu = VarB1/Varall[0]
    S1Bsd = VarB1sd/Varall[1]
                     
    print('\n')
    print('S1-Temp = ',S1Tmu,'+/-',np.sqrt(S1Tsd))
    print('S1-Burn = ',S1Bmu,'+/-',np.sqrt(S1Bsd))
    
    print('ST-Temp = ',StotTmu,'+/-',np.sqrt(StotTsd))
    print('ST-Burn = ',StotBmu,'+/-',np.sqrt(StotBsd))
    
    STemp = [S1Tmu, S1Tsd, StotTmu, StotTsd]
    SBurn = [S1Bmu, S1Bsd, StotBmu, StotBsd]
    return STemp[:2],SBurn[:2]
    
# In[]
    
X = np.load('',allow_pickle=True)
Y = np.load('',allow_pickle=True).item()

XtestG = np.load('',allow_pickle=True)
# =============================================================================
# Order inputs, customize depending on the case:
# =============================================================================
T = np.array(list(set(XtestG[:,0])))
B = np.array(list(set(XtestG[:,1])))
T,B = np.meshgrid(T,B)
XtestG = np.array([T.ravel(),B.ravel()]).T

# In[]
isotopes = [iso for iso in Y]
Out = []
nuclides = []
for isotope in tqdm(isotopes):
    try:
        Kernel = np.load('/kernel.npy'.format(isotope),
                         allow_pickle=True).item()
        size=100
        
        Out.append(Sensitivity_Analysis(X,Y[isotope],Kernel,XtestG,int(size)))
        nuclides.append(isotope)
    except FileNotFoundError:
        continue
Out = np.array(Out)

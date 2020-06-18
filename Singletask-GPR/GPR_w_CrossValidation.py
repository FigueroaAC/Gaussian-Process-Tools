#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:42:02 2020

@author: figueroa
"""
import sys
import numpy as np
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from scipy import linalg
from scipy.spatial import distance
from tqdm import tqdm
from multiprocessing import Pool
from scipy.optimize import curve_fit
np.random.seed()
warnings.filterwarnings('ignore')

# In[]
"""
GPR Tools
"""
# =============================================================================
# Ordinary Kriging Optimizer
# =============================================================================
def kriging(X, Y):
    """
    Scikit-learn Implementation for training the GP parameters. In here a 
    Squared Exponential kernel is used, plus a small term of white noise
    """
    # =============================================================================
    #     Instantiate Gaussian Process
    # =============================================================================
    kernel = C(np.var(Y), (1e-6, 1e6)) *\
        RBF([10 for i in range(len(X[0]))] , (1e-5, 1e10)) +\
        WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+4))
    gp = GaussianProcessRegressor(kernel=kernel , n_restarts_optimizer=50, 
                                  normalize_y=True)
    gp.fit(X, Y)
    Params = np.exp(gp.kernel_.theta)
    # =============================================================================
    # Precompute Kernel Parameters
    # =============================================================================
    Params = np.exp(gp.kernel_.theta)
    alpha_ = gp.alpha_
    L_inv = linalg.solve_triangular(gp.L_.T, np.eye(gp.L_.shape[0]))
    #K_inv = L_inv.dot(L_inv.T)
    LAMBDA = np.eye(len(X[0]))
    length_scales = (1/Params[1:-1])**2
    np.fill_diagonal(LAMBDA,length_scales)
    print(Params)
    return #[Params, LAMBDA, alpha_, L_inv]
# =============================================================================
# GPR Model:
# =============================================================================
def GPR_MODEL(Constant,Lambda_Inv,Xtrain,Ytrain,alpha_,Xtest):#,tconstant):
    """
    Implementation of the calculation of the mean of a GP
    """
    #Distance = Xtest[:-1] - Xtrain
    Distance = Xtest - Xtrain
    Distance_sq = np.diag(np.dot(np.dot(Distance,Lambda_Inv),Distance.T))
    Ktrans = Constant * np.exp(-0.5*Distance_sq)
    Y_pred = np.dot(Ktrans,alpha_)
    Y_pred = (Y_pred*np.std(Ytrain))+np.mean(Ytrain)
    return Y_pred #* np.exp(-tconstant * Xtest[-1]) 

def alpha_calculator(Params,X,Y):
    """
    Calculate the array alpha_ used to predict the posterior GP mean given 
    the parameters of the GP and a given training set (X,Y)
    
    inputs:
        - Params =  GP Parameters [Cs,P.....,alpha]
        - X = GP Xtrain dataset
        - Y = GP Ytrain dataset
    returns:
        - alpha_ = alpha_ array for further use in prediction
    """
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
    Y_norm = (Y - np.mean(Y))/np.std(Y) # Normalize Outputs
    KSelf = KSelf + alpha*np.eye(len(KSelf))
    L_ = linalg.cholesky(KSelf,lower=True)
    alpha_ = linalg.cho_solve((L_,True),Y_norm)
    return alpha_

# In[]
"""
CV Tools
"""

def CV_Analysis(X,Y,Nfold,N_repeats,multithreading=True,option='sequential'):
    """
    This code performs K-fold Cross-Validation analysis by splitting the entire 
    dataset into several 'folds'. GPR is performed on each fold and the 
    performance of the hyperparameters is tracked across each fold. In the end,
    point estimations of the CV function are produced from which the kernel 
    parameters which result on its minimum should be taken. In principle, these
    kernel parameters are those that perform the best independently of the
    training set taken, which of course is of a given size.
    We have added an N_repeats option, so the process of random selection of the
    data set can be repeated, in order to obtain better estimates of the cross
    validation error
        * The 'option' parameter allows the selection between 
        Multi-threading (Recommended) / Single-threading
    """
    Indexes = np.array(range(len(Y)))
    N = len(Y)//Nfold
    CV_Error = []
    Models = []
    
    if multithreading == True:
        print('GPR Training Mode = Multi-thread')
        if option=='sequential':
            print('GPR Training Type = {}'.format(option))
            print('\n Training GPR Models... \n')
            print('\n The counter might increase and decrease during calculation. \n Multiple processes are running on parallel and share the same progress bar \n')
            # =============================================================================
            #   Generate the different folds to perform the cross validation in:
            # =============================================================================
            IdxSets = [np.arange(N*i,N*(i+1)) for i in range(Nfold-1)]
            IdxSets.append(np.arange(IdxSets[-1][-1]+1,len(Y)))
            Models, CV_Error = get_CV_Multithread(X,Y,Nfold,IdxSets)
            """Override the user input, as more repeats is not efficient,
            due to Sobol Sampling"""
            return CV_Error, Models   
        else:
            print('GPR Training Type = Random')
            print('\n Training GPR Models... \n')
            print('\n The counter might increase and decrease during calculation. \n Multiple processes are running on parallel and share the same progress bar \n')
            for a in range(N_repeats):
                # =============================================================================
                #   Generate the different folds to perform the cross validation in:
                # =============================================================================
                IdxSets = [np.random.choice(Indexes,size = N,replace=False) for i in range(Nfold)]
                Models_Data, CVE = get_CV_Multithread(X,Y,Nfold,IdxSets)
                Models.extend(Models_Data)
                CV_Error.extend(CVE)
            return CV_Error, Models   
    else:
        print('GPR Training Mode = Single-thread')
# =============================================================================
# Single Threading Option --> Much Slower!:
# =============================================================================
        for a in range(N_repeats):
            # =============================================================================
            #   Generate the different folds to perform the cross validation in:
            # =============================================================================
            print('\n GPR with {}-fold Cross Validation Computation | Repetition {} of {}\n'.format(Nfold,a+1,N_repeats))
            IdxSets = [np.random.choice(Indexes,size = N,replace=False) for i in range(Nfold)]
            Yfolds = [Y[idx] for idx in IdxSets]
            Xfolds = [X[idx] for idx in IdxSets]
            print('\n Training GPR Models... \n')
            print('\n The counter might increase and decrease during calculation. \n Multiple processes are running on parallel and share the same progress bar \n')
            Models_Data = []
            for i in tqdm(range(len(Yfolds))):
                M = kriging(Xfolds[i],Yfolds[i])
                Models_Data.append(M)
                Models.append(M)
            # =============================================================================
            #   Perform Cross-Validation:
            # =============================================================================
            for i in tqdm(range(len(Models_Data))):
                CV_Error.append(CV_Error_Calc(Xfolds,Yfolds,i,Models_Data,Nfold)) 
        return CV_Error, Models

def CV_Error_Calc(Xfolds,Yfolds,index,Models_Data,Nfold):
    """
    Calculate the total error of the k-fold iteration by training on each fold
    and using each set of parameters iterating over each fold to predict the 
    values of the rest. The total error is then a sum over all folds error, 
    representing the global predictive error associated with the parameter set
    indicated by 'index'
    """
    Cons = Models_Data[index][0][0]
    Lambda = Models_Data[index][1]
    Fold_Errors = []
    for k in range(len(Yfolds)):
        Fold_Error = Fold_Error_Calc(Cons,Lambda,Models_Data[index],Xfolds,Yfolds,k)
        Fold_Errors.append(Fold_Error)
    CV_Error = sum(Fold_Errors)/Nfold
    return CV_Error

def Fold_Error_Calc(Cons,Lambda,Model_Data,Xfolds,Yfolds,k):
    """
    Calculate the Prediction error when using the training parameters of a 
    given fold to predict the values of the training (test) set assigned to
    the rest of the folds. 
        xf = xfold
        yf = yfold
        xt = xtest
        yt = ytest
    """
    xf = Xfolds[k]
    yf = Yfolds[k]
    xt = np.vstack([Xfolds[j] for j in range(len(Xfolds)) if j!=k])
    yt = np.hstack([Yfolds[j] for j in range(len(Yfolds)) if j!=k])
    alpha_ = alpha_calculator(Model_Data[0],xf,yf)
    Ypred = np.array([GPR_MODEL(Cons,Lambda,xf,yf,alpha_,x) for x in xt])
    Ydiff = np.power(yt - Ypred,2)
    Fold_Error = np.sum(Ydiff)
    return Fold_Error
    
def multi_kriging(Xfolds,Yfolds):
    """
    Multi-threading implementation of the GP training function
    """
    Results = []
    with Pool() as pool:
        Results = pool.starmap(kriging, 
                               [(Xfolds[i],Yfolds[i]
                                 ) for i in tqdm(range(len(Yfolds)))])
    pool.close()
    pool.join()
    return Results

def CV_Errors_Multi(Xfolds,Yfolds,Models_Data,Nfold):
    """
    Multi-threading implementation of the CV Error Calculation function
    """
    with Pool() as pool:
        Results = pool.starmap(CV_Error_Calc,
                               [(Xfolds,Yfolds,i,Models_Data,Nfold
                                 ) for i in tqdm(range(len(Models_Data)))])
    pool.close()
    pool.join()
    return Results

def get_CV_Multithread(X,Y,Nfold,IdxSets):
    """
    Train GPR models on the different folds provided by IdxSets. 
    Calculate Model Performance and 
    
    inputs: 
        - X = Dataset of input parameters onto which GP models are trained
        - Y = Dataset of input parameters onto which GP models are trained
        - Nfold = Number of folds
        - IdxSets = Set of list of indexes used to generate the different 
                    k-folds
    returns:
        - Models_Data = array consisting of trained model parameters
        - ResultsCV = array consisting of the model's predictive Errors under 
                      Cross-Validation
    """
    Yfolds = [Y[idx] for idx in IdxSets]
    Xfolds = [X[idx] for idx in IdxSets]
    print('\n Training GPR Models... \n')
    Models_Data = []
    Results = multi_kriging(Xfolds,Yfolds)
    for M in Results:
        Models_Data.append(M)
    # Each entry in Models data has the following order: Params_array, Lambda_Matrix, alpha_, L_inv
    # =============================================================================
    #   Perform Cross-Validation:
    # =============================================================================
    print('\n Performing Cross Validation...\n')
    ResultsCV = CV_Errors_Multi(Xfolds,Yfolds,Models_Data,Nfold)
    return Models_Data, ResultsCV
    
# In[]
    
def Kernel_Maker(isotope,Nfolds,Nrepeats,Multithreading=True,option='random',fit=None):
    Path = ''
    Xdata = np.load(Path+'Training_Sets/X_Candu_Grid625.npy',allow_pickle=True)
    # =========================================================================
    # Use this to perform the regression on the atom density data set
    # =========================================================================
    
    #Ydata = np.load(Path+'Training_Sets/Y_Candu_Grid625.npy',allow_pickle=True).item()
    
    
    # =========================================================================
    # Use this to perform the regression on the total mass data set
    # =========================================================================
    Ydata = np.load(Path+'Training_Sets/YTotal_Output_GridMdens.npy',allow_pickle=True).item()
    
    
    Data = np.array(Ydata[isotope])
    print('GPR Started')
    
    # =============================================================================
    # Fit a linear function and fit the GPR on the remainder of this fit    
    # =============================================================================
    def linear_fit(xdata,a,b,c):
        return a*xdata[0] + b*xdata[1] + c

    def quadratic_fit(xdata,a,b,c,d,e):
        return a*(xdata[0]**2) + b*(xdata[1]**2) + c*xdata[0] + d*xdata[1] + e  
    
    def cubic_fit(xdata,a,b,c,d,e,f,g,h,i,j):
        return a*(xdata[0]**3) + b*(xdata[1]**3) + c*(xdata[0]**2)*xdata[1] + d*(
                xdata[1]**2)*xdata[0] + e*(xdata[0]**2) + f*(xdata[1]**2) + g*(
                xdata[0]*xdata[1]) + h*xdata[0] + i*xdata[1] + j
    
    #poptq,pcovq = curve_fit(quadratic_fit,Xdata.T,Data)
    #remainderq = Data-quadratic_fit(Xdata.T,poptq[0],poptq[1],poptq[2],poptq[3],poptq[4])  
    poptq,pcov = curve_fit(cubic_fit,Xdata.T,Data)
    remainder = Data-cubic_fit(Xdata.T,poptq[0],poptq[1],poptq[2],poptq[3],poptq[4],
                                poptq[5],poptq[6],poptq[7],poptq[8],poptq[9])
    #Xdata[:,0] = (Xdata[:,0] - np.mean(Xdata[:,0]))/np.std(Xdata[:,0])
    #Xdata[:,1] = (Xdata[:,1] - np.mean(Xdata[:,1]))/np.std(Xdata[:,1])
    
    if fit == 'do':
        Data = remainder
    else:
        Data = Data
    
    if Nfolds > 1:
    
        CV, Models = CV_Analysis(Xdata,Data,Nfolds,Nrepeats,Multithreading,option)
    
    # =============================================================================
    # Select the parameters that result in the smallest cross validation error
    # =============================================================================
    
        idx = np.argmin(CV)
        print(idx)
        print(len(CV),len(Models))
        Params = Models[idx][0]
        Cons = Params[0]
        Lambda = Models[idx][1]
    # =============================================================================
    # Diagnostics to evaluate the prediction error using now the first 500 
    # entries of the Dataset as Training Data    
    # =============================================================================
        stIdx = 500
        alpha_ = alpha_calculator(Params,Xdata[:stIdx],Data[:stIdx])
        Ypred = np.array([GPR_MODEL(Cons,Lambda,Xdata[:stIdx],Data[:stIdx],alpha_,x) for x in Xdata[stIdx:]])
        Error = Data[stIdx:] - np.array(Ypred)
        rel_Error = Error*100/Data[stIdx:]
        print(np.max(Error),np.max(rel_Error))
        try:
            Half_Life = np.log(2)/np.load(Path+'Training_Sets/Halflives.npy',allow_pickle=True).item()[isotope]
        except KeyError:
            Half_Life = 0
        Kernel = {}
        Kernel['Params'] = Models[idx][0]
        Kernel['LAMBDA'] = Models[idx][1]
        Kernel['alpha_'] = Models[idx][2]
        Kernel['Half-Life'] = Half_Life
        Kernel['Cubic'] = poptq
        
        Kernel_Path = 'Kernels-v2/Grid/{}.npy'.format(isotope)
        np.save(Path+Kernel_Path,Kernel)
    else:
        Models = kriging(Xdata,Data)
        try:
            Half_Life = np.log(2)/np.load(Path+'Training_Sets/Halflives.npy',allow_pickle=True).item()[isotope]
        except KeyError:
            Half_Life = 0
        Kernel = {}
        Kernel['Params'] = Models[0]
        Kernel['LAMBDA'] = Models[1]
        Kernel['alpha_'] = Models[2]
        Kernel['Half-Life'] = Half_Life
        Kernel['Cubic'] = poptq
        
        Kernel_Path = 'Kernels-All/Grid/{}.npy'.format(isotope)
        np.save(Path+Kernel_Path,Kernel)
 
if __name__ == '__main__':
    """
    Performing 'sequential' sampling to generate the folds generally results in
    good GPR models if the original simulation sampling is done approprietly, 
    e.g Sobol, Halton.
    Otherwise, a slower but often better approach -provided enough folds and 
    repetitions are made- is using 'random' sampling to generate the folds
    """
    isotope = sys.argv[1]
    # E.G:
    Nfolds = 5
    Nrepeats = 1
    Multithreading = True
    Type = 'random'#'sequential'|'random'|'sequential'
    Fit = None#'Linear'|'Quadratic'|'Cubic'|None
    Kernel_Maker('Pu239',Nfolds,Nrepeats,Multithreading,Type,Fit)
    
# In[]
#import matplotlib.pyplot as plt
#==============================================================================
# Max Error Visualization:
#==============================================================================
#idx = np.argmin(CV)
#print(idx)
#Params = Models[idx][0]
#Cons = Params[0]
#Lambda = Models[idx][1]
#
#def getError(Cons,Lambda,X,Y,index):
#    alpha_ = alpha_calculator(Params,X[:index],Y[:index])
#    Ypred = np.array([GPR_MODEL(Cons,Lambda,Xdata[:index],Data[:index],alpha_,x) for x in Xdata[index:]])
#    Diff = Data[index:] - np.array(Ypred)
#    Error = np.max(np.abs(Diff))
#    print('Good!')
#    return Error
#    
#L = list(range(100,1000,50))
#with Pool() as pool:
#    Error = pool.starmap(getError,[(Cons,Lambda,Xdata,Data,i) for i in L])
#pool.close()
#pool.join()
#
#plt.figure()
#plt.scatter(range(100,1000,50),Error)
#plt.xlabel('Training set size')
#plt.ylabel('Max. Prediction Error')
#plt.grid(True)



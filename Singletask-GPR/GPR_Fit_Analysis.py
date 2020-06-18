#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:52:13 2019

@author: figueroa
"""

import os
import numpy as np
os.chdir('/home/figueroa/Desktop/PyMC3')
#import Inputs_GPR as GPR
import scipy.stats as stats
from scipy.optimize import curve_fit
from tqdm import tqdm
import GPR_Inputs_NonTheano as GPR
#import smtplib, ssl
import matplotlib.pyplot as plt
import math
# In[0]:

def get_prediction(isotope_name,xtest):
    Example_Sol = [0.015,320,100000,1.5]#,0]
    CONSTANT, LAMBDA, Xtrain, Ytrain, Alpha, Ytarget = GPR.Inputs_Vector(isotope_name,Example_Sol)
    Ypred = np.zeros(len(xtest))
    
    for a in range(len(xtest)):
        #v1 = xtest[a][0]
        #v2 = xtest[a][1]
        #v3 = xtest[a][2]
        #v4 = xtest[a][3]
        Ypred[a] = GPR.GPR_MODEL(CONSTANT[0],LAMBDA[0],Xtrain,Ytrain[0],Alpha[0],xtest[a])#.eval()
    return Ypred

#def get_prediction(isotope_name,xtest):
#    Example_Sol = [0.015,320,100000,1.5,0]
#    CONSTANT, LAMBDA, Xtrain, Ytrain, Alpha, Params, Ytarget = GPR.Inputs_Vector(isotope_name,Example_Sol)
#    Ypred = np.zeros(len(xtest))
#    Ypred = []
#    for a in range(len(xtest)):
#        v1 = xtest[a][0]
#        v2 = xtest[a][1]
#        v3 = xtest[a][2]
#        v4 = xtest[a][3]
        #Ypred[a] = GPR.GPR_MODEL(CONSTANT[0],LAMBDA[0],Xtrain,Ytrain[0],Alpha[0],v1,v2,v3,v4)
#        Y = GPR.GPR_MODEL(CONSTANT[0],LAMBDA[0],Xtrain,Ytrain[0],Alpha[0],v1,v2,v3,v4)
#        Ypred.append(Y)
    #return Ypred.eval()
#    return [item.eval() for item in Ypred]

def get_stats(isotope_name,xtest,ytest):
    try:
        Ypred = get_prediction(isotope_name,xtest)
    except IndexError:
        return -1e50, -1e50
    #YpredHist, bin_edges = np.histogram(Ypred,bins=30)
    #YtestHist,bin_edges2 = np.histogram(ytest, bin_edges)
    #KolmD, Kolm_p = stats.ks_2samp(YpredHist,YtestHist)#(Ypred,ytest)#(YpredHist,YtestHist)
    #Chisq, Chi_p = stats.chisquare(YpredHist,YtestHist)#(Ypred,ytest)#(YpredHist,YtestHist)
    Deviation = Ypred - ytest
    Deviation_perc = np.zeros(len(Ypred))
    for a in range(len(Ypred)):
        Deviation_perc[a] = (1-(Ypred[a]/ytest[a]))*100
        
    #return KolmD, Kolm_p, Chisq, Chi_p
    return Deviation, Deviation_perc

# In[1]:

Ytest = np.load('/home/figueroa/Desktop/PyMC3/Ytrainingset_1000.npy',allow_pickle=True).item()
Xtest = np.load('/home/figueroa/Desktop/PyMC3/Xtrainingset_1000.npy',allow_pickle=True)[100:]

Isotope_list = [item for item in Ytest][:1229]

#Xtest = np.array([np.append(item,0) for item in Xtest])
# In[2]:
#Kolmogorov_Stat = []
#Kolmogorov_Pvalues = []
#Chisq_Stat = []
#Chi_Pvalues = []
Deviations = []
Deviations_perc = []
for isotope in tqdm(Isotope_list):
    #KolmD, Kolm_p, Chisq, Chi_p = get_stats([isotope],Xtest,Ytest[isotope][100:])
    #Kolmogorov_Stat.append(KolmD)
    #Kolmogorov_Pvalues.append(Kolm_p)
    #Chisq_Stat.append(Chisq)
    #Chi_Pvalues.append(Chi_p)
    Dev, Dev_perc = get_stats([isotope],Xtest,Ytest[isotope][100:])
    Deviations.append(Dev)
    Deviations_perc.append(Dev_perc)
    
Isotope_Data = {}
Faulty_Nuclides = []
for isotope in Isotope_list:
    Isotope_Data[isotope] = {}
    Isotope_Data[isotope]['Deviations'] = Deviations[Isotope_list.index(isotope)]
    Isotope_Data[isotope]['Deviations_perc'] = Deviations_perc[Isotope_list.index(isotope)]
    Isotope_Data[isotope]['Mean_Deviations_perc'] = np.mean(Isotope_Data[isotope]['Deviations_perc'])

# In[2]:
def print_histogram(isotope_Data, isotope):
    f = plt.figure()
    plt.hist(isotope_Data,bins=100)
    plt.suptitle('{} Difference Histogram'.format(isotope),fontweight='bold')
    plt.ylabel('Frequency',fontweight='bold')
    plt.xlabel(r'$Y_{\mathrm{pred}}-Y_{\mathrm{Sim}}$')
    plt.grid(True)
    f.savefig('/Users/AFigueroa/Desktop/PyMC3/StatisticalTests/Histograms/{}_histogram.pdf'.format(isotope))
    plt.close()
    return

#for isotope in tqdm(Isotope_list):
#    print_histogram(Isotope_Data[isotope]['Deviations'],isotope)

np.save('/home/figueroa/Desktop/PyMC3/Isotopes_Statistics.npy',Isotope_Data)
# In[3]:    
def gaussian_pdf(x,a,mu,sigma):
    return a*np.exp(-(x-mu)**2/(2*(sigma**2)))

def gaussian_cdf(x,mu,sigma):
    return 0.5 * (1+math.erf((x-mu)/(sigma*np.sqrt(2))))

def Chi_squared_test(a,mu,sigma,isotope_histogram,Bin_centers,Nsamples):
    #Expected = np.ones(len(Bin_centers)) * 1/(len(Bin_centers))
    Expected = [gaussian_pdf(Bin_centers[i],a,mu,sigma) for i in range(len(Bin_centers))]
    #Expected = [(gaussian_cdf(Bin_centers[j+1],mu,sigma)-gaussian_cdf(Bin_centers[j],mu,sigma))*Nsamples for j in range(len(Bin_centers)-1)]
    
    # Automatic Chi Squared Calculation:
    
#    if all(bins > 5 for bins in isotope_histogram) and all(bins > 5 for bins in Expected):
#        auto_chisq, auto_p = stats.chisquare(isotope_histogram,Expected,ddof = len(Expected)-1)
#    else:
#        auto_chisq = 1e50
#        auto_p = -1
    auto_chisq, auto_p = stats.chisquare(isotope_histogram,Expected,ddof = len(Expected)-1)    
    # Manual Chi Squared Calculation:
    Chi_sq = [(isotope_histogram[i] - Expected[i])**2/Expected[i] for i in range(len(Bin_centers))]
    Chi_sq = sum(Chi_sq)/len(Bin_centers)
    ND = len(Bin_centers) - 3 # Degrees of Freedom
    p = np.abs(np.sqrt(2*Chi_sq) - np.sqrt((2*ND)-1))
    
    # Scipy Chi Squared Calculation:
    #Chi_sq, p = stats.chisquare(isotope_histogram,Expected,ddof=len(Bin_centers)-3)
    return Chi_sq, p, auto_chisq, auto_p

def Kolmogorov_smirnov(a,mu,sigma,isotope_data):
    Samples = 10000
    Expected = (sigma * np.random.randn(Samples)) + mu
    Cum_Exp = np.cumsum(Expected)
    Cum_Data = np.cumsum(isotope_data)
    D, p = stats.ks_2samp(isotope_data,Expected)
    return D, p

def plot_fit(a,mu,sigma,isotope_data,nbins,isotope,Nsigmas):
    Hist, Edges = np.histogram(isotope_data,nbins)
    Bin_centers = [np.mean([Edges[i+1],Edges[i]]) for i in range(nbins)] 
    N = 2# Level of sigma to delete background
    Background_level = np.where(np.logical_or(Bin_centers >= mu + N*sigma, Bin_centers <= mu - N*sigma))[0]
    Hist[Background_level] = 0                          
    plt.figure()
    plt.suptitle(r'$Y_{\mathrm{pred}} - Y_{\mathrm{sim}}$ Distribution - '+'{}'.format(isotope),fontweight='bold')
    plt.hist(isotope_data,bins=nbins,label=r'$Y_{\mathrm{pred}} - Y_{\mathrm{sim}}$')
    plt.plot(Bin_centers, gaussian_pdf(Bin_centers,a,mu,sigma),label='Fit')
    plt.grid(True)
    plt.show(True)

def gaussian_fit(isotope,isotope_data,Background_clean,N):
    Nsamples = len(isotope_data)
    limit = 8 # minimum number of samples per bin on average if it were uniformely distributed
    binmax = int(Nsamples/limit)
    a0 = 1
    mu0 = np.mean(isotope_data)
    sigma0 = np.std(isotope_data)
    bins = np.array(range(6,binmax)) #[17]
    As = []
    Mus = []
    Sigmas = []
    Chis = []
    Ps = []
    Shapiro = []
    KolmD = []
    KolmP = []
    Auto_Chi = []
    Auto_P = []
    # Redo the optimization for different bin sizes and return the parameters for which Chi-squared is minimized
    for Bin in tqdm(bins):
        isotope_histogram, Edges = np.histogram(isotope_data,Bin)
        #if np.all(isotope_histogram > 5): 
        Bin_centers = [np.mean([Edges[i+1],Edges[i]]) for i in range(Bin)]
        Nsamples = len(isotope_data)
        popt,pcov = curve_fit(gaussian_pdf,Bin_centers,isotope_histogram,p0=[a0,mu0,sigma0])
        a = popt[0]
        mu = popt[1]
        sigma = popt[2]
        As.append(a)
        Mus.append(mu)
        Sigmas.append(sigma)
        #Nvalues = np.linspace(0,6,100)
        if Background_clean:
            Background_level = np.where(np.logical_or(Bin_centers >= mu0 + N*sigma0, Bin_centers <= mu0 - N*sigma0))[0]
            isotope_histogram[Background_level] = 0
            #Non_Background = np.arange(Bin)
            #Histogram = isotope_histogram
            #while not all(bins > 5 for bins in Histogram[Non_Background]):
            #    for N in Nvalues: # Level of sigma to delete background
            #        Index_List = set(np.arange(Bin))
                    #Background_level = np.where(np.logical_or(Bin_centers >= mu0 + N*sigma0, Bin_centers <= mu0 - N*sigma0))[0]
                    #Non_Background = set([idx for idx in Index_List if idx not in Background_level])
                    #isotope_histogram[Background_level] = 0
                #if N == Nvalues[-1]:
                    #break
        Chi_sq, p, auto_chisq, auto_p = Chi_squared_test(a,mu,sigma,isotope_histogram,Bin_centers,Nsamples)
        Chis.append(Chi_sq)
        Ps.append(p)
        Auto_Chi.append(auto_chisq)
        Auto_P.append(auto_p)
        Shapiro.append(stats.shapiro(isotope_histogram))
        KD, Kp = Kolmogorov_smirnov(a,mu,sigma,isotope_data)
        KolmD.append(KD)
        KolmP.append(Kp)
        #else:
        #    continue
    
    As = np.array(As)
    Mus = np.array(Mus)
    Sigmas = np.array(Sigmas)
    Chis = np.array(Chis)
    Ps = np.array(Ps)
    KolmP = np.array(KolmP)
    minIdx = np.nanargmin(Ps)
    Nbins = bins[minIdx]
#    last_valid_bin = np.where(np.array(Auto_Chi) == 1e50)[0] - 1
#    isotope_histogram, Edges = np.histogram(isotope_data,Nbins)
#    Bin_centers = [np.mean([Edges[i+1],Edges[i]]) for i in range(Nbins)]
#    plt.figure()
#    plt.hist(isotope_data,Nbins)
#    plt.plot(Bin_centers,gaussian_pdf(Bin_centers,As[minIdx],Mus[minIdx],Sigmas[minIdx]))
#    plt.plot(Bin_centers,isotope_histogram)
#    plt.grid()
#    plt.show(True)
    print('Isotope = {} \n'.format(isotope),' Number of Bins = {} \n'.format(Nbins),'Amplitude = {} \n'.format(As[minIdx]),'Mean = {} \n'.format(Mus[minIdx]),
          'Std = {} \n'.format(Sigmas[minIdx]),'Chi_sq = {} \n'.format(Chis[minIdx]),'y = {} \n'.format(Ps[minIdx]))
    return isotope, Nbins, As, Mus, Sigmas, Chis, Ps,minIdx, bins, KolmD, KolmP#, Shapiro[minIdx][1], KolmP[minIdx]
Nsigmas = 1
isotope = 'Pu239'
Background_clean = True
isotope, Nbins, As, Mus, Sigmas, Chis, Ps,Index, bins, KolmD, KolmP = gaussian_fit(isotope,Isotope_Data[isotope]['Deviations'],Background_clean,Nsigmas)

plt.figure()
plt.suptitle('Fit information {}'.format(isotope),fontweight='bold')
plt.scatter(np.array(range(len(Chis)))+6,Chis,label=r'$\chi^{2}$')
plt.scatter(np.array(range(len(Ps)))+6,Ps,label='y_value')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.show(True)

#Index = 92
#Index = 28
#Nbins = bins[Index]
plot_fit(As[Index],Mus[Index],Sigmas[Index],Isotope_Data[isotope]['Deviations'],Nbins,isotope,Nsigmas)

def wilcoxon(isotope_name,ytest,xtest):
    Ypred = get_prediction(isotope_name, xtest)
    return stats.ranksums(ytest,Ypred)

wilcoxon([isotope],Ytest[isotope][100:],Xtest)
    
    

#print('Kolmogorov Mean',np.mean(Kolmogorov_Stat),'Kolmogorov Std',np.std(Kolmogorov_Stat))
#print('Kolmogorov P Mean',np.mean(Kolmogorov_Pvalues),'Kolmogorov P Std',np.std(Kolmogorov_Pvalues))
#print('Chisq Mean',np.mean(Chisq_Stat),'Chisq Std',np.std(Chisq_Stat))
#print('Chisq P Mean',np.mean(Chi_Pvalues),'Chisq p Std',np.std(Chi_Pvalues))

#with open('/home/mq887168/PyMC3/GPR_Statistics.txt','w') as f:
#    f.write('Kolmogorov_Mean Kolmogorov_Std Kolmogorov_P_Mean Kolmgorov_P_Std Chisq_Mean Chisq_Std Chisq_P_Mean Chisq_P_Std \n')
#    f.write(str(np.mean(Kolmogorov_Stat))+' '+str(np.std(Kolmogorov_Stat))+' '+str(np.mean(Kolmogorov_Pvalues))+' '+str(np.std(Kolmogorov_Pvalues))+
#            ' '+str(np.mean(Chisq_Stat))+' '+str(np.std(Chisq_Stat))+' '+str(np.mean(Chi_Pvalues))+' '+str(np.std(Chi_Pvalues)))

#with open('/Users/AFigueroa/Desktop/PyMC3/GPR_Statistics_expanded.txt','w') as f:
# f.write('Isotope'+' '+'Kolmogorov_D'+' '+'Kolmogorov_P'+' '+'Chisq'+' '+'Chisq_P'+'\n')
# for a in range(1229):
# f.write(str(Isotope_list[a])+' '+str(Kolmogorov_Stat[a])+' '+str(Kolmogorov_Pvalues[a])+' '+str(Chisq_Stat[a])+' '+str(Chi_Pvalues[a])+'\n')

#port = 465  # For SSL
#smtp_server = "smtp.gmail.com"
#sender_email = "afcpymail@gmail.com"  # Enter your address
#receiver_email = "figueroa@aices.rwth-aachen.de"  # Enter receiver address
#password = 'Test_Password'
#message = """\Statistical Test Terminated Succesufully"""

#context = ssl.create_default_context()
#with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
#    server.login(sender_email, password)
#    server.sendmail(sender_email, receiver_email, message)

# In[3]:

def get_chisq(isotope_name,xtest,ytest,nbins):
    Ypred = get_prediction(isotope_name,xtest)
    YpredHist, bin_edges = np.histogram(Ypred,bins=nbins)
    YtestHist,bin_edges2 = np.histogram(ytest, bin_edges)
    if all(bins > 5 for bins in YpredHist) and all(bins > 5 for bins in YtestHist): # Condition for applicability
        Chisq, Chi_p = stats.chisquare(YpredHist,YtestHist,ddof = nbins-5) #nbins - 5 # 4 kernel lengths plus an amplitude
        return Chisq, Chi_p
    else:
        return 1e50, -1
    
def plot_distributions(isotope_name, xtest,ytest,nbins):
    Ypred = get_prediction(isotope_name, xtest)
    YpredHist, bin_edges = np.histogram(Ypred,bins=nbins)
    YtestHist,bin_edges2 = np.histogram(ytest, bin_edges)
    bin_width = bin_edges[1]-bin_edges[0]
    edges_l = [edge - (0.15*bin_width) for edge in bin_edges]
    edges_r = [edge + (0.15*bin_width) for edge in bin_edges]
    w = (bin_edges[1]-bin_edges[0])/4
    plt.figure()
    plt.suptitle('{} Distribution Comparison for Serpent and GPR'.format(isotope_name[0]),fontweight='bold')
    plt.bar(edges_l[:-1], YpredHist, width=w, label='GPR Predictions Distribution')
    plt.bar(edges_r[:-1], YtestHist, width=w, label='SERPENT Simulation Distribution')
    plt.xlabel('Mass (Kg)',fontweight='bold')
    plt.ylabel('Frecuency',fontweight='bold')
    plt.legend()    
    plt.grid()
    
def chisq_analysis(isotope,Xtest,ytest):
    Nsamples = len(ytest)
    limit = 8 # minimum number of samples per bin on average if it were uniformely distributed
    binmax = int(Nsamples/limit)
    bins = np.array(range(6,binmax)) #[17]
    ChisqS = []
    ChipS = []
    for nbins in tqdm(bins):
        chi, p = get_chisq(isotope,Xtest,ytest,nbins)
        ChisqS.append(chi)
        ChipS.append(p)
    last_valid_bin = np.where(np.array(ChisqS) == 1e50)[0][0] - 1
    idx = last_valid_bin#np.nanargmax(np.array(ChisqS)[:last_valid_bin])
    #idx = last_valid_bin#np.argmin(np.array(ChisqS))
    print('Number of Bins: {} \n'.format(bins[idx])+'Chi Squared: {} \n'.format(np.array(ChisqS)[idx])+
          'P Value: {}'.format(np.array(ChipS[idx])) )
    plot_distributions(isotope,Xtest,ytest,idx)
    return bins[idx], np.array(ChisqS)[idx], np.array(ChipS)[idx],ChisqS,ChipS, last_valid_bin

isotope = 'Cs137'
bins, optChisq, optP, ChisqS, PS, last_valid_bin = chisq_analysis([isotope],Xtest,Ytest[isotope][100:])        

# In[4]
Yhist, edges = np.histogram(Ytest[isotope],bins=17)
if all(BINS > 5 for BINS in Yhist):
    print('Yes')
    # In[5]:
#for nbins in range(1,last_valid_bin):
#    plot_distributions([isotope],Xtest,Ytest[isotope][100:],nbins)
plot_distributions([isotope],Xtest,Ytest[isotope][100:],last_valid_bin)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:18:26 2020

@author: AFigueroa
"""

import numpy as np
from scipy.spatial.distance import cdist
from functools import reduce
#import warnings
#warnings.filterwarnings('ignore')
# In[]

class Kernel:

    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        self._x1 = x1
        self._x2 = x2
        self._with_gradient = with_gradient
        self._params = params
        self.K = self._computeKernel()
        self._sigma = (self._params[-1] ** 2) * np.eye(self._x1.shape[0])
        self._dsigma = 2 * self._params[-1] * np.eye(self._x1.shape[0])

        if self._with_gradient:
            self.Kgrad = self._compute_gradient()
            self.K = self.K + self._sigma
        else:
            self.Kgrad = None

    def _init_message(self):
        print(self._params)

    def _computeKernel(self):
        pass

    def _compute_gradient(self):
        pass


#######################################################################
class SQEKernel(Kernel):

    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):
        self.__sqdist = cdist(self._x1, self._x2, metric='sqeuclidean').T
        K = (self._params[0] ** 2) * np.exp(-0.5 * self.__sqdist / self._params[1] ** 2)
        return K

    def _compute_gradient(self):
        return [2 * self._params[0] * self.K,
                np.multiply(self.__sqdist / (self._params[1] ** 3), self.K),
                self._dsigma]


#######################################################################
class ASQEKernel(Kernel):

    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):

        LAMBDA = np.eye(len(self._x2[0]))
        length_scales = 1/self._params[1:-1]
        np.fill_diagonal(LAMBDA, length_scales)
        self._x1 = np.dot(self._x1, LAMBDA)
        self._x2 = np.dot(self._x2, LAMBDA)
        sqdist = cdist(self._x1, self._x2, metric='sqeuclidean').T
        K = (self._params[0]**2) * np.exp(-0.5 * sqdist)
        return K

    def _compute_gradient(self):
        g = [cdist(np.expand_dims(self._x1[:, i], -1),
                   np.expand_dims(self._x2[:, i], -1),
                   metric='sqeuclidean') / (self._params[i + 1]) \
             for i in range(self._x1.shape[1])]
        gradients = [np.multiply(g[i], self.K) for i in range(self._x1.shape[1])]
        gradients = [2 * self._params[0] * self.K] + gradients + [self._dsigma]
        return gradients

#######################################################################
class LAPKernel(Kernel):

    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):
        self.__dist = np.sqrt(np.sum(self._x1 ** 2, 1).reshape(-1, 1) +
                       np.sum(self._x2 ** 2, 1) - 2 * np.dot(self._x1, self._x2.T))
        K = (self._params[0] ** 2) * np.exp(-0.5 * self.__dist / self._params[1])
        return K.T

    def _compute_gradient(self):
        gradients = [2 * self._params[0] * self.K, np.multiply(self.__dist / (self._params[1] ** 2), self.K),
                     self._dsigma]
        return gradients


##################################################################
class ALAPKernel(Kernel):

    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):
        self.__dist = np.sqrt(cdist(self._x1 / self._params[1:-1], self._x2 / self._params[1:-1], metric='euclidean').T)
        K = (self._params[0] ** 2) * np.exp(-0.5 * self.__dist)
        return K

    def _compute_gradient(self):
        g = [cdist(np.expand_dims(self._x1[:, i] / self._params[i + 1], -1),
                   np.expand_dims(self._x2[:, i] / self._params[i + 1], -1),
                   metric='euclidean') / (self._params[i + 1]) \
             for i in range(self._x1.shape[1])]
        gradients = [np.multiply(g[i], self.K) for i in range(self._x1.shape[1])]
        gradients = [2 * self._params[0] * self.K] + gradients + [self._dsigma]
        return gradients


#######################################################################
class LinearKernel(Kernel):
    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):
        LAMBDA = np.eye(len(self._params[2:-1]))
        length_scales = 1 / self._params[2:-1]
        np.fill_diagonal(LAMBDA, length_scales)
        self._x1 = np.dot(self._x1, LAMBDA)
        self._x2 = np.dot(self._x2, LAMBDA)
        K = (self._params[0] * np.dot(self._x1, self._x2.T).T) + self._params[1]
        return K

    def _compute_gradient(self):
        gradients = [np.dot(self._x1, self._x2.T).T] + \
                    [np.eye(self.K.shape[0])] + \
                    [2 * self._params[0] *
                     np.dot(np.expand_dims(self._x1[:, i], -1)
                            , np.expand_dims(self._x2[:, i], -1).T).T / self._params[i]
                     for i in range(self._x1.shape[1])]
        gradients = gradients + [self._dsigma]
        return gradients


##################################################################
class PolyKernel(Kernel):
    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):

        LAMBDA = np.eye(len(self._params[3:-1]))
        length_scales = 1/self._params[3:-1]
        np.fill_diagonal(LAMBDA, length_scales)
        self._x1 = np.dot(self._x1, LAMBDA)
        self._x2 = np.dot(self._x2, LAMBDA)
        K = ((self._params[0]*np.dot(self._x1, self._x2.T).T)+self._params[1])**self._params[2]
        return K

    def _compute_gradient(self):
        db = self._params[2] * ((self._params[0] * np.dot(self._x1, self._x2.T)) +
                                   self._params[1]) ** (self._params[2] - 1)
        da = np.multiply(np.dot(self._x1, self._x2.T), db)
        dc = np.multiply(self.K, np.log((self._params[0] * np.dot(self._x1, self._x2.T)) + self._params[1]))
        gradients = [da, db, dc] + \
                    [2 * self._params[0] * np.multiply(np.dot(np.expand_dims(self._x1[:, i], -1),
                                                           np.expand_dims(self._x2[:, i], -1).T) / \
                                                    self._params[i], db) \
                     for i in range(self._x1.shape[1])]
        gradients = gradients + [self._dsigma]
        return gradients


###########################################################################
class AnovaLinearKernel(Kernel):
    # Does not converge
    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):
        distances = [cdist(self._x1[:, i].reshape(-1, 1), self._x2[:, i].reshape(-1, 1),
                       metric='sqeuclidean') for i in range(self._x1.shape[1])]
        kernel_objs = [1 + (self._params[i] * distance) for i, distance in enumerate(distances)]
        K = reduce(lambda k1, k2: np.dot(k1, k2), kernel_objs)
        return K

class AnovaASQEKernel(Kernel):
    # this doesnt converge
    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):
        LAMBDA = np.eye(len(self._params[self._x1.shape[1]: -1]))
        length_scales = 1 / self._params[self._x1.shape[1]: -1]
        np.fill_diagonal(LAMBDA, length_scales)
        self._x1 = np.dot(self._x1, LAMBDA)
        self._x2 = np.dot(self._x2, LAMBDA)
        distances = [cdist(self._x1[:, i].reshape(-1, 1), self._x2[:, i].reshape(-1, 1),
                           metric='euclidean') for i in range(self._x1.shape[1])]
        kernel_objs = [1 + (self._params[i] * np.exp(0.5 * distance)) for i, distance in enumerate(distances)]
        K = reduce(lambda k1, k2: np.dot(k1, k2), kernel_objs)
        return K
###########################################################################
class AnovaKernel(Kernel):
    # This kernel implementation is not correct!
    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):
        K = np.exp((-self._params[0] * cdist(self._x1, self._x2, metric='sqeuclidean')) ** self._params[1]) + \
            np.exp((-self._params[0] * cdist(self._x1 ** 2, self._x2 ** 2, metric='sqeuclidean')) ** self._params[1]) + \
            np.exp((-self._params[0] * cdist(self._x1 ** 3, self._x2 ** 3, metric='sqeuclidean')) ** self._params[1])
        # Not working
        return K

    def _compute_gradient(self):
        dd = self._params[1] * (np.exp((-self._params[0] * cdist(self._x1, self._x2, metric='sqeuclidean')) ** (self._params[1] - 1)) + \
                             np.exp((-self._params[0] * cdist(self._x1 ** 2, self._x2 ** 2, metric='sqeuclidean')) ** (
                                         self._params[1] - 1)) +
                                   np.exp(
                                 (-self._params[0] * cdist(self._x1 ** 3, self._x2 ** 3, metric='sqeuclidean')) ** (self._params[1] - 1)))
        gradients = [-np.multiply(cdist(self._x1, self._x2, metric='sqeuclidean'), dd),
                     -np.multiply(cdist(self._x1 ** 2, self._x2 ** 2, metric='sqeuclidean'), dd),
                     -np.multiply(cdist(self._x1 ** 3, self._x2 ** 3, metric='sqeuclidean'), dd), dd,
                     self._dsigma]

        return gradients


#####################################################
class SigmoidKernel(Kernel):
    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):
        K = np.tanh(self._params[0] * np.dot(self._x1, self._x2.T) + self._params[1])
        return K.T

    def _compute_gradient(self):
        sech2 = 1 / (np.cosh(self._params[0] * np.dot(self._x1, self._x2.T) + self._params[1]) ** 2)
        gradients = [np.multiply(np.dot(self._x1, self._x2.T), sech2), sech2,
                     self._dsigma]
        return gradients


######################################################
class RQKernel(Kernel):
    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):
        LAMBDA = np.eye(len(self._x1[0]))
        length_scales = 1 / self._params[1:-1]
        np.fill_diagonal(LAMBDA, length_scales)
        self._x1 = np.dot(self._x1, LAMBDA)
        self._x2 = np.dot(self._x2, LAMBDA)
        self.__sqdist = cdist(self._x1, self._x2, metric='sqeuclidean').T
        K = 1 - (self.__sqdist / (self.__sqdist + (self._params[0]) ** 2))
        return K

    def _compute_gradient(self):
        denom = 1 / ((self.__sqdist + (self._params[0]) ** 2)) ** 2
        g = [cdist(np.expand_dims(self._x1[:, i], -1),
                   np.expand_dims(self._x2[:, i], -1),
                   metric='sqeuclidean') / (self._params[i + 1]) \
             for i in range(self._x1.shape[1])]
        gradients = [-2 * denom * g[i] * (self._params[0]) ** 2 for i in range(self._x1.shape[1])]
        gradients = [2 * self._params[0] * self.__sqdist * denom] + gradients + \
                    [self._dsigma]
        return gradients


######################################################
class SRQKernel(Kernel):
    # rarely converges
    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):
        LAMBDA = np.eye(len(self._x1[0]))
        length_scales = 1 / self._params[:-1]
        np.fill_diagonal(LAMBDA, length_scales)
        self._x1 = np.dot(self._x1, LAMBDA)
        self._x2 = np.dot(self._x2, LAMBDA)
        sqdist = cdist(self._x1, self._x2, metric='sqeuclidean').T
        K = (1/(1 + (sqdist/self._params[-1])) )**self._params[-1]
        return K
    # Here is missing a method for computing these gradients


####################################################
class MultiQuadKernel(Kernel):
    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):
        LAMBDA = np.eye(len(self._x1[0]))
        length_scales = 1 / self._params[1:-1]
        np.fill_diagonal(LAMBDA, length_scales)
        self._x1 = np.dot(self._x1, LAMBDA)
        self._x2 = np.dot(self._x2, LAMBDA)
        sqdist = cdist(self._x1, self._x2, metric='sqeuclidean').T
        K = np.sqrt(sqdist + (self._params[0] ** 2))
        return K

    def _compute_gradient(self):
        g = [cdist(np.expand_dims(self._x1[:, i], -1),
                   np.expand_dims(self._x2[:, i], -1),
                   metric='sqeuclidean') / (self._params[i + 1])
             for i in range(self._x1.shape[1])]
        gradients = [np.multiply(g[i], 1 / self.K) for i in range(self._x1.shape[1])]
        gradients = [self._params[0] / self.K] + gradients + \
                    [self._dsigma]
        return gradients


###########################################################
class InvMultiQuadKernel(Kernel):
    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):
        LAMBDA = np.eye(len(self._x1[0]))
        length_scales = 1 / self._params[1:-1]
        np.fill_diagonal(LAMBDA, length_scales)
        self._x1 = np.dot(self._x1, LAMBDA)
        self._x2 = np.dot(self._x2, LAMBDA)
        sqdist = np.sqrt(cdist(self._x1, self._x2, metric='sqeuclidean').T + (self._params[0] ** 2))
        K = 1 / sqdist
        return K

    def _compute_gradient(self):
        g = [cdist(np.expand_dims(self._x1[:, i], -1),
                   np.expand_dims(self._x2[:, i], -1),
                   metric='sqeuclidean') / (self._params[i + 1]) \
             for i in range(self._x1.shape[1])]
        gradients = [-np.multiply(g[i], self.K ** 3) for i in range(self._x1.shape[1])]
        gradients = [-self._params[0] * (self.K ** 3)] + gradients + \
                    [self._dsigma]
        return gradients


############################################################
class WaveKernel(Kernel):
    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):
        self.__dist = cdist(self._x1, self._x2, metric='euclidean')
        K = ((self._params[0] / self.__dist) * np.sin(self.__dist / self._params[0]))
        return K

    def _compute_gradient(self): # Check for errors here
        arg = self.__dist / self._params[0]
        gradients = (1 / self.__dist) * (np.sin(arg) - (np.cos(arg) / (self._params[0]) ** 2)) + \
                    [self._dsigma]
        return gradients


#############################################################
class PowerKernel(Kernel):
    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):
        self.__dist = cdist(self._x1 / self._params[1:-1], self._x2 / self._params[1:-1], metric='euclidean')
        K = self.__dist ** self._params[0]
        return K

    def _compute_gradient(self):
        dd = self._params[0] * self.__dist ** (self._params[0] - 1)
        g = [cdist(np.expand_dims(self._x1[:, i] / self._params[i + 1], -1),
                   np.expand_dims(self._x2[:, i] / self._params[i + 1], -1),
                   metric='sqeuclidean') / (self._params[i + 1]) \
             for i in range(self._x1.shape[1])]
        gradients = [np.multiply(g[i], dd) for i in range(self._x1.shape[1])]
        gradients = [dd] + gradients + [self._dsigma]
        return gradients


##################################################################
class LogKernel(Kernel):
    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):
        self.__dist = cdist(self._x1, self._x2, metric='euclidean').T
        K = -np.log((self.__dist ** self._params[0]) + 1)
        return K

    def _compute_gradient(self):
        arg = self.__dist ** self._params[0]
        gradients = arg * np.log(self.__dist) / (arg + 1) + \
                    [self._dsigma]
        return gradients


###############################################################
class CauchyKernel(Kernel):
    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):
        self.__sqdist = cdist(self._x1, self._x2, metric='sqeuclidean').T
        K = 1 / (1 + (self.__sqdist / (self._params[0] ** 2)))
        return K

    def _compute_gradient(self):
        gradients = [self.__sqdist / (((self._params[0] ** 2) + self.__sqdist) ** 2)] + \
                    [self._dsigma]
        return gradients


################################################################
class TstudentKernel(Kernel):
    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 params: np.ndarray, with_gradient: bool):
        super().__init__(x1=x1, x2=x2, params=params, with_gradient=with_gradient)

    def _computeKernel(self):
        LAMBDA = np.eye(len(self._x1[0]))
        length_scales = 1 / self._params[1:]
        np.fill_diagonal(LAMBDA, length_scales)
        self._x1_orig = np.copy(self._x1)
        self._x2_orig = np.copy(self._x2)
        self._x1 = np.dot(self._x1, LAMBDA)
        self._x2 = np.dot(self._x2, LAMBDA)
        self.__sqdist = cdist(self._x1, self._x2, metric='euclidean').T
        K = 1 / (1 + (self.__sqdist ** self._params[0]))
        return K

    def _compute_gradient(self):
        # This gradient is not correct
        da = -(self.__sqdist ** self._params[0]) * np.log(self.__sqdist**2) * 0.5 * (self.K ** 2)
        arg = self._params[0] * (self.__sqdist ** (self._params[0] - 1)) * (self.K ** 2)

        gradients = [da] + [np.dot(cdist(
            np.expand_dims(self._x1[:, i], -1), np.expand_dims(self._x2[:, i], -1), metric='sqeuclidean') /
                                        (self._params[i + 1]),
                                        arg)
                            for i in range(self._x1.shape[1])] + [self._dsigma]
        return gradients


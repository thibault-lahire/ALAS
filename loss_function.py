#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:15:37 2019

@author: lahire
"""

import numpy as np


def phi(w):
    return 1/(1+np.exp(-w))


def phiprim(w):
    if np.exp(-w) < 10**(16):
        if np.exp(-w) > 10**(-16):
            f = np.exp(-w)/((1+np.exp(-w))**2)
        else:
            f = np.exp(-w)
    else:
        f = np.exp(w)
    return f


def phi2prim(w):
    if np.exp(-w) < 10**(16):
        if np.exp(-w) > 10**(-16):
            f = -np.exp(w)*(np.exp(w)-1)/((1+np.exp(w))**3)
        else:
            f = -np.exp(-w)
    else:
        f = np.exp(w)
    return f




def loss(x, S, A, y, arg=1, penalization=False):
    '''
    Input parameters :
    x: array, point for which the loss must be computed
    S: list, subset of indices
    A: matrix of the datapoints
    y: vector of labels
    arg: int, number of output parameters
    penalization: bool, use of a penalized empirical loss function or not

    Returns:
    value of the loss function at point x if arg==1
    value of the loss function and its gradient at point x if arg==2
    value of the loss function, its gradient, and its Hessian at point x if arg==3
    '''
    d = np.size(A, 1)
    card_S = len(S)
    f = 0
    if arg == 2 or arg == 3:
        g = np.zeros(d)
    if arg == 3:
        H = np.zeros((d, d))

    for i in range(card_S):
        y_S_i = y[S[i]]
        A_S_i = A[S[i]]
        scal_prod = np.dot(x, A_S_i)
        f += (y_S_i - phi(scal_prod))**2
        if arg == 2 or arg == 3:
            g += 2*A_S_i*phiprim(scal_prod)*(phi(scal_prod)-y_S_i)
        if arg == 3:
            H += 2*((phi(scal_prod)-y_S_i)*phi2prim(scal_prod)+\
                    phiprim(scal_prod)**2)*np.dot(np.transpose([A_S_i]),\
                           [A_S_i])
    if penalization == True:
        f += 0.5*np.linalg.norm(x)**2
    f = f/card_S

    if arg == 3:
        H = 0.5*(H + np.transpose(H))
        if penalization == True:
            g += x
            H += np.eye(d)
        g = g/card_S
        H = H/card_S

        return f, g, H

    elif arg == 2:
        if penalization == True:
            g += x
        g = g/card_S
        return f, g

    else:
        return f

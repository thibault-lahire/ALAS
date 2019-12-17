#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:13:31 2019

@author: lahire
"""

import math
import time
import numpy as np

def optimizer(fun, sample_size, A, y, penalization, maxit=9):
    '''
    Input parameters :
    fun : cost function
    sample_size : float, percentage of the dataset (0 <= sample_size <= 1)
    A : matrix of datapoints
    y : vector of labels
    penalization : bool, indicate whether to use a penalization or not in the
    loss function
    maxit : int, max number of iterations

    Return :
    x : solution point found
    f : value of the cost function on the point x
    numf : number of function calls
    numg : number of gradient calls
    ex : number of the exit procedure
    it : number of iterations
    ng : value of the gradient norm
    duration : array containing the duration of each iteration
    real_value_f : array containing the value of the entire empirical loss function at the end of each iteration
    points : array containing the points at each iteration : {x_k, k being the iteration number}
    real_value_ng : array containing the gradient norm at each iteration : {||g_k||, k being the iteration number}
    '''
    
    #param
    gtol = 10**(-5)
    InitialStepSize = 1
    N = A.shape[0]
    dim = A.shape[1]
    x0 = np.zeros(dim)
    
    initialsamplingsize = math.floor(sample_size*N) 
    
    S0 = np.random.choice(N, initialsamplingsize, replace=False)
    
    points = [x0]
    duration = []

    
    #Evaluate function in the starting point
    [f0, g0] = fun(x0, S0, A, y, 2, penalization)
    ng0 = np.linalg.norm(g0)
    
    
    # Entire function
    f_real, g_real = fun(x0, np.asarray([i for i in range(N)]), A, y, 2, penalization)
    real_value_f = [f_real]
    
    
    it = 1
    EpochStep = math.floor(N/initialsamplingsize)
    numf = 1
    numg = 1
    
    #Initialization
    S = S0
    x = x0
    f = f0
    g = g0
    ng = ng0
    G = np.dot(np.transpose([g]), [g])
    
    ng_real = np.linalg.norm(g_real)
    real_value_ng = [ng_real]


    # Check the stopping criteria with some confidence
    flag_cv = (ng <= gtol)    
    if flag_cv:
        ex = 1
        duration = np.asarray(duration)
        real_value_f = np.asarray(real_value_f)
        points = np.asarray(points)     
        real_value_ng = np.asarray(real_value_ng)
        return x, f, numf, numg, ex, it, ng, duration, real_value_f, points, real_value_ng

    J = 0

    #main loop
    while (True):
        
        print(it)
        
        tmps1 = time.time()
        
        # choose the right direction
        d = -g
        for i in range(dim):
            d[i] = -g[i]/np.sqrt(G[i][i]) 
            
        x = x + InitialStepSize*d
        
        # Update the sampling set S_k
        S = np.random.choice(N, initialsamplingsize, replace=False)
        [f, g] = fun(x, S, A, y, 2, penalization)
        G += np.dot(np.transpose([g]),[g])
        numf += 1
        numg += 1
        ng = np.linalg.norm(g)
        
        
        if it>maxit:
            ex = 0
            duration = np.asarray(duration)
            real_value_f = np.asarray(real_value_f)
            points = np.asarray(points)
            real_value_ng = np.asarray(real_value_ng)
            return x, f, numf, numg, ex, it, ng, duration, real_value_f, points, real_value_ng
        
        it += 1
        
        tmps2 = time.time()
        delta_time = tmps2 - tmps1
        duration.append(delta_time)
        
        points.append(x)
        
        f_real, g_real = fun(x, np.asarray([i for i in range(N)]), A, y, 2, penalization)
        real_value_f.append(f_real)
        real_value_ng.append(np.linalg.norm(g_real))

        
        flag_cv = (ng <= gtol)
#        flag_cv_2 = (f<0.4)
        if flag_cv : #or flag_cv_2:
            J += 1
        else:
            J = 0
        if J == EpochStep:
            ex = 1
            duration = np.asarray(duration)
            real_value_f = np.asarray(real_value_f) 
            points = np.asarray(points)
            real_value_ng = np.asarray(real_value_ng)
            return x, f, numf, numg, ex, it, ng, duration, real_value_f, points, real_value_ng


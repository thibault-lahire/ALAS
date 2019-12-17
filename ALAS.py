#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:18:36 2019

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
    numH : number of Hessian calls
    ex : number of the exit procedure
    it : number of iterations
    ng : value of the gradient norm
    lambdA : value of the smallest eigenvalue of the Hessian matrix
    duration : array containing the duration of each iteration
    direction_used : array containing the directions used at each iteration
    real_value_f : array containing the value of the entire empirical loss
    function at the end of each iteration
    points : array containing the points at each iteration : {x_k, k being
    the iteration number}
    real_value_ng : array containing the gradient norm at each iteration :
    {||g_k||, k being the iteration number}
    '''

    N = A.shape[0]
    dim = A.shape[1]

    #param
    gtol = 10**(-5)
    Htol = np.sqrt(gtol)
    theta = 0.9
    eta = 10**(-3)
    x0 = np.zeros(dim)

    Htol_aux = Htol

    initialsamplingsize = math.floor(sample_size*N)

    S0 = np.random.choice(N, initialsamplingsize, replace=False)


    points = [x0]
    duration = []
    direction_used = []

    #Evaluate function in the starting point
    n = np.size(x0)
    [f0, g0, H0] = fun(x0, S0, A, y, 3, penalization)
    D, VV = np.linalg.eig(H0)
    D = np.real(D)
    j = np.argmin(D)
    v0 = VV[:, j]
    v0 = np.real(v0)
    lambda0 = D[j]
    ng0 = np.linalg.norm(g0)

    # Entire function
    f_real, g_real = fun(x0, np.asarray([i for i in range(N)]), A, y, 2, penalization)
    real_value_f = [f_real]

    it = 1
    EpochStep = math.floor(N/initialsamplingsize)
    numf = 1
    numg = 1
    numH = 1

    #Initialization
    S = S0
    x = x0
    f = f0
    g = g0
    H = H0
    v = v0
    lambdA = lambda0
    ng = ng0
    sqrt_ng = np.sqrt(ng)

    ng_real = np.linalg.norm(g_real)
    real_value_ng = [ng_real]

    # Check the stopping criteria with some confidence
    flag_cv = (ng <= gtol) and (lambdA > -Htol)
    if flag_cv:
        J = 1
    else:
        J = 0

    if J == EpochStep:
        ex = 1
        duration = np.asarray(duration)
        direction_used = np.asarray(direction_used)
        real_value_f = np.asarray(real_value_f)
        points = np.asarray(points)
        real_value_ng = np.asarray(real_value_ng)
        return x, f, numf, numg, numH, ex, it, ng, lambdA, duration, \
    direction_used, real_value_f, points, real_value_ng


    #main loop
    while True:
        print(it)
        tmps1 = time.time()

        Htol_aux = Htol

        if lambdA < -Htol:
            d = -lambdA*v
            direction_type = 'Negative Curvature'
            if np.vdot(d, g) >= 0:
                d = -d
        elif lambdA > sqrt_ng:
            d = -np.dot(np.linalg.inv(H), g)
            direction_type = 'Newton '
        else:
            d = -np.linalg.solve(H+Htol_aux*np.eye(n), g)
            direction_type = 'Regularized Newton'

        direction_used.append(direction_type)

        if lambdA >= -Htol and ng == 0:
            alpha = 0
        else:
            # Compute the trial point and evaluate function
            if it == 1:
                alpha = 1
                alphat = 1

            if it > 1:
                alphat = min(2*alpha, 1)

            xt = x+alphat*d
            nd = np.linalg.norm(d)
            ft = fun(xt, S, A, y, arg=1, penalization=penalization)
            numf = numf+1

            # Proceed the Line Search strategy
            while ft - f > -eta*(alphat**3*nd**3)/6:
                # Update the trial point and evaluate function
                alphat = alphat*theta
                xt = x+alphat*d
                ft = fun(xt, S, A, y, 1, penalization)
                numf = numf+1

            alpha = alphat
            x = xt


        S = np.random.choice(N, initialsamplingsize, replace=False)
        [f, g, H] = fun(x, S, A, y, 3, penalization)
        numg = numg+1
        numH = numH+1
        D, VV = np.linalg.eig(H)
        D = np.real(D)
        j = np.argmin(D)
        v = VV[:, j]
        v = np.real(v)
        lambdA = D[j]
        ng = np.linalg.norm(g)
        sqrt_ng = np.sqrt(ng)


        if it > maxit:
            ex = 0
            duration = np.asarray(duration)
            direction_used = np.asarray(direction_used)
            real_value_f = np.asarray(real_value_f)
            points = np.asarray(points)
            real_value_ng = np.asarray(real_value_ng)
            return x, f, numf, numg, numH, ex, it, ng, lambdA, duration, \
        direction_used, real_value_f, points, real_value_ng

        it = it + 1
        tmps2 = time.time()
        delta_time = tmps2 - tmps1
        duration.append(delta_time)

        points.append(x)

        f_real, g_real = fun(x, np.asarray([i for i in range(N)]), A, y, 2, penalization)
        real_value_f.append(f_real)
        real_value_ng.append(np.linalg.norm(g_real))


        flag_cv = (ng <= gtol) and (lambdA > -Htol)
        if flag_cv == True:
            J += 1
        else:
            J = 0
        if J == EpochStep:
            ex = 1
            duration = np.asarray(duration)
            direction_used = np.asarray(direction_used)
            real_value_f = np.asarray(real_value_f)
            points = np.asarray(points)
            real_value_ng = np.asarray(real_value_ng)
            return x, f, numf, numg, numH, ex, it, ng, lambdA, duration, \
        direction_used, real_value_f, points, real_value_ng

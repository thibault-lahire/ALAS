#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:25:40 2019

@author: lahire
"""
import time
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from data_from_text import extract_data
from ALAS import optimizer as alas
from SGD import optimizer as sgd
from ADAGRAD import optimizer as adagrad
from loss_function import loss


def run_simulation(filename, N, dim, sample_size, penalization, maxit=9):
    '''
    Input parameters:
    filename: str, name of the file containing the datapoints
    N: int, number of datapoints
    dim: int, dimension
    sample_size: float, percentage of the dataset (0 <= sample_size <= 1)
    penalization: bool, indicate whether to use a penalization or not in the
    loss function
    maxit: int, max number of iterations
    '''
    percentage = str(int(100*sample_size))
    A, y = extract_data(filename, N, dim)

    tmps1 = time.time()
    res_alas = alas(loss, sample_size, A, y, penalization, maxit)
    tmps2 = time.time()
    dur = tmps2 - tmps1
    label_alas = 'ALAS, ' + percentage + '%, time = ' + str(dur)[:5] + ' s'
    print("Time for running ALAS =", dur)    
    
    tmps1 = time.time()
    res_sgd = sgd(loss, sample_size, A, y, penalization, maxit)
    tmps2 = time.time()
    dur = tmps2 - tmps1
    label_sgd = 'SGD, ' + percentage + '%, time = ' + str(dur)[:5] + ' s'
    print("Time for running SGD =", dur) 

    tmps1 = time.time()
    res_adagrad = adagrad(loss, sample_size, A, y, penalization, maxit)
    tmps2 = time.time()
    dur = tmps2 - tmps1
    label_adagrad = 'ADAGRAD, ' + percentage + '%, time = ' + str(dur)[:5] + ' s'
    print("Time for running ADAGRAD =", dur)     



    ## Decrease of the real value of the cost function
    real_value_f_alas = res_alas[11]
    real_value_f_sgd = res_sgd[8]
    real_value_f_adagrad = res_adagrad[8]
    iterations = np.asarray([i + 1 for i in range(len(real_value_f_alas))])
        
    plt.plot(iterations,real_value_f_alas,label=label_alas)
    plt.plot(iterations,real_value_f_sgd,label=label_sgd)
    plt.plot(iterations,real_value_f_adagrad,label=label_adagrad)
    
    plt.loglog()
    plt.grid(True, which="both", linestyle='--')
    plt.suptitle('Comparison', fontweight='bold', fontsize=14)
    plt.title('Evolution of the value of the empirical loss function', loc='center', fontsize=8)
    plt.xlabel("Iterations")
    plt.ylabel("Real value of F")
    plt.legend()
    #plt.savefig("comparison_on_f")
    plt.show()
    

    
    ## Decrease of the real value of the gradient norm
    real_value_ng_alas = res_alas[13]
    real_value_ng_sgd = res_sgd[10]
    real_value_ng_adagrad = res_adagrad[10]        
    
    plt.plot(iterations,real_value_ng_alas,label=label_alas)
    plt.plot(iterations,real_value_ng_sgd,label=label_sgd)
    plt.plot(iterations,real_value_ng_adagrad,label=label_adagrad)
    
    plt.loglog()
    plt.grid(True, which="both", linestyle='--')
    plt.suptitle('Comparison', fontweight='bold', fontsize=14)
    plt.title('Evolution of the value of the gradient of the objective function', loc='center', fontsize=8)
    plt.xlabel("Iterations")
    plt.ylabel("Real value of ||g||")
    plt.legend()
    #plt.savefig("comparison_on_ng")
    plt.show()



    
if __name__ == '__main__':
    filename = 'data/ijcnn1.txt'
    N = 49990
    dim = 22
    sample_size = 0.01
    penalization = False
    run_simulation(filename, N, dim, sample_size, penalization)
    
    
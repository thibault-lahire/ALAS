#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:40:31 2019

@author: lahire
"""


import numpy as np

def extract_data(name: str, n: int, d: int):
    '''
    Input parameters :
    name: str, name of the file containing the datapoints
    n: int, number of datapoints
    d: int, dimension of the problem

    Returns:
    A: matrix of shape (n,d) containing the datapoints
    y : vector of size n containing the labels
    '''
    A = np.zeros((n, d))
    y = np.zeros(n)

    file = open(name, 'r', encoding='ASCII')
    lines = file.readlines()


    intab = ":"
    outtab = " "
    trantab = str.maketrans(intab, outtab)


    for i in range(len(lines)):
        chain = lines[i].translate(trantab).split()
        y[i] = chain[0]
        for j in range(1, len(chain), 2):
            elem = int(chain[j])
            for k in range(1, d+1):
                if elem == k:
                    A[i][elem-1] = chain[j+1]

    for i in range(n):
        if y[i] < 0:
            y[i] = 0

    return A, y

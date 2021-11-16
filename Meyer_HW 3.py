# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 20:27:57 2021

@author: smeyer
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
x = 1
function_appro = (x+1)**0.5 - x**0.5
error = np.abs(function_appro)
function_appro_list = [function_appro]
error_list = [error]

error_target = 1e-7/2
max_x = 1e5

while (error > error_target) & (x < max_x):
    x += 1
    temp = copy.copy(function_appro)
    function_appro = (x+1)**0.5 - x**0.5
    function_appro_list.append(function_appro)
    error = np.abs(function_appro - temp)
    error_list.append(error)
    
    print('Current x is %d'%x)
    print('Current f(x) is %f'%function_appro)
    print('Current error is %f' %error)

plt.figure(figsize=(5,5))
plt.plot(np.arange(len(function_appro_list)),function_appro_list)
plt.xlabel('x')
plt.ylabel('approximated f(x)')
plt.legend(['Approx'])
plt.show()
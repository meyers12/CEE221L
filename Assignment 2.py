# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:40:21 2021

@author: smeyer
"""

import numpy as np
import matplotlib.pyplot as plt

# list
test_list = [12,2,3]
nested_list = [[1,2],'test',123]
string = 'absqwertyuiop'
# text ascii
print(string[0:3])

# convert to ndarray
a = np.array(test_list)
# tuple do it once never change
size = (3,4)
b = np.random.randn(3,4)

# operations
# broadcasting
c = b + 1
d = b ** 3
d = np.abs(b)
e = np.sqrt(d)

x = b @ d.T

flag = (b > 0) & (b < 0.5)
flag = (b < 0) | (b > 0.5)
pos_num = b[flag]

x_n = 0
x_n1 = 1
y = 0
for i in np.arange(10):
    y = x_n + x_n1
    x_n = x_n1
    x_n1 = y
    print(y)

x = np.zeros(10)
for i in np.arange(10):
    if i == 0:
        x[i] = 1
    elif i == 1:
        x[i] = 2
    else:
        x[i] = x[i-2] * x[i-1]
print(x)
plt.figure(figsize=(10,6))
plt.plot(np.arange(10), np.log(x), 'ro')

x = [0]
i = 0
while x[-1] < 1000:
    i = i + 1
    if i == 1:
        x.append(1)
    else:
        x.append(x[-1]+x[-2])
        
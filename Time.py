# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:48:50 2018

@author: mrsom
"""
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('time measurement.txt', skiprows=12)
data_60 = np.loadtxt('60 degrees.txt', skiprows=12)

def findmax(data, ran):
    datacut = data[ran[0]:ran[1]]
    y = 5000
    i = 0
    while i < len(datacut):
        if datacut[i] > y:
            j = i+ran[0]
            break
        i += 1
    print('first index: ',j-5)
    return data[j-5:j+5]

def linearfit(x, a, b):
    return a*x + b

'''
(a,b), c = optimize.curve_fit(linearfit, channels, delays, p0=[80./7000,5.])

fit = []
for i in range(len(data)):
    fit.append(linearfit(i, a, b))
    
plt.figure()
plt.title(r'Time')
plt.ylabel(r'number of events')
plt.xlabel(r'Channel')
plt.plot(range(len(data)),data)
plt.xlim(0,len(data))
#plt.legend(loc=1)
plt.show()

plt.figure()
plt.title(r'Time - channel')
plt.ylabel(r'delay')
plt.xlabel(r'Channel')
plt.plot(channels, delays,'+')
plt.plot(range(len(data)), fit)
plt.xlim(0,len(data))
#plt.legend(loc=1)
plt.show()
'''

delays = [4, 12, 28, 50, 82]
channels = [29, 560, 1791, 4290, 6800]   #corresponding indices
final_params_a_b = (0.0111343099457, 5.20416900548) #fitted parameters, first position: a, second position: b

def linearrescale(data, a, b):
    scale_old = range(len(data))
    scale_new = []
    for x in scale_old:
        scale_new.append(a*x + b)
    return scale_new

plt.figure()
plt.title(r'60° measurement')
plt.ylabel(r'number of events')
plt.xlabel(r'Channel')
plt.plot(range(len(data_60)),data_60)
plt.xlim(0,len(data))

plt.figure()
plt.title(r'60° measurement')
plt.ylabel(r'number of events')
plt.xlabel(r'time')
plt.plot(linearrescale(data_60, final_params_a_b[0], final_params_a_b[1]),data_60)
plt.plot([48,48],[0,12], 'k', linewidth = 3)
#plt.xlim(0,len(data))

plt.show()






# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 17:13:13 2017

@author: mrsom
"""


import numpy as np
import matplotlib.pyplot as plt

def findmax(data, ran):
    datacut = data[ran[0]:ran[1]]
    y = max(datacut)
    i = 0
    while i < len(datacut):
        if datacut[i] <= y+1 and datacut[i] >= y-1:
            return (i+ran[0], y)
        i += 1

data_small = np.loadtxt('spectrum_small.txt', skiprows=12)
data_large = np.loadtxt('spectrum_big.txt', skiprows=12)

peak1_small= findmax(data_small, (3500,4000))
peak2_small= findmax(data_small, (4000,5000))
peak1_large= findmax(data_large, (3300,3800))
peak2_large= findmax(data_large, (3800,4500))

print(peak1_small, peak2_small, peak1_large, peak2_large)

ymax_small = 400
ymax_large = 500

plt.figure()
plt.title(r'Energy Spectrum at small detector')
plt.ylabel(r'number of events')
plt.xlabel(r'Channel')
plt.plot(range(len(data_small)),data_small, linewidth = 0.1)
plt.plot([peak1_small[0],peak1_small[0]],[0,ymax_small])
plt.plot([peak2_small[0],peak2_small[0]],[0,ymax_small])
plt.xlim(0,len(data_small))
plt.ylim(0, ymax_small)
#plt.legend(loc=1)
#plt.savefig('Diffraction.jpg', frameon=True, dpi=480)

plt.figure()
plt.title(r'Energy Spectrum at large detector')
plt.ylabel(r'number of events')
plt.xlabel(r'Channel')
plt.plot(range(len(data_small)),data_large, linewidth = 0.1)
plt.plot([peak1_large[0],peak1_large[0]],[0,ymax_large])
plt.plot([peak2_large[0],peak2_large[0]],[0,ymax_large])
plt.xlim(0,len(data_large))
plt.ylim(0, ymax_large)
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

def rescale(data, point1, point2):
    scale_old = range(len(data))
    factor = (point2[1] - point1[1]) / (point2[0] - point1[0])
    shift = point1[1] - point1[0]*factor
    scale_new = []
    for i in scale_old:
        scale_new.append(shift + i*factor)
    return scale_new

t_small = 228
t_large = 196

data_small = list(map(lambda x : x/t_small, np.loadtxt('spectrum_small.txt', skiprows=12)))
data_large = list(map(lambda x : x/t_large, np.loadtxt('spectrum_big.txt', skiprows=12)))

'''
peak1_small= findmax(data_small, (3500,4000))
peak2_small= findmax(data_small, (4000,5000))
peak1_large= findmax(data_large, (3300,3800))
peak2_large= findmax(data_large, (3800,4500))
'''

manual_peak1_pos_small=3790
manual_peak2_pos_small=4300
manual_peak1_pos_large=3490
manual_peak2_pos_large=3939


point1_small = (manual_peak1_pos_small, 1172)
point2_small = (manual_peak2_pos_small, 1332)

point1_large = (manual_peak1_pos_large, 1172)
point2_large = (manual_peak2_pos_large, 1332)


ymax_small = 300/228    
ymax_large = 500/196
'''
plt.figure()
plt.title(r'Energy Spectrum at small detector')
plt.ylabel(r'number of events')
plt.xlabel(r'Channel')
plt.plot(range(len(data_small)),data_small, linewidth = 0.1)
plt.plot([manual_peak1_pos_small,manual_peak1_pos_small],[0,ymax_small])
plt.plot([manual_peak2_pos_small,manual_peak2_pos_small],[0,ymax_small])
plt.xlim(0,len(data_small))
plt.ylim(0, ymax_small)
#plt.legend(loc=1)
#plt.savefig('Diffraction.jpg', frameon=True, dpi=480)
plt.show()
'''
ticks = range(0,1800,200)

plt.figure()
plt.title(r'Energy spectrum measured by small detector')
plt.ylabel(r'Events per time [Hz]')
plt.xlabel(r'Energy [keV]')
plt.plot(rescale(data_small, point1_small, point2_small),data_small,'k', linewidth = 0.6 ,label='energy spectrum')
plt.plot([1172,1172],[0,ymax_small], label='first gamma ray')
plt.plot([1332,1332],[0,ymax_small], label='second gamma ray')
plt.plot([962,962],[0,ymax_small], label='first Compton edge')
plt.plot([1118,1118],[0,ymax_small], label='second Compton edge')
plt.plot([200,200],[0,ymax_small], label='backscattering peak')
plt.xlim(46,2000)
plt.ylim(0, ymax_small)
plt.xticks(range(200,2001,200),range(200,2001,200))
plt.legend(loc=1)
#plt.savefig('Diffraction.jpg', frameon=True, dpi=480)
#plt.show()

'''
plt.figure()
plt.title(r'Energy Spectrum at large detector')
plt.ylabel(r'number of events')
plt.xlabel(r'Channel')
plt.plot(range(len(data_small)),data_large, linewidth = 0.1)
plt.plot([manual_peak1_pos_large,manual_peak1_pos_large],[0,ymax_large])
plt.plot([manual_peak2_pos_large,manual_peak2_pos_large],[0,ymax_large])
plt.xlim(0,len(data_large))
plt.ylim(0, ymax_large)
plt.show()
'''

plt.figure()
plt.title(r'Energy spectrum measured by large detector')
plt.ylabel(r'Events per time [Hz]')
plt.xlabel(r'Energy [keV]')
plt.plot(rescale(data_large, point1_large, point2_large),data_large,'k', linewidth = 0.6, label='energy spectrum')
plt.plot([1172,1172],[0,ymax_large], label='first gamma ray')
plt.plot([1332,1332],[0,ymax_large], label='second gamma ray')
plt.plot([962,962],[0,ymax_large], label='first Compton edge')
plt.plot([1118,1118],[0,ymax_large], label='second Compton edge')
plt.plot([150,150],[0,ymax_large], label='backscattering peak')
plt.xlim(0,2000)
plt.ylim(0, ymax_large)
plt.xticks(range(200,2001,200),range(200,2001,200))
plt.legend(loc=1)
plt.show()

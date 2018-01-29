#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from read import *


# Gaussian
def gaussian(x,amp,mu,sigma):
    return amp*np.exp(-(x-np.float(mu))**2/(2*sigma**2))


# Model Function
def model(x,a0,a1,a2):
    return a0 + a1*(np.cos(x*np.pi/180.))**2 + a2*(np.cos(x*np.pi/180.))**4


# Rescaler
def linearrescale(data, a, b):
    scale_old = range(len(data))
    scale_new = []
    for x in scale_old:
        scale_new.append(a*x + b)
    return scale_new


# Index Finder: finds entry in array that is closest to value and returns its index
def index(data,value):
    index = 0
    for i in range(len(data)):
        if np.fabs(data[i]-value) <= np.fabs(data[index]-value):
            index = i
    return index


# Merging Bins, size: how many bins should be merged, output: new xdata, new ydata
def binmerge(xdata,ydata,size):
    i = 0
    newx,newy = [],[]
    while i <= len(xdata)-size:
        ybuf = 0
        xbuf = 0
        for j in range(size):
            ybuf += ydata[i+j]
            xbuf += xdata[i+j]
        newy.append(1.*ybuf)
        newx.append(1.*xbuf/size)
        i += size
    return np.array(newx),np.array(newy)



# Background Subtracter: calculates mean of all bins in range of one sigma outside n*sigma
# and subtracts that value from all entrys, returns initial arrays and value of background
def subtr_backg(xdata,ydata,n):
    newy = []
    popt,pcov = curve_fit(gaussian,xdata,ydata,p0=paramg)
    mu = popt[1]
    sigma = popt[2]
    background = 0
    #imu = index(xdata,mu)
    #iunten = index(xdata,mu-n*sigma)
    #ioben = index(xdata,mu+n*sigma)
    i1,i2,i3,i4 = index(xdata,mu-n*sigma-2),index(xdata,mu-n*sigma),index(xdata,mu+n*sigma),index(xdata,mu+n*sigma+2)
    for j in ydata[i1:i2]:
        background += j
    for j in ydata[i3:i4]:
        background += j
    background = background/(len(ydata[i1:i2])+len(ydata[i3:i4]))
    for k in ydata:
        newy.append(k-background)
    return xdata,np.array(newy),background



# Rescale Time Scale
a,b = 0.0111343099457,5.20416900548
channelaxis = range(num[0]+1)
timeaxis = linearrescale(channelaxis,a,b)


# Change Units from events/20min to events/s
yraw = np.array(val)/1200.
scaler = scaler/1200.
scalerbig = scalerbig/1200.
dist = dist/1200.

"""
# Satistical Error on Big Scaler
rbuf = 0
for k in scalerbig:
    rbuf += k
rmean = rbuf/9
ubuf,n = 0,0
for k in scalerbig:
    ubuf += (k-rmean)**2
    n += 1
ru = np.sqrt(ubuf/(n*(n-1)))
rurel = ru/rmean
"""

"""
# Procentual Variation of rate raw data
dbuf,n = 0,0
for k in dist:
    dbuf += k
    n += 1
dmean = dbuf/n
#for k in dist:
    #print(k/dmean)
"""


# Set Starting Parameters for Gaussian and Model Fit
paramg = [0.1,50,10]        # Parameters for Gaussian Fit
paramm = [1,0.1,0.01]       # Parameters for Model Fit


"""
# Plot Raw Histogram for angle k
k = 8
plt.bar(channelaxis[1:],val[k]/1200.,width=0.15)
plt.title(r'$\theta =$ %i$^{\circ}$' %ang[k])
plt.xlim(0,8200)
plt.ylim(0,0.011)
plt.xlabel('ADC Channels')
plt.ylabel(r'Events per second')
#plt.savefig('180degraw.png')
plt.show()
"""


# Merge Bins, Subtract Background for all Angles
binsize = 20
x0,y0 = timeaxis,yraw
x,y,bg,bgtot = [],[],[],[]

for k in range(9):
    x1,y1 = binmerge(x0,y0[k],binsize)
    xbuf,ybuf,bgbuf = subtr_backg(x1,y1,3)
    x.append(xbuf)
    y.append(ybuf)
    bg.append(bgbuf)                # Background that was subtracted from each bin
    bgtot.append(bgbuf*len(y1))     # Total Background Events


# Make list with number of events for all angles
nevents = []        # List of Events per second for all angles
for i in y:
    evbuf = 0
    for j in i:
        evbuf += j
    nevents.append(evbuf)



"""
# Plot Gaussian Fit for all angles
fig, ax = plt.subplots(3,3,sharex=True,sharey=False)
k = 0
for i in range(3):
    for j in range(3):
        ax[i,j].set_title(r'$\theta =$ %i$^{\circ}$' %ang[k])
        ax[i,j].bar(x[k],y[k],label='data',width=0.15)
        popt,pcov = curve_fit(gaussian,x[k],y[k],p0=paramg)
        amp,mu,sigma = popt
        ax[i,j].plot(x[k],gaussian(x[k],*popt),'r')
        #ax[i,j].plot([mu]*2,[0,np.amax(y[k])],'g--')
        #for n in range(1,4,1):
            #ax[i,j].plot([mu-n*sigma]*2,[0,np.amax(y[k])],'y--')
            #ax[i,j].plot([mu+n*sigma]*2,[0,np.amax(y[k])],'y--')
        ax[i,j].set_ylim(0,0.09)
        #ax[i,j].set_xlim(mu-4*sigma,mu+4*sigma)
        ax[i,j].set_xlabel('Time Delay in ns')
        ax[i,j].set_ylabel(r'Events in s$^{-1}$')
        k += 1
plt.show()
"""


"""
# Plot Gaussian fit for angle k
k = 4             # Which angle
plt.title(r'$\theta\,=$ %i$^{\circ}$' %ang[k])
plt.bar(x[k],y[k],label='data',width=0.4,color='k')
popt,pcov = curve_fit(gaussian,x[k],y[k],p0=paramg)
amp,mu,sigma = popt
plt.plot(x[k],gaussian(x[k],*popt),'r')
#plt.plot([mu]*2,[0,np.amax(y[k])+0.02],'g')
#for n in range(1,5,1):
    #plt.plot([mu-n*sigma]*2,[0,np.amax(y[k])+0.02],'y')
    #plt.plot([mu+n*sigma]*2,[0,np.amax(y[k])+0.02],'y')
plt.xlabel('Time delay [ns]')
plt.ylabel(r'Events per time [Hz]')
plt.ylim(0,0.09)
#plt.xlim(mu-4*sigma,mu+4*sigma)
#plt.savefig('180deg.png',dpi=300)
plt.show()
"""


#Alternative mit Errors:
# Plot Gaussian fit for all angles
for k in range(9):
    plt.title(r'$\theta\,=$ %i$^{\circ}$' %ang[k])
    errors = list(map(lambda x : np.sqrt(np.abs(1200*x))/1200, y[k]))
    plt.bar(x[k],y[k],label='Data',width=0.15,color='k',yerr=errors)
    popt,pcov = curve_fit(gaussian,x[k],y[k],p0=paramg,sigma=errors)
    amp,mu,sigma = popt
    plt.plot(x[k],gaussian(x[k],*popt),'r',label='Gaussian fit')
    #plt.plot([mu]*2,[0,np.amax(y[k])+0.02],'g')
    #for n in range(1,5,1):
        #plt.plot([mu-n*sigma]*2,[0,np.amax(y[k])+0.02],'y')
        #plt.plot([mu+n*sigma]*2,[0,np.amax(y[k])+0.02],'y')
        plt.xlabel('Time delay [ns]')
        plt.ylabel(r'Events per time [Hz]')
        #plt.xlim(mu-4*sigma,mu+4*sigma)
        plt.ylim(0,0.1)
        plt.legend()
        #plt.savefig(str(ang[k])+'deg.png',dpi=300)
        #plt.show()




"""
# Delete value for theta=165deg for in ang and nevents
ang_cor, nevents_cor = [],[]
for i in range(7):
    ang_cor.append(ang[i])
    nevents_cor.append(nevents[i])
ang_cor.append(ang[8])
nevents_cor.append(nevents[8])
"""

'''
# Print Number of Events in Hertz, Background
#print(nevents)
#print(scaler)
#print(bg)
#print(bgtot)
#print((x[8][408]-x[8][0])/409)


# Normalise events per second (set ev/sec for 90deg to 1)
ratenorm = nevents/nevents[2]
bgnorm = bgtot/nevents[2]
scalernorm = scaler/nevents[2]
#print(bgnorm)



# Fit Data Points to Model Curve
popt,pcov = curve_fit(model,ang,ratenorm,p0=paramm)
a0,a1,a2 = popt
perr = np.sqrt(np.diag(pcov))
print('a0 = %.3f +/- %.3f' %(popt[0],perr[0]))
print('a1 = %.3f +/- %.3f' %(popt[1],perr[1]))
print('a2 = %.3f +/- %.3f' %(popt[2],perr[2]))



# Plot Data with Fit and Model
curve = model(np.arange(55,186,1),*popt)
curvemin = model(np.arange(55,186,1),*(popt-perr))
curvemax = model(np.arange(55,186,1),*(popt+perr))
theo = model(np.arange(55,186,1),1.,1/8.,1/24.)
plt.plot(ang,ratenorm,'go',label='Measurement')
#plt.plot(ang,scalernorm,'ko',label='Scaler')
plt.plot(np.arange(55,186,1),curve,'b',label='Fit')
#plt.plot(np.arange(55,186,1),curvemin,'c--',label='Min')
#plt.plot(np.arange(55,186,1),curvemax,'c--',label='Max')
plt.plot(np.arange(55,186),theo,'r',label='Prediction')
plt.xlim(55,185)
plt.title('Normalized Angular Distribution')
plt.xlabel(r'$\theta$ in $^{\circ}$')
plt.ylabel(r'$W(\theta)$')
legend = plt.legend(loc='upper left')
plt.show()
#plt.savefig('dist.png',dpi=300)
'''


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
for k in range(9):
    plt.bar(channelaxis[1:],val[k],width=0.15)
    plt.title(r'$\theta =$ %i$^{\circ}$' %ang[k])
    plt.xlim(0,8200)
    plt.ylim(0,13)
    plt.xlabel('Channel')
    plt.ylabel(r'Number of events')
    plt.savefig(str(ang[k])+'degraw.png',dpi=300)
    #plt.show()
    plt.clf()
    plt.cla()
    plt.close()
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



# Make Gaussian fit for all angles, empty all bins outside 3sigma, plot histograms 
nevents = [0,0,0,0,0,0,0,0,0]
uncerts = [0,0,0,0,0,0,0,0,0]
for k in range(9):
    #plt.title(r'$\theta\,=$ %i$^{\circ}$' %ang[k])
    errors = list(map(lambda x : np.sqrt(np.abs(1200*x))/1200, y[k]))
    #plt.bar(x[k],y[k],label='Data',width=0.15,color='k',yerr=errors)
    popt,pcov = curve_fit(gaussian,x[k],y[k],p0=paramg,sigma=errors)
    amp,mu,sigma = popt
    i0,i1,i2 = index(x[k],mu),index(x[k],mu-3*np.abs(sigma)),index(x[k],mu+3*np.abs(sigma))
    for j in y[k][i1:i2]:
        nevents[k] += j
    uncerts[k] = np.sqrt(np.abs(nevents[k]))/np.sqrt(1200)
    #plt.plot(x[k],gaussian(x[k],*popt),'r',label='Gaussian fit')
    #plt.plot([mu]*2,[0,np.amax(y[k])+0.02],'g')
    #for n in range(1,5,1):
        #plt.plot([mu-n*sigma]*2,[0,np.amax(y[k])+0.02],'y')
        #plt.plot([mu+n*sigma]*2,[0,np.amax(y[k])+0.02],'y')
    #plt.xlabel('Time delay [ns]')
    #plt.ylabel(r'Event rate [Hz]')
    #plt.xlim(mu-4*sigma,mu+4*sigma)
    #plt.ylim(0,0.1)
    #plt.legend()
    #plt.savefig(str(ang[k])+'deg.png',dpi=300)
    #plt.show()
    #plt.clf()
    #plt.cla()
    #plt.close()



# Print Number of Events in Hertz, Background
#print(nevents)
#print(scaler)
#print(uncerts)
#print(bgtot)



# Normalise events per second (set ev/sec for 90deg to 1)
ratenorm = nevents/nevents[2]
uncertnorm = uncerts/nevents[2]
bgnorm = bgtot/nevents[2]
scalernorm = scaler/nevents[2]
scalernorm2 = scaler/scaler[2]
scaleruncert = []
for k in scaler:
    scaleruncert.append(np.sqrt(k))
scaleruncert = scaleruncert/np.sqrt(1200.)
scaleruncertnorm = scaleruncert/scaler[2]
#print(bgnorm)



# Fit Data Points to Model Curve
popt,pcov = curve_fit(model,ang,ratenorm,sigma=uncertnorm,p0=paramm)
a0,a1,a2 = popt
perr = np.sqrt(np.diag(pcov))
print('a0 = %.3f +/- %.3f' %(popt[0],perr[0]))
print('a1 = %.3f +/- %.3f' %(popt[1],perr[1]))
print('a2 = %.3f +/- %.3f' %(popt[2],perr[2]))



# Plot Data with Fit and Model
curve = model(np.arange(55,186,1),a0,a1,-a2)
curvemin = model(np.arange(55,186,1),*(popt-perr))
curvemax = model(np.arange(55,186,1),*(popt+perr))
theo = model(np.arange(55,186,1),1.,1/8.,1/24.)
plt.errorbar(ang,ratenorm,yerr=uncertnorm,fmt='go',label='Data')
#plt.errorbar(ang,scalernorm2,yerr=scaleruncertnorm,fmt='ko',label='Scaler')
plt.plot(np.arange(55,186,1),curve,'b',label='Fit')
#plt.plot(np.arange(55,186,1),curvemin,'c--',label='Min')
#plt.plot(np.arange(55,186,1),curvemax,'c--',label='Max')
plt.plot(np.arange(55,186),theo,'r',label='Prediction')
plt.xlim(55,185)
plt.title('Normalized Angular Correlation Function')
plt.xlabel(r'$\theta$ [$^{\circ}$]')
plt.ylabel(r'$W(\theta)$')
legend = plt.legend(loc='upper left')
plt.show()
#plt.savefig('a2flipped.png',dpi=300)




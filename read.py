#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# 60 Degrees
d60 = np.loadtxt('data/60degrees.txt')  #Data saved
r60 = 0                             #number of rows in d60
n60 = 0                             #total number of events in d60
for i in d60:
    r60 += 1
    n60 += i

# 75 Degrees
d75 = np.loadtxt('data/75degrees.txt')  #Data saved
r75 = 0                             #number of rows in d75
n75 = 0                             #total number of events in d75
for i in d75:
    r75 += 1
    n75 += i

# 90 Degrees
d90 = np.loadtxt('data/90degrees.txt')  #Data saved
r90 = 0                             #number of rows in d90
n90 = 0                             #total number of events in d90
for i in d90:
    r90 += 1
    n90 += i

# 105 Degrees
d105 = np.loadtxt('data/105degrees.txt')  #Data saved
r105 = 0                             #number of rows in d105
n105 = 0                             #total number of events in d105
for i in d105:
    r105 += 1
    n105 += i

# 120 Degrees
d120 = np.loadtxt('data/120degrees.txt')  #Data saved
r120 = 0                             #number of rows in d120
n120 = 0                             #total number of events in d120
for i in d120:
    r120 += 1
    n120 += i

# 135 Degrees
d135 = np.loadtxt('data/135degrees.txt')  #Data saved
r135 = 0                             #number of rows in d135
n135 = 0                             #total number of events in d135
for i in d135:
    r135 += 1
    n135 += i

# 150 Degrees
d150 = np.loadtxt('data/150degrees.txt')  #Data saved
r150 = 0                             #number of rows in d150
n150 = 0                             #total number of events in d150
for i in d150:
    r150 += 1
    n150 += i

# 165 Degrees
d165 = np.loadtxt('data/165degrees.txt')  #Data saved
r165 = 0                             #number of rows in d165
n165 = 0                             #total number of events in d165
for i in d165:
    r165 += 1
    n165 += i

# 180 Degrees
d180 = np.loadtxt('data/180degrees.txt')  #Data saved
r180 = 0                             #number of rows in d180
n180 = 0                             #total number of events in d180
for i in d180:
    r180 += 1
    n180 += i

# Measured Values, dist=[n60,n75,...] number of events for each angle
# num=[r60,r75,...] number of bins for each angle
# val=[d60,d75,...] where d60 is data for each angle
dist = []
dist.append(n60)
dist.append(n75)
dist.append(n90)
dist.append(n105)
dist.append(n120)
dist.append(n135)
dist.append(n150)
dist.append(n165)
dist.append(n180)

num = []
num.append(r60)
num.append(r75)
num.append(r90)
num.append(r105)
num.append(r120)
num.append(r135)
num.append(r150)
num.append(r165)
num.append(r180)

val = []
val.append(d60)
val.append(d75)
val.append(d90)
val.append(d105)
val.append(d120)
val.append(d135)
val.append(d150)
val.append(d165)
val.append(d180)

ang=np.array([60,75,90,105,120,135,150,165,180])   # list of used angles
scaler = np.array([5203.,4912.,4947.,4986.,5034.,5152.,5219.,5223.,5167.])
scalerbig = np.array([1763689,1763159,1763096,1761493,1756916,1756281,1761526,1759589,1743748])
dist = np.array(dist)


#dist = dist/n90            # norm s.t. number of events at angle 90 = 1

import matplotlib.pyplot as plt
import pandas as pds
import numpy as np
from mpl_toolkits import mplot3d
#d=pds.read_csv("rainfall in india 1901-2015.csv")
data=pds.read_csv("25april 38400 delay afternoon features-edited.csv")

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.set_xlabel('mean')
ax.set_ylabel('stddev')
ax.set_zlabel('rms')

s=data['O']==0
data0=data[s]
print(data0.head())
#plt.scatter(data0['G'],data0['H'],color='b')
ax.scatter3D(data0['F'],data0['G'],data0['H'],color='b');

s=data['O']==30
data30=data[s]
print(data30.head())
#plt.scatter(data30['G'],data30['H'],color='r')
ax.scatter3D(data30['F'],data30['G'],data30['H'],color='r');

s=data['O']==60
data60=data[s]
print(data60.head())
#plt.scatter(data60['G'],data60['H'],color='g')
ax.scatter3D(data60['F'],data60['G'],data60['H'],color='g');

s=data['O']==90
data90=data[s]
print(data90.head())
#plt.scatter(data90['G'],data90['H'],color='y')
ax.scatter3D(data90['F'],data90['G'],data90['H'],color='y');

s=data['O']==120
data120=data[s]
print(data120.head())
#plt.scatter(data120['G'],data120['H'],color='m')
ax.scatter3D(data120['F'],data120['G'],data120['H'],color='m');

s=data['O']==150
data150=data[s]
print(data150.head())
#plt.scatter(data150['G'],data150['H'],color='c')
ax.scatter3D(data150['F'],data150['G'],data150['H'],color='c');
plt.show()

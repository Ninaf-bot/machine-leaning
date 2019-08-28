import pandas as pd
import quandl,math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing,  svm
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pywt
d_s=pd.read_csv("25april 38400 delay afternoon features-edited.csv")

t=d_s['O']==0
data0=d_s[t]
plt.plot(data0['A'],color='b')

t=d_s['O']==30
data30=d_s[t]
plt.plot(data30['A'],color='r')

t=d_s['O']==60
data60=d_s[t]
plt.plot(data60['A'],color='g')


t=d_s['O']==90
data90=d_s[t]
plt.plot(data90['A'],color='y')


t=d_s['O']==120
data120=d_s[t]
plt.plot(data120['A'],color='m')


t=d_s['O']==150
data150=d_s[t]
plt.plot(data150['A'],color='c')
plt.show()

wp = pywt.WaveletPacket(data=d_s['F'], wavelet='db1', mode='symmetric',maxlevel=6)
plt.plot(wp['aaaaaa'].data)
plt.show()

x=np.array(d_s.drop(['O'],1))
y=np.array(d_s['O'])
x=preprocessing.scale(wp)
y=np.array(d_s['O'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
clt=svm.SVC()
clt.fit(x_train,y_train)
clt.score(x_test,y_test)
accuracy=clt.score(x_test,y_test)
print(accuracy)

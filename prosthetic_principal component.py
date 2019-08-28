import pandas as pd
import quandl,math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing,  svm
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
d_s=pd.read_csv('25april 38400 delay afternoon features-edited.csv')
d_si=d_s[['F','G','H','O']]
plt.plot(d_si[['F']])
plt.ylabel('mean')
plt.show()
plt.plot(d_si[['G']])
plt.ylabel('stddev')
plt.show()
plt.plot(d_si[['H']])
plt.ylabel('rms')
plt.show()

d_s_stand=StandardScaler().fit_transform(d_si)
pca=PCA(n_components=2)
principalcomponents=pca.fit_transform(d_s_stand)

t=d_s['O']==0
data0=principalcomponents[t]
plt.scatter(data0[:,0],data0[:,1],color='b')


t=d_s['O']==30
data30=principalcomponents[t]
plt.scatter(data30[:,0],data30[:,1],color='r')


t=d_s['O']==60
data60=principalcomponents[t]
plt.scatter(data60[:,0],data60[:,1],color='g')


t=d_s['O']==90
data90=principalcomponents[t]
plt.scatter(data90[:,0],data90[:,1],color='y')


t=d_s['O']==120
data120=principalcomponents[t]
plt.scatter(data120[:,0],data120[:,1],color='m')


t=d_s['O']==150
data150=principalcomponents[t]
plt.scatter(data150[:,0],data150[:,1],color='c')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
           
plt.show()
plt.plot(principalcomponents[:,0])
plt.show()
plt.plot(principalcomponents[:,1])
plt.show()

x=np.array(principalcomponents)
y=np.array(d_s['O'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
clt=svm.SVC()
clt.fit(x_train,y_train)
clt.score(x_test,y_test)
accuracy=clt.score(x_test,y_test)
print(accuracy)
    


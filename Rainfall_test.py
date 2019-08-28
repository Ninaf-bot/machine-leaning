import pandas as pd
import quandl,math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing,  svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
d=pd.read_csv("E:\datasets\weatherAUS.csv")
d=d.set_index('Date')
d_s=d.copy()[d['Location']=='MATATHWADA']
print(d_s.head())
forecast_out=int(math.ceil(0.01*len(d_s)))
d_s['label']=d_s['JUN'].shift(-forecast_out)
d_s=d_s[['JUN','Jun-Sep','label']]
d_s.dropna(inplace=True)
print(d_s.head())
print(d_s.tail())
print(d_s.corr())
print(d_s.cov())
#plt.plot(d_s['JUN'])
plt.show()

x=np.array(d_s.drop(['label'],1))
y=np.array(d_s['label'])
x=preprocessing.scale(x)
y=np.array(d_s['label'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
clt=LinearRegression()
clt.fit(x_train,y_train)
clt.score(x_test,y_test)
accuracy=clt.score(x_test,y_test)
print(accuracy)


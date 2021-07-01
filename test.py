from keras.layers import Input, Dense, Dropout, BatchNormalization, InputLayer
from keras.models import Model
from keras.optimizers import Adam,SGD,RMSprop
from keras.datasets import boston_housing
import tensorflow as tf
from keras import metrics
from keras.metrics import AUC

from sklearn.linear_model import RANSACRegressor
from pandas import DataFrame

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
optimizer = Adam(lr=0.0001,decay=0.00001)
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,classification_report,confusion_matrix
y_true = np.array([0, 0, 0, 0, 1, 1,2,2])
y_pred = np.array([0, 1, 0, 1, 0, 1,2,1])
target_names = ['class 0', 'class 1','class_2']
recall_score(y_true, y_pred, average='micro')
c = classification_report(y_true, y_pred, target_names=target_names)
cnf_matrix = confusion_matrix(y_true, y_pred)

accuracy_score(y_true, y_pred)
precision_score(y_true, y_pred, average='binary')

import pandas as pd
import numpy as np

hpgStock = pd.read_csv(r'E:\StocksData\HPG.csv',names=['date','Open','Close','Low','High','Volume','g'])
hpgStock = hpgStock.drop('g',axis=1)
hpgArray = np.asarray(hpgStock)


vnIndex = pd.read_csv(r'E:\StocksData\/^VNINDEX.csv',names=['date','Open','Close','Low','High','Volume','g'])
#vnIndex = vnIndex.iloc[-hpgStock.shape[0]:,0:2]
vnArray = np.asarray(vnIndex)[:,0:2]

a = np.ndarray.astype(vnArray[-10:,0],dtype='int')
a = np.array([[1,2], [3, 4], [5, 6]])
b = np.array([[22,3], [3, 3], [5, 3], [5, 3]])

bool_idx = (a == b) # Tìm các phần tử lớn hơn 2; = np.array([[1, 2, 3], [4, 5, 6]])

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v


a = np.array([(1,2,3,4,5,4,3,2)],dtype='uint8')
print(a.itemsize)
a=256
a=np.linspace(0,9,19)

x= np.array([(1,2,3),(1,1,1),(4,3,1)])
y= np.array([(1,2,3),(3,4,5),(1,5,1)])
print(np.vstack((x,y)))

print(np.hstack((x,y)))

np.concatenate((x,y),axis=1)

print(x.ravel())

x= np.arange(0,3*np.pi,0.1)
x = np.arange(-10,10,0.1)
y = np.tanh(x)
plt.plot(x,y)



import math
import numpy as np
import matplotlib.pyplot as plt

rg = np.random.default_rng(1)
mu, sigma = 2, 0.8
v = rg.normal(mu,sigma,1000)
# Plot a normalized histogram with 50 bins
plt.hist(v, bins=50, density=1)       # matplotlib version (plot)
# Compute the histogram with numpy and then plot it
(n, bins) = np.histogram(v, bins=50, density=True)  # NumPy version (no plot)
plt.plot(.5*(bins[1:]+bins[:-1]), n)

X = np.random.random((1,100))
Y = 3*(X**4)+5 + 0.5*np.random.random((1,100))
K = np.vstack((X,Y))
np.cov(X,Y)
plt.plot(X.T,Y.T,'ro')

K = np.array([[50772, 73756, 74251, 77601],[102492, 100406, 97762, 98191]])
M = np.sum(K,axis=1)/K.shape[1]
M = np.array([M,M,M,M]).T
D = K-M
C = np.dot(D,D.T)/3
np.cov(K)

from sklearn.preprocessing import Normalizer
X = [[5, 1, 2, 2],
     [1, 3, 9, 3],
     [5, 7, 5, 1]]
transformer = Normalizer(norm='max')#.fit(X)  # fit does nothing.

transformer.fit_transform(X)


from sklearn import preprocessing
import numpy as np
X_train = np.array([[ 1., 24.,  41.],
                    [ 12.,  13.,  26.],
                    [ 449.,  35., -41.]])
X_scaled = preprocessing.scale(X_train)



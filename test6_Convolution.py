import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout

from tensorflow.keras import Sequential

tf.enable_eager_execution()#Chạy trực tiếp tensorflow, sẽ ko báo lỗi


def minmaxScale(Data):
    _min = min(Data)
    _max = max(Data)
    result = (Data - _min)/(_max - _min)
    return result


def draw(a,b,c):  
    fig, ax1 = plt.subplots()
    ax1.plot(a,color='red')
    ax1.set_ylabel('Error')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(b,color='green')
    ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax3.plot(c,color='black')
    fig.legend(['Errors', 'Data','Kener'], loc='upper center')
    

def calculate_Error(DataA=[], DataB=[]):   
    _dataA = minmaxScale(np.asarray(DataA))
    _dataB = minmaxScale(np.asarray(DataB))
    #_dataB = _dataB[::-1]
    _kenerWitdh = len(_dataB)
    _dataWitdh = len(_dataA)
    Errors = []
    for i in range(_dataWitdh - _kenerWitdh):
        Errors.append(mean_squared_error(minmaxScale(_dataA[i:i+_kenerWitdh]),_dataB[:_kenerWitdh]))
    
    return np.argmin(Errors), np.asarray(Errors) ,_dataA, _dataB #np.asarray(Errors).reshape(-1,1),_dataA,_dataB


def makeData(DataA,DataB,TimeStep,KenerWitdh):
    _dataA = minmaxScale(np.asarray(DataA)).reshape(-1,1)
    _dataB = minmaxScale(np.asarray(DataB)).reshape(-1,1)
    _timeStep = TimeStep
    _kenerWitdh = KenerWitdh
    Data = []
    for i in range(0,len(_dataA)-_timeStep,1):
        Data.append((_dataA[i:i+_timeStep],_dataB[i:i+_kenerWitdh]))
    return Data
   

SP500 = pd.read_csv(r'E:\StocksData\SP500.csv')
SP500 = SP500.drop('g',axis=1)
SP500Close = SP500['Close'].values.reshape(-1,1)
SP500Close = SP500Close[:,:].reshape(-1,1)[::-1,:]

VANG = pd.read_csv(r'E:\StocksData\VANG.csv')
VANGClose = VANG['Close'].values.reshape(-1,1)
VANGClose = VANGClose[:,:].reshape(-1,1)[::-1,:]

DAUTHO = pd.read_csv(r'E:\StocksData\DAUTHO.csv')
DAUTHOClose = DAUTHO['Close'].values.reshape(-1,1)
DAUTHOClose = DAUTHOClose[:,:].reshape(-1,1)[::-1,:]


   
kener = np.ones((20,))/20
hpgStock = pd.read_csv(r'E:\StocksData\HPG.csv',names=['date','Open','High','Low','Close','Volume','g'])
hpgStock = hpgStock.drop('g',axis=1)
hpgClose = hpgStock['Close'].values.reshape(-1,1)[:,:]
#hpgVolume = hpgStock['Volume'].values.reshape(-1,1)[:,:]

vnIndex = pd.read_csv(r'E:\StocksData\FPT.csv',names=['date','Open','High','Low','Close','Volume','g'])
vnIndex = vnIndex.drop('g',axis=1)
vnIndexClose = vnIndex['Close'].values.reshape(-1,1)
vnIndexClose = vnIndexClose[-3539:,:].reshape(-1,1)
l = int(min(len(hpgClose),len(vnIndexClose))*(-1))
hpgClose = np.convolve(hpgClose[l:,0].T,kener,'valid').reshape(-1,1)
vnIndexClose = np.convolve(vnIndexClose[l:,0].T,kener,'valid').reshape(-1,1)

witdh = 30
step = witdh-5
start = 0
Data = makeData(hpgClose,vnIndexClose,100,witdh)
Result = []
for i in range(start,len(Data)-0,step):
    #print(i)
    r,a,b,c = calculate_Error(Data[i][0],Data[i][1])
    Result.append(r)
    #draw(a,b,c)
    
Result = np.asarray(Result)
plt.hist(Result,200,color='blue')


plt.plot(minmaxScale(hpgClose[:]),color='red')
plt.plot(minmaxScale(vnIndexClose[:]))
plt.legend(['HPG','FPT'])

plt.plot(minmaxScale(SP500Close[-1000:]),color='red')
plt.plot(minmaxScale(vnIndexClose[-1000:]))






kener = np.array([1,1])/2
StockA = pd.read_csv(r'E:\StocksData\MWG.csv',names=['date','Open','High','Low','Close','Volume','g'])
StockA = StockA.drop('g',axis=1)
StockAClose = StockA['Close'].values.reshape(-1,1)[:,:]
StockB = pd.read_csv(r'E:\StocksData\GMD.csv',names=['date','Open','High','Low','Close','Volume','g'])
StockB = StockB.drop('g',axis=1)
StockBClose = StockB['Close'].values.reshape(-1,1)

l = int(min(len(StockBClose),len(StockAClose))*(-1))

StockAClose = np.convolve(StockAClose[l:,0].T,kener,'valid').reshape(-1,1)
StockBClose = np.convolve(StockBClose[l:,0].T,kener,'valid').reshape(-1,1)

plt.plot(minmaxScale(StockAClose[:]),minmaxScale(StockBClose[:]),'ro',markersize=1)












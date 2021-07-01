import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout

from tensorflow.keras import Sequential

tf.enable_eager_execution()#Chạy trực tiếp tensorflow, sẽ ko báo lỗi

kener = np.array([1,1,1,1,1])/5
hpgStock = pd.read_csv(r'E:\StocksData\HPG.csv',names=['date','Open','High','Low','Close','Volume','g'])
hpgStock = hpgStock.drop('g',axis=1)
hpgClose = hpgStock['Close'].values.reshape(-1,1)[400:,:]
hpgVolume = hpgStock['Volume'].values.reshape(-1,1)[400:,:]
#hpgClose = np.convolve(hpgClose[:,0].T,kener,'same').reshape(-1,1)



vnIndex = pd.read_csv(r'E:\StocksData\^VNINDEX.csv',names=['date','Open','High','Low','Close','Volume','g'])
vnIndex = vnIndex.drop('g',axis=1)
vnIndexClose = vnIndex['Close'].values.reshape(-1,1)
vnIndexClose = vnIndexClose[-2787:,:].reshape(-1,1)
#vnIndexClose = np.convolve(vnIndexClose[:,0].T,kener,'same').reshape(-1,1)

X_Data = vnIndexClose# np.concatenate((hpgClose,vnIndexClose),axis=1)
timestep = 20
num_feature = 2
num_predict = 5
 
def makeTimeSerialData(data=[],timestep=20,num_feature=2,num_predict=10):
    X_train=[]    
    batchsize = data.shape[0]
    for i in range(batchsize-timestep-num_predict):
        X_train.append(data[i:i+timestep,:])    
    return X_train

X_DataSerial = makeTimeSerialData(X_Data,timestep=timestep,num_feature=num_feature,num_predict=num_predict)
Y_DataSerial = makeTimeSerialData(X_Data[20:,0:1],timestep=5,num_feature=1,num_predict=0) #makeTimeSerialData(data=X_Data[5:,0].reshape(-1,1),timestep=1,num_feature=1,num_predict=0)        

X_Train = np.asarray(X_DataSerial[:2000])
Y_Train = np.asarray(Y_DataSerial[:2000]).reshape(-1,5)
#Y_Train = Y_Train/X_Train[:,0,0].reshape(-1,1)
#Y_Train = np.where(Y_Train>1.00,1,0)

X_Test = np.asarray(X_DataSerial[2000:])
Y_Test = np.asarray(Y_DataSerial[2000:]).reshape(-1,5)
#Y_Test = Y_Test/X_Test[:,0,0].reshape(-1,1)
#Y_Test = np.where(Y_Test>1.00,1,0)

n_features = 2
n_input = 20

es = EarlyStopping(monitor='val_loss', mode='min', verbose=10,patience=10)
model = Sequential()
model.add(LSTM(units=32,activation='relu', input_shape=(n_input, n_features), return_sequences=False))
model.add(layers.BatchNormalization())
# =============================================================================
# model.add(Dropout(0.2))
# model.add(LSTM(units=32,activation='relu', return_sequences=False))
# model.add(Dropout(0.2))
# =============================================================================
#model.add(layers.BatchNormalization())
#model.add(LSTM(units=8,activation='relu', return_sequences=True))
#model.add(layers.BatchNormalization())
# =============================================================================
#model.add(LSTM(units=64,activation='relu', return_sequences=False))
#model.add(LSTM(units=20,activation='relu', return_sequences=True))
#model.add(LSTM(units=20,activation='relu', return_sequences=True))
#model.add(LSTM(units=20,activation='relu', return_sequences=False))
# =============================================================================
#model.add(Dense(32,activation='relu'))
#model.add(Dense(16,activation='relu'))
#model.add(Dense(32,activation='relu'))
#model.add(Dense(16,activation='relu'))
#model.add(layers.BatchNormalization()) 
model.add(Dense(5))
model.compile(optimizer='adam', loss='mse')#,metrics=['mse']

model.summary()


history = model.fit(X_Train,Y_Train, validation_data =(X_Test[:400,:],Y_Test[:400,:]), epochs=60, verbose=1, batch_size=64,callbacks=[es])

#model = keras.models.load_model("stock_ver1")


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'], loc='upper left')

model.evaluate(X_Test[200:201,:],Y_Test[200:201,:],verbose=2)
model.evaluate(X_Test[201:202,:],Y_Test[201:202,:],verbose=2)
model.evaluate(X_Test[0:202,:],Y_Test[0:202,:],verbose=2)
model.evaluate(X_Train[1000:1500],Y_Train[1000:1500],verbose=2)
ypre = model.predict(X_Train[1000:1500])
y_true = Y_Train[1000:1500]

model.evaluate(X_Test[400:,:],Y_Test[400:,:],verbose=2)

ypre = model.predict(X_Train[1550:2000])
y_true = Y_Train[1550:2000]

ypre = model.predict(X_Test[400::5])
y_true = Y_Test[395::5]

plt.plot(y_true[:,0].T,color='red')
plt.plot(ypre[:,0].T-1)
plt.legend(['True', 'Pre'], loc='lower left')

plt.plot(y_true[:,:].reshape(-1,1),color='red')
plt.plot(ypre[:,:].reshape(-1,1))
plt.legend(['True', 'Pre'], loc='lower left')

plt.plot(X_Data[:,0],color='red')
plt.plot(X_Data[:,1],color='blue')
plt.legend(['True', 'Pre'], loc='lower left')

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('HPG', color=color)
ax1.plot(X_Data[400:,0], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('VNIndex', color=color)  # we already handled the x-label with ax1
ax2.plot(vnIndexClose[400:,:], color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped

#model.save('stock_ver1')
#model.save_weights('stock_ver1_weigh')
print(classification_report(y_true, ypre))
a=classification_report(y_true, ypre)
del model
# =============================================================================
# 
# input_shape = (1,colClose.shape[0], 1)
# colClose=np.reshape(colClose,input_shape)
# x = colClose[:,10:,:]
# 
# kernel_size=5
# conv1D  = layers.Conv1D(filters=1,kernel_size=kernel_size,use_bias=False,kernel_initializer=keras.initializers.Constant(1/kernel_size))(x)
# 
# y=conv1D.numpy()
# 
# np.argmax(colVolume)
# 
# =============================================================================
# =============================================================================
# input_shape = (1,10, 1)
# x = tf.random.normal(input_shape)
# y = tf.keras.layers.Conv1D(filters=1, kernel_size=3, activation='linear',kernel_initializer=keras.initializers.Constant(1/3), input_shape=(1,1))(x)
# 
# a = x.numpy()
# b = y.numpy()
# =============================================================================

startPoint = -2000
kenerWitdh = 15
timeStep = 30

kener = np.asarray(vnIndexClose[startPoint:startPoint + kenerWitdh,0])
hpg = np.asarray(hpgClose[startPoint:startPoint + timeStep,0])

hpgConv = np.convolve(hpg,kener,'valid')


fig, ax1 = plt.subplots()
ax1.plot(hpg,color='red')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(kener,color='green')
ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax3.plot(hpgConv,color='blue')

def draw(_startPoint):
    startPoint = _startPoint
    kenerWitdh = 15
    timeStep = 30
    
    kener = np.asarray(vnIndexClose[startPoint:startPoint + kenerWitdh,0])
    hpg = np.asarray(hpgClose[startPoint:startPoint + timeStep,0])
    
    hpgConv = np.convolve(hpg,kener,'valid')
    
    
    fig, ax1 = plt.subplots()
    ax1.plot(hpg,color='red')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(kener,color='green')
    ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax3.plot(hpgConv,color='blue')

for i in range(0,200,15):
    _startPoint = -500
    draw(_startPoint+i)








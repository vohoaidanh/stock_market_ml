from keras.layers import Input, Dense, Dropout, BatchNormalization, InputLayer
from keras.models import Model
from keras.optimizers import Adam,SGD,RMSprop
from keras.datasets import boston_housing
from keras.callbacks import EarlyStopping
from sklearn.linear_model import RANSACRegressor
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

mycallBack = [EarlyStopping(monitor='val_loss', patience=10,mode='min',restore_best_weights=True)]

optimizer = Adam(lr=0.0001,decay=0.00001)
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


x_train = x_train[:,(1,3,5,6,7,8,9,10,11,12)]
x_test = x_test[:,(1,3,5,6,7,8,9,10,11,12)]
#y_train = y_train/max(y_train)

#y_test = y_test/max(y_test)

#x_train1 = np.reshape(x_train,(404,13))

input_layer = Input(shape=(10,))
dense_1 = Dense(units = 32, activation='relu')(input_layer)
dense_1 = BatchNormalization()(dense_1)
dense_1 = Dense(units = 128, activation='relu')(dense_1)
dense_1 = BatchNormalization()(dense_1)
dense_1 = Dense(units = 256, activation='relu')(dense_1)
dense_1 = BatchNormalization()(dense_1)
dense_1 = Dense(units = 128, activation='relu')(dense_1)
#dense_1 = BatchNormalization()(dense_1)
dense_1 = Dense(units = 1, activation='linear')(dense_1)
model = Model(inputs = input_layer, outputs = dense_1)
model.summary()
model.compile(optimizer=optimizer, loss='mean_squared_error')
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

history = model.fit(x_train,y_train,epochs=500,validation_split = 0.25, verbose=1,callbacks=mycallBack)


print(history.history.keys())

plt.plot('loss',data=history.history)
plt.plot('val_loss',data=history.history)
plt.legend()

#score  = model.evaluate(x_test[:100],y_test[:100],verbose=2)
#model.evaluate(x_train,y_train)
# =============================================================================
# hist = DataFrame(history.history)
# hist['epoch'] = history.epoch
# hist.tail()
# 
# plt.plot(hist['val_mean_absolute_error'])
# 
# y_pre=model.predict(x_test[:100])
# =============================================================================

y_pre=model.predict(x_test[22:44])
y_pre = np.reshape(y_pre,(-1,))

mean_squared_error(y_test[22:44],y_pre)

np.sqrt(36)

x = np.array(np.arange(0.1,10,0.1))
y = np.log(x)

plt.plot(x,y,'ro')


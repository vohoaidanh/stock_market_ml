import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.enable_eager_execution()


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

loss = keras.losses
inputs = layers.Input(shape=(784,))

denses = layers.Dense(64,activation='relu')
x = denses(inputs)
x = layers.Dense(64,activation='relu')(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs = outputs, name='mnist_model')
model.summary()
model.compile(
    loss = keras.losses.CategoricalHinge(),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
    )


model.fit(x_train,y_train,batch_size=1200,epochs=10,validation_split=0.2)

test_score= model.evaluate(x_test,y_test,verbose=2)

pre = model.predict(x_test[325:326])

np.argmax(pre)

plt.imshow(x_test[325:326].reshape(28,28))

del model


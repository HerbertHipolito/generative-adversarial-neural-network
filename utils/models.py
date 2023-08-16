import keras
from keras.layers import Dense, LeakyReLU, Reshape, Conv2DTranspose

def generator(): 

  model = keras.Sequential()
  model.add(Dense(units=256,activation=LeakyReLU(0.2),input_shape=(794,)))
  model.add(Dense(units=512,activation=LeakyReLU(0.2)))
  model.add(Dense(units=1024,activation=LeakyReLU(0.2)))
  model.add(Dense(units=784,activation='tanh'))

  return model

def discriminator():

  model = keras.Sequential()
  model.add(Dense(units=1024,activation=LeakyReLU(0.2), input_shape=(794,)))
  model.add(Dense(units=512,activation=LeakyReLU(0.2)))
  model.add(Dense(units=256,activation=LeakyReLU(0.2)))
  model.add(Dense(units=1,activation='sigmoid'))
  
  return model

"""
def generator():
 
  model = keras.Sequential()
  model.add(Dense(units=1568, use_bias=False, input_shape=(794,)))
  model.add(Reshape((7, 7, 32)))
  model.add(Conv2DTranspose(64, 3, strides=2, padding='same', use_bias=False,activation=LeakyReLU(0.2)))
  model.add(Conv2DTranspose(1, 3, strides=2, padding='same', use_bias=False,activation='tanh'))
  model.add(Reshape((784,)))
  
  return model

"""
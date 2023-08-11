import keras
from keras.layers import Dense, LeakyReLU

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
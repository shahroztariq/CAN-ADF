import keras
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, TimeDistributed, LSTM, regularizers
from scipy.stats import mode
import pandas
import numpy as np

batch_size = 128 # 16 no good
num_classes = 4
epochs = 20
first_dim = 40
feature_idx = 11 # Number of features

def CANRNN():
  model=Sequential()
  model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.0001), input_shape=(first_dim, feature_idx,1)))
  model.add(BatchNormalization())
  model.add(Dropout(0.2))
  model.add(Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
  model.add(BatchNormalization())
  model.add(Dropout(0.2))
  model.add(Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
  model.add(BatchNormalization())
  model.add(Dropout(0.2))
  model.add(TimeDistributed(LSTM(512,kernel_regularizer=regularizers.l2(0.0001))))
  model.add(LayerNorm1D())
  model.add(LSTM(512,kernel_regularizer=regularizers.l2(0.0001)))
  model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l2(0.0001)))
  model.summary()
  model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
  return model

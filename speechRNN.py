# TODO:
## replace linear layer by GMM layer
## try scikit-learn for parameter optimization
## check activations, at least of the last layer

from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import EarlyStopping
from keras.utils.visualize_util import plot

import numpy as np
import random
import sys
import os
import math
import matplotlib.pyplot as plt
import struct
import graphviz
import speechRNN_utils

speech = []
timesteps = 100
timesteps_x = timesteps - 1
samples = load_ahocoder_data()
data = split_samples(samples)
data, mu, sigma = normalize_data(data)

# plt.imshow(data, aspect='auto', interpolation='nearest')
# plt.show()

data = np.dstack(data).swapaxes(1,2).swapaxes(0,1) # tensor of [nb_samples, timesteps, input_dim]
nb_samples, input_dim = data.shape[0], data.shape[2]
(X_train, y_train), (X_test, y_test) = train_test_split(data)

# build model
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# prcatical considerations from Goodfellow/Bengio:
## optimizer: SGD w/ momentum w/ decaying learning rate (linearly, exponentially, by a factor of 2-10), ADAM
## regularization: early stopping, dropout, batch normalization
## unsupervised pre-training (if it's known to be helpful)
## hyperparams: manually vs. automatically
### <hyperparam>: <increases capacity when>
### no of hidden units: increased
### learning rate: tuned optimally
### weight decay coefficient: decreased
### dropout rate: decreased
## monitor histograms of activations and gradient (see details in book)

print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=False, input_shape=(timesteps_x, input_dim)))
model.add(Dropout(0.2))
# model.add(LSTM(512, return_sequences=False))
# model.add(Dropout(0.2))
model.add(Dense(input_dim))
model.add(Activation('linear'))

model.compile(loss='rmse', optimizer='adam')
callbacks = [early_stopping]
hist = model.fit(X_train, y_train, batch_size=128, nb_epoch=20, validation_split=0.1, callbacks=callbacks)
plot_lc(hist)
plot(model, to_file="model.png")

pred_len = 5000
for i in range(min(X_test.shape[0], 1)):
  print('-' * 50)
  print('Iteration', i+1)
  # model.fit(X_train, y_train, batch_size=128, nb_epoch=1) # what's the benefit of fitting the model n times?

  generated = []
  signal = X_test[i,:,:].reshape(1,timesteps_x,input_dim)

  for n in range(pred_len):
    preds = model.predict(signal, verbose=0)[0]
    generated.append(preds)
    signal = np.concatenate((signal[:,1:,:], preds.reshape((1,1,input_dim))),1)

  pred = np.vstack(generated)
  pred = denormalize_data(pred, mu, sigma)
  save_ahocoder_pred(pred)


# plot last sample
X_test_norm = denormalize_data(X_test[i,:,:], mu, sigma)
plt.figure()
plt.imshow(X_test_norm.T, aspect='auto', interpolation='nearest')
plt.figure()
plt.imshow(np.vstack((X_test_norm, pred)).T, aspect='auto', interpolation='nearest')
plt.show()
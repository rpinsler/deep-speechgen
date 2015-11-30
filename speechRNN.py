# TODO:
## replace linear layer by GMM layer
## try scikit-learn for parameter optimization
## check activations, at least of the last layer

# open issues:
## how to incorporate vuv bit into cost function?
## why normalize outputs to lie between [0.01, 0.99]
## postfiltering in cepstral domain?
## interpolate first, then calculate derivatives, or vice versa?

# things to try out:
## input normalization
## GMM layer
## v/uv bit
## deeper architecture
## regularization, e.g. Dropout, BatchNormalization, L1/L2, structure regularization
## given timesteps 1:t, predict t+1:t+n
## given timesteps 1:t, predict 2:t+1
## different initialization
## EarlyStopping
## RNN-specific tricks

from __future__ import print_function
from __future__ import absolute_import

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Activation, Dropout, Lambda
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
import speechRNN_utils as sp

# pre-processing of data:
# - downsample data from 44.1KHz to 16KHz: python downsample.py
# - extract audio features: ./ahocoder.sh
# - extract derivatives of audio features: ./extractdelta.sh 

timesteps = 100
use_delta = False
data = sp.load_ahocoder_data(100, use_delta=use_delta)

interp_data = []
vuv_data = []
for sample in data:
  vuv = sp.get_vuv_flag(sample, use_delta=use_delta)
  sample = sp.interp_uv(sample, use_delta=use_delta)
  interp_data.append(sample)
  vuv_data.append(vuv)  

data, vuv = sp.split_samples(interp_data, vuv_data)
interp_data = vuv_data = None

# plt.imshow(data, aspect='auto', interpolation='nearest')
# plt.show()

data = np.dstack(data).transpose((2,0,1))
vuv = np.dstack(vuv).transpose((2,0,1))
data, mu, sigma = sp.normalize_data(data)
data = np.concatenate((data, vuv),-1)
vuv = None
(X_train, y_train), (X_test, y_test) = sp.train_test_split(data)
data = None
nb_samples, timesteps, input_dim = X_train.shape

# build model
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# practical considerations from Goodfellow/Bengio:
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

# linear output layer
model = Sequential()
model.add(LSTM(512, return_sequences=False, input_shape=(timesteps, input_dim)))
model.add(Dropout(0.2))
# model.add(LSTM(512, return_sequences=False))
# model.add(Dropout(0.2))
model.add(Dense(input_dim))
model.compile(loss='rmse', optimizer='adam')

# GMM output layer
M = 2 # no of GMM components
model_gmm = Sequential()
model_gmm.add(LSTM(512, return_sequences=False, input_shape=(timesteps, input_dim)))
model_gmm.add(Dropout(0.2))
model_gmm.add(Dense((M+2)*input_dim))
model_gmm.add(Lambda(partial(gmm_activation,M=3)))
model_gmm.compile(loss=gmm_loss(M), optimizer='adam')

callbacks = [early_stopping]
hist = model.fit(X_train, y_train, batch_size=128, nb_epoch=20, validation_split=0.1, callbacks=callbacks)
sp.plot_lc(hist)
plot(model, to_file="model.png")

pred_len = 2000
for i in range(1):
  print('-' * 50)
  print('Iteration', i+1)
  # model.fit(X_train, y_train, batch_size=128, nb_epoch=1) # what's the benefit of fitting the model n times?

  generated = []
  signal = X_test[i,:,:].reshape(1,timesteps,input_dim)

  for n in range(pred_len):
    preds = model.predict(signal, verbose=0)[0]
    generated.append(preds)
    signal = np.concatenate((signal[:,1:,:], preds.reshape((1,1,input_dim))),1)

  pred = np.vstack(generated)
  pred[:,:-1] = sp.denormalize_data(pred[:,:-1], mu, sigma)
  # pred = sp.scale_output(X_train, pred)
  if use_delta:
    (mcp, mcpd, mcpdd), (lf0, lf0d, lf0dd), (mfv, mfvd, mfvdd), vuv = sp.split_features(pred, use_vuv=True)
    pred = np.hstack((mcp, lf0, mfv, mcpd, lf0d, mfvd, mcpdd, lf0dd, mfvdd))
    variance = np.repeat(np.expand_dims(np.square(sigma),0), len(generated), 0)
    pred = sp.mlpg(pred, variance, i+1)
    pred = sp.replace_uv(pred, vuv)
  else:
    pred = sp.replace_uv(pred)

  sp.save_ahocoder_pred(pred, i+1)

# plot last sample
test_sample = X_test[i,:,:]
# test_sample = sp.denormalize_data(test_sample, mu, sigma)
# mmcd, rmse, class_err = sp.evaluate(y_test, pred) # requires longer test trajectories
plt.figure()
plt.imshow(test_sample.T, aspect='auto', interpolation='nearest')
plt.figure()
plt.imshow(np.vstack((test_sample, pred)).T, aspect='auto', interpolation='nearest')
plt.show()
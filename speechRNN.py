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
## regularization, e.g. Dropout, BatchNormalization, L1/L2, structured regularization
## given timesteps 1:t, predict t+1:t+n
## given timesteps 1:t, predict 2:t+1
## different initialization
## EarlyStopping
## gradient clipping
## RNN-specific tricks

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

from __future__ import print_function
from __future__ import absolute_import

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Activation, Dropout, Lambda
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
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

def load(n=0, cut=0, use_delta=False):
  # pre-processing of data:
  # - downsample data from 44.1KHz to 16KHz: python downsample.py
  # - extract audio features: sudo ./ahocoder.sh
  # - extract derivatives of audio features: sudo ./extractdelta.sh 

  data = sp.load_ahocoder_data(n, use_delta=use_delta)
  interp_data = []
  vuv_data = []
  for sample in data:
    sample = sample[cut:sample.shape[0]-cut,:]
    vuv = sp.get_vuv_flag(sample, use_delta=use_delta)
    sample = sp.interp_uv(sample, use_delta=use_delta)
    interp_data.append(sample)
    vuv_data.append(vuv)  

  data, vuv = sp.split_samples(interp_data, vuv_data)
  interp_data = vuv_data = None # free some space
  data = np.dstack(data).transpose((2,0,1))
  vuv = np.dstack(vuv).transpose((2,0,1))
  data = np.concatenate((data, vuv),-1)
  return data

def preprocess(data, mu=None, sigma=None):
  data, mu, sigma = sp.normalize_data(data, mu, sigma)
  return data, mu, sigma

def build(timesteps, input_dim):
  # linear output layer
  model = Sequential()
  model.add(LSTM(512, return_sequences=False, input_shape=(timesteps, input_dim)))
  # model.add(Dropout(0.2))
  # model.add(LSTM(512, return_sequences=False))
  # model.add(Dropout(0.2))
  model.add(Dense(input_dim))
  model.compile(loss='rmse', optimizer='adam')

  # GMM output layer
  # M = 2 # no of GMM components
  # model = Sequential()
  # model.add(LSTM(512, return_sequences=False, input_shape=(timesteps, input_dim)))
  # model.add(Dropout(0.2))
  # model.add(Dense((input_dim+2)*M))
  # model.add(sp.GMMActivation(M))
  # model.compile(loss=sp.gmm_loss, optimizer='adam')

  return model

def predict(X_test, model, pred_len=0, M=None):
  generated = []
  nb_samples, timesteps, input_dim = X_test.shape
  
  # sample start sequence from data
  idx = np.random.randint(nb_samples)
  signal = X_test[idx,:,:].reshape(1,timesteps,input_dim)

  for n in range(pred_len):
    if (n%500) == 0:
      print("Predictions completed: " + n + "/" + pred_len) 
      
    preds = model.predict(signal, verbose=0)[0]
    if M is not None:
      preds = sp.sample(preds, M)

    generated.append(preds)
    signal = np.concatenate((signal[:,1:,:], preds.reshape((1,1,input_dim))),1)

  pred = np.vstack(generated)
  return pred, X_test[idx,:,:]

def postprocess(data, approach, pred_len, use_delta = False, X_train=None, mu=None, sigma=None):
  # data[:,:-1] = sp.denormalize_data(data[:,:-1], mu, sigma)
  # data = sp.scale_output(X_train, data)
  if use_delta:
    (mcp, mcpd, mcpdd), (lf0, lf0d, lf0dd), (mfv, mfvd, mfvdd), vuv = sp.split_features(data, use_vuv=True)
    data = np.hstack((mcp, lf0, mfv, mcpd, lf0d, mfvd, mcpdd, lf0dd, mfvdd))
    variance = np.repeat(np.expand_dims(np.square(sigma),0), pred_len, 0)
    data = sp.mlpg(data, variance, approach)
    data = sp.replace_uv(data, vuv)
  else:
    data = sp.replace_uv(data)

  return data

def evaluate(pred, test_sample, y_test=None, show_plot=False, save_plot=None):
  if y_test is not None:
    mmcd, rmse, class_err = sp.evaluate(y_test, pred) # requires longer test trajectories

  plt.figure()
  plt.imshow(np.vstack((test_sample, pred)).T, aspect='auto', interpolation='nearest')
  if save_plot is not None:
    plt.savefig(save_plot, dpi=100)
  
  if show_plot:
    plt.show()

def debug():
  pass
  # w = sp.get_layer_weights(2, model)[0]
  # plt.imshow(w)
  # plt.hist(w.flatten())
  # plt.show()

  # a = sp.get_layer_outputs(2, model, np.expand_dims(X_test[0,:,:], 0))
  # plt.hist(a.flatten())
  # plt.show()

  # find data/pred/wav16/ -type f -name "*.wav" > split_ids.txt
  # python fbank_features/signal2logspec.py -p
  # python fbank_features/logspec_viewer.py

  # what to plot
  ## mixture weights
  ## vuv sequence
  ## keep track of variance of activations

approach = "baseline"
load_weights = True
nb_rawsamples = 0
timesteps = 100
batch_size = 128
use_delta = False
nb_epoch = 20
pred_len = 2000
cut = 0 # no of timesteps that are removed at beginning and end of each sample

print("Load data")
data = load(nb_rawsamples, cut, use_delta)
(X_train, y_train), (X_test, y_test) = sp.train_test_split(data)
nb_samples, timesteps, input_dim = X_train.shape
data = None # free some space

print("Preprocess data")
# X_train, mu, sigma = preprocess(X_train)
# y_train, X_test, y_test = preprocess(y_train, mu, sigma), preprocess(X_test, mu, sigma), preprocess(y_test, mu, sigma)

print("Build model")
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
checkpointer = ModelCheckpoint(filepath="weights/" + "weights_" + approach + ".hdf5", verbose=1, save_best_only=True)
callbacks = [early_stopping, checkpointer]
model = build(timesteps, input_dim)

if load_weights:
  print("Load weights")
  model.load_weights("weights/" + "weights_" + approach + ".hdf5")
else:
  print("Train model")
  hist = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.1, callbacks=callbacks)
  sp.plot_lc(hist, to_file="img/lc_" + approach + ".png")
  plot(model, to_file="img/model_" + approach + ".png")

  hist_file = open("eval/hist_" + approach + ".np", "wb")
  hist.to_file(hist_file)
  hist_file.close()

print("Predict")
pred, test_sample = predict(X_test, model, pred_len)
fname = "pred_" + approach
pred_file = open("data/pred/raw/" + fname + '.np', 'wb')
pred.tofile(pred_file)
pred_file.close()

# evaluate(pred, test_sample, y_test, save_plot="img/pred_" + approach + ".png")
pred = postprocess(pred, approach, pred_len)
sp.save_ahocoder_pred(pred, fname)

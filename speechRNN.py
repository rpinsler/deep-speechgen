# TODO:
## replace linear layer by GMM layer
## try scikit-learn for parameter optimization
## check activations, at least of the last layer

from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.datasets.data_utils import get_file
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

# load speech data
# find ../EN_livingalone/train/wav/ -type f -name "*.wav" | head -100 > split_ids.txt
# python ../fbank_features/signal2logspec.py -p
speech = []
timesteps = 100
timesteps_x = timesteps - 1
shift = 10
path = "../EN_livingalone/train/"

# for file in os.listdir(path + "wav/"):
#     if file.endswith(".logspec.npy"):
#         data = np.load(path + "wav/" + file)
#         nsamples = (data.shape[0] - timesteps) // shift
#         for t in range(nsamples):
#           speech.append(data[t*shift : timesteps + t*shift, :])

# load binary files
lf0 = np.fromfile('f0.bin', 'float32')
lf0 = np.reshape(lf0, (len(lf0), 1))
lf0[lf0 < -1e10] = -1e10
mcp = np.fromfile('cc.bin', 'float32')
mcp = np.reshape(mcp, (len(lf0),mcp.size//len(lf0)))
mfv = np.fromfile('fv.bin', 'float32')
mfv = np.reshape(mfv, (len(mfv), 1))
data = np.hstack((lf0, mcp, mfv))

mu = data.mean(0)
sigma = data.std(0)
data = (data - mu) / sigma # normalize

plt.imshow(data, aspect='auto', interpolation='nearest')
plt.show()

nsamples = (data.shape[0] - timesteps) // shift
for t in range(nsamples):
  speech.append(data[t*shift : timesteps + t*shift, :])

speech = np.dstack(speech).swapaxes(1,2).swapaxes(0,1) # tensor of [nb_samples, timesteps, input_dim]
nb_samples = speech.shape[0]
input_dim = speech.shape[2]

(X_train, y_train), (X_test, y_test) = train_test_split(speech)  # retrieve data

# load transcripts
with open(path + "speech_transcript.txt", 'r') as f:
  read_data = f.read().lower().split("\n")

read_data = read_data[:-1]
transcripts = {}
for line in read_data:
  parts = line.split("\"")
  k = parts[0][1:-1]
  v = parts[1][:-1]
  transcripts[k] = v

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
hist = model.fit(X_train, y_train, batch_size=128, nb_epoch=20, validation_split = 0.1) # , callbacks=[early_stopping]
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.legend(['train loss', 'validation loss'], loc='upper right')
plt.show()
plot(model, to_file='model.png')

pred_len = 5000
for i in range(min(X_test.shape[0], 1)): # what's the benefit of fitting the model n times?
  print('-' * 50)
  print('Iteration', i+1)
  # model.fit(X_train, y_train, batch_size=128, nb_epoch=1)

  generated = []
  signal = X_test[i,:,:].reshape(1,timesteps_x,input_dim)

  for n in range(pred_len):
    preds = model.predict(signal, verbose=0)[0]
    generated.append(preds)
    signal = np.concatenate((signal[:,1:,:], preds.reshape((1,1,input_dim))),1)

gen = np.array(generated, 'float32')
gen = np.vstack(gen)
gen = gen * sigma + mu
X_test_norm = X_test[i,:,:] * sigma + mu

plt.figure()
plt.imshow(X_test_norm.T, aspect='auto', interpolation='nearest')
plt.figure()
plt.imshow(np.vstack((X_test_norm, gen)).T, aspect='auto', interpolation='nearest')
plt.show()

# write back to files
lf0_pred, mcp_pred, mfv_pred = gen[:,0], gen[:,1:-1].flatten(), gen[:,-1]

lf0_file = open('f0_pred.bin', 'wb')
lf0_pred.tofile(lf0_file)
lf0_file.close()

mcp_file = open('cc_pred.bin', 'wb')
mcp_pred.tofile(mcp_file)
mcp_file.close()

mfv_file = open('fv_pred.bin', 'wb')
mfv_pred.tofile(mfv_file)
mfv_file.close()

def train_test_split(data, test_size=0.1):  
    """
    This just splits data to training and testing parts.
    data: tensor of [nb_samples, timesteps, input_dim]
    """
    ntrn = round(data.shape[0] * (1 - test_size))

    X_train, y_train = data[:ntrn,:-1,:], data[:ntrn,-1,:]
    X_test, y_test = data[ntrn:,:-1,:], data[ntrn:,-1,:]

    return (X_train, y_train), (X_test, y_test)

def get_layer_outputs(layer, model, input, train=False):
  fn = theano.function([model.layers[0].input], model.layers[layer].get_output(train=train))
  return fn(input) 

def get_layer_weights(layer, model):
  return model.layers[layer].get_weights()
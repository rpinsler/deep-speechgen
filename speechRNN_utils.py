import os
import numpy as np
import theano.tensor as T
from functools import partial
import matplotlib.pyplot as plt
import subprocess

def load_spectrogram_data(path="../EN_livingalone/train/wav/"):
  """
  Loads all the spectrogram training data
  """
  # python ../fbank_features/signal2logspec.py -p
  data = []
  for file in os.listdir(path):
    if file.endswith(".logspec.npy"):
        data.append(np.load(path + file))
  return data

def load_ahocoder_data(nsamples=0, use_delta=True, path_lf0="data/train/lf0/", path_mcp="data/train/mcp/", path_mfv="data/train/mfv/"):
  """
  Loads all the ahocoder training data
  """

  data = []
  i = 0
  for file in os.listdir(path_lf0):
    if file.endswith(".lf0" + use_delta*'d'):
      if (nsamples != 0) & (nsamples == i):
        break
      i += 1

      lf0 = np.fromfile(path_lf0 + file, 'float32')
      lf0 = np.reshape(lf0, (-1, 1+2*use_delta))
      mcp = np.fromfile(path_mcp + file.split(".")[0] + ".mcp" + use_delta*'d', 'float32')
      mcp = np.reshape(mcp, (-1, mcp.size//lf0.shape[0]))
      mfv = np.fromfile(path_mfv + file.split(".")[0] + ".mfv" + use_delta*'d', 'float32')
      mfv = np.reshape(mfv, (-1, 1+2*use_delta))

      idx = (lf0 == -1e10)
      lf0[idx] = mfv[idx] = 'NaN'

      data.append(np.hstack((mcp, lf0, mfv))) # 40dims mcp, 1dim lf0, 1dim mfv

  return data

def save_ahocoder_pred(pred, i=1, path_lf0="data/pred/lf0/", path_mcp="data/pred/mcp/", path_mfv="data/pred/mfv/"):
  """
  Saves prediction of a single test sample
  """
  pred = np.array(pred, 'float32')
  mcp_pred, lf0_pred, mfv_pred = pred[:,0:-2].flatten(), pred[:,-2:-1], pred[:,-1:]
  idx = np.logical_and(np.isnan(lf0_pred), np.isnan(mcp_pred))
  lf0_pred[idx] = -1e10 
  mfv_pred[idx] = 0

  lf0_file = open(path_lf0 + 'pred_' + str(i) + '.lf0', 'wb')
  lf0_pred.tofile(lf0_file)
  lf0_file.close()

  mcp_file = open(path_mcp + 'pred_' + str(i) + '.mcp', 'wb')
  mcp_pred.tofile(mcp_file)
  mcp_file.close()

  mfv_file = open(path_mfv + 'pred_' + str(i) + '.mfv', 'wb')
  mfv_pred.tofile(mfv_file)
  mfv_file.close()

def load_transcripts(path="../EN_livingalone/train/"):
  """
  Loads all transcripts corresponding to the speech files and 
  maps each transcript to speech file id. 
  """
  with open(path + "speech_transcript.txt", 'r') as f:
    read_data = f.read().lower().split("\n")

  read_data = read_data[:-1]
  transcripts = {}
  for line in read_data:
    parts = line.split("\"")
    k = parts[0][1:-1]
    v = parts[1][:-1]
    transcripts[k] = v

  return transcripts

def get_vuv_flag(data, use_delta=True):
  """
  Extracts voiced/unvoiced flag as additional feature, indicating whether the timestep is voiced
  and therefore the values for lf0 and mfv are properly defined.
  """
  idx_lf0, idx_mfv = 40, 41
  if use_delta:
    idx_lf0 *= 3
    idx_mfv *= 3
  vuv = np.logical_not(np.logical_and(np.isnan(data[:,idx_lf0]), np.isnan(data[:,idx_mfv])))
  return vuv[np.newaxis].T

def interp_uv(data, use_delta=True):
  def interpolate(y):
    nans, x = np.isnan(y), lambda z: z.nonzero()[0]
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y
  
  frm, to = 40, 42
  if use_delta:
    frm *= 3
    to *= 3

  for i in range(frm, to):
    interpolate(data[:,i])
  return data

def replace_uv(data, vuv=None):
  if vuv == None:
    vuv = data[:,-1]
    data = data[:,:-1]
  
  idx_uv = vuv[vuv < 0.5]
  data[:,40] = np.nan 
  data[:,41] = np.nan
  return data

def get_feature_dim():
  DIM_MCP = 40
  DIM_LF0 = 1
  DIM_MFV = 1
  return DIM_MCP, DIM_LF0, DIM_MFV 

def split_features(data, use_delta=True, use_vuv=False):
  dim_mcp, dim_lf0, dim_mfv = get_feature_dim()
  if use_delta:
    mcp, mcpd, mcpdd = data[:,:dim_mcp], data[:,dim_mcp:dim_mcp*2], data[:,dim_mcp*2:dim_mcp*3]
    off = dim_mcp*3
    lf0, lf0d, lf0dd = data[:,off:off+dim_lf0], data[:,off+dim_lf0:off+dim_lf0*2], data[:,off+dim_lf0*2:off+dim_lf0*3]
    off = off+dim_lf0*3 
    mfv, mfvd, mfvdd = data[:,off:off+dim_mfv], data[:,off+dim_mfv:off+dim_mfv*2], data[:,off+dim_mfv*2:off+dim_mfv*3]
  else:
    mcp, lf0, mfv = data[:,:dim_mcp], data[:,dim_mcp:dim_mcp+dim_lf0], data[:,dim_mcp+dim_lf0:dim_mcp+dim_lf0+dim_mfv]
    mcpd = mcpdd = lf0d = lf0dd = mfvd = mfvdd = None
  if use_vuv:
    vuv = data[:,-1]
  else:
    vuv = None
  return (mcp, mcpd, mcpdd), (lf0, lf0d, lf0dd), (mfv, mfvd, mfvdd), vuv

def split_samples(samples, vuv_samples=None, timesteps=100, shift=10):
  """
  Splits all samples into multiple sub-samples with fixed length.
  """
  data = []
  vuv = []
  for sample in zip(samples,vuv_samples):
    nsplits = (sample[0].shape[0] - timesteps) // shift
    for s in range(nsplits):
      data.append(sample[0][s*shift : timesteps + s*shift, :])
      vuv.append(sample[1][s*shift : timesteps + s*shift, :])
  
  return data, vuv

def normalize_data(data):
  """
  Normalizes each feature of the data using z-normalization.
  """

  mu = data.mean((0,1))
  sigma = data.std((0,1))
  data -= mu
  data /= sigma

  return data, mu, sigma

def denormalize_data(data, mu, sigma):
  """
  Denormalizes each feature of the data by reversing the z-normalization.
  """
  return data * sigma + mu

def scale_output(training_data, output):
  maximum = np.max(training_data, axis=(0,1))
  minimum = np.min(training_data, axis=(0,1))
  return 0.01 + (output-minimum)*(0.99-0.01)/(maximum-minimum)

def train_test_split(data, test_size=0.1):  
  """
  Splits data into training and test sets.
  data: tensor of [nb_samples, timesteps, input_dim]
  """
  ntrn = round(data.shape[0] * (1 - test_size))

  X_train, y_train = data[:ntrn,:-1,:], data[:ntrn,-1,:]
  X_test, y_test = data[ntrn:,:-1,:], data[ntrn:,-1,:]

  return (X_train, y_train), (X_test, y_test)

def mlpg(mean, variance, i, path_delta="SPTK-3.8/windows/delta", path_accel="SPTK-3.8/windows/accel"):
  # mkdir SPTK-3.8/windows
  # cd SPTK-3.8/windows
  # echo "-0.5 0 0.5" | x2x +af > delta
  # echo "0.25 -0.5 0.25" | x2x +af > accel
  fname, M = save_mlpg_pred(mean, variance, i)
  cmd = 'mlpg -l '+str(M)+' -d '+path_delta+' -d '+path_accel+' '+fname 
  proc = subprocess.Popen(cmd, cwd=os.getcwd(), stdout=subprocess.PIPE, shell=True)
  pred = np.fromstring(proc.stdout.read(),dtype="float32")
  pred = np.reshape(pred, (-1, 42))
  return pred

def save_mlpg_pred(mean, variance, i=1, path="data/pred/mlpg/"):
  M = mean.shape[1]/3
  pred = np.vstack((mean, variance)).T.flatten() # flatten time-step major (features change fastest)
  pred = np.array(pred, 'float32')
  fname = path + 'pred_' + str(i) + '.mlpg'
  pred_file = open(fname, 'wb')
  pred.tofile(pred_file)
  pred_file.close()
  return fname, M

def get_layer_outputs(layer, model, input, train=False):
  """
  Returns output of particular layer of a network
  """
  fn = theano.function([model.layers[0].input], model.layers[layer].get_output(train=train))
  return fn(input) 

def get_layer_weights(layer, model):
  """
  Returns weights of particular layer of a network
  """
  return model.layers[layer].get_weights()

def plot_lc(hist):
  """
  Plots the learning curve of a network
  """
  plt.plot(hist.history["loss"])
  plt.plot(hist.history["val_loss"])
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.legend(['train loss', 'validation loss'], loc='upper right')
  plt.show()

def gmm_activation(M):
  """
  GMM-like activation function.
  Assumes that input has (M+2)*D dimensions, where D is the dimensionality of the target data.
  The first M*D features are treated as means, the next M features as variances and the last M features 
  as mixture components of the GMM. 
  """

  def gmm_activation_fn(x): 
    D = T.shape(x)[1]/(M+2)
    return T.concatenate([x[:,:M*D-1], T.exp(x[:,M*D:2*M*D-1]), T.nnet.sigmoid(x[:,2*M*D:])], axis=1)

def gmm_loss(M):
  """
  GMM loss function.
  Assumes that y_pred has (M+2)*D dimensions and y_true has D dimensions.
  The first M*D features are treated as means, the next M features as variances and the last M features 
  as mixture components of the GMM. 
  """
  def loss(m, M, y_true, y_pred):
    D = T.shape(y_true)[1]
    return (y_pred[:,(D+1)*M+m]/y_pred[:,D*M+m]) * T.exp(-T.sum(T.sqr(y_pred[:,D*m:(m+1)*D] - y_true))/(2*y_pred[:,D*M+m]**2))

  def gmm_loss_fn(y_true, y_pred):
    seq = T.arange(M)
    result, _ = theano.scan(fn=loss, outputs_info=None, sequences=seq, non_sequences=[M, y_true, y_pred])
    return result.sum()

  return gmm_loss_fn

def evaluate(y_true, y_pred):
  vuv_true = y_true[:,-1:]
  mcp_true, lf0_true = y_true[vuv_true,0:-3], y_true[vuv_true,-3:-2], 
  mcp_pred, lf0_pred, vuv_pred = y_pred[vuv_true,0:-3], y_pred[vuv_true,-3:-2], y_pred[:,-1:]
  return mmcd(mcp_true, mcp_pred), rmse(lf0_true, lf0_pred), class_error(vuv_true, np.round(vuv_pred)) 

def rmse(y_true, y_pred):
  return np.sqrt(np.mean(np.square(y_true-y_pred),-1))

def class_error(y_true, y_pred):
  return np.sum(np.not_equal(y_true, y_pred))/len(y_true)

def mmcd(y_true, y_pred):
  alpha = 10*math.sqrt(2)/np.log(10)
  return alpha*np.mean(rmse(y_true,y_pred))

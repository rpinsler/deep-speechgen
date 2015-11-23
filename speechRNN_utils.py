def load_spectrogram_data(path = "../EN_livingalone/train/wav/"):
  """
  Loads all the spectrogram training data
  """
  # python ../fbank_features/signal2logspec.py -p
  data = []
  for file in os.listdir(path):
    if file.endswith(".logspec.npy"):
        data.append(np.load(path + file))
  return data

def load_ahocoder_data(path_lf0 = "data/train/lf0/", path_mcp = "data/train/mcp/", path_mfv = "data/train/mfv/"):
  """
  Loads all the ahocoder training data
  """
  for file in os.listdir(path_lf0)
    lf0 = np.fromfile(path_lf0 + file, 'float32')
    lf0 = np.reshape(lf0, (len(lf0), 1))
    lf0[lf0 < -1e10] = -1e10

    mcp = np.fromfile(path_mcp + file.split(".")[0] + ".mcp", 'float32')
    mcp = np.reshape(mcp, (len(lf0),mcp.size//len(lf0)))
    
    mfv = np.fromfile(path_mfv + file.split(".")[0] + ".mfv", 'float32')
    mfv = np.reshape(mfv, (len(mfv), 1))

  return np.hstack((lf0, mcp, mfv))

def save_ahocoder_pred(pred, path_lf0 = "data/pred/lf0/", path_mcp = "data/pred/mcp/", path_mfv = "data/pred/mfv/"):
  """
  Saves prediction of a single test sample
  """
  pred = np.array(pred, 'float32')
  lf0_pred, mcp_pred, mfv_pred = pred[:,0], pred[:,1:-1].flatten(), pred[:,-1]

  lf0_file = open('f0_pred.bin', 'wb')
  lf0_pred.tofile(lf0_file)
  lf0_file.close()

  mcp_file = open('cc_pred.bin', 'wb')
  mcp_pred.tofile(mcp_file)
  mcp_file.close()

  mfv_file = open('fv_pred.bin', 'wb')
  mfv_pred.tofile(mfv_file)
  mfv_file.close()

def load_transcripts(path = "../EN_livingalone/train/"):
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

def split_samples(samples, timesteps = 100, shift = 10):
  """
  Splits all samples into multiple sub-samples with fixed length.
  """
  data = []
  for sample in samples:
    nsplits = (sample.shape[0] - timesteps) // shift
    for s in range(nsplits):
      data.append(sample[s*shift : timesteps + s*shift, :])
  
  return data

def normalize_data(data):
  """
  Normalizes each feature of the data using z-normalization.
  """
  mu = data.mean(0)
  sigma = data.std(0)
  data = (data - mu) / sigma
  
  return data, mu, sigma

def denormalize_data(data, mu, sigma):
  """
  Denormalizes each feature of the data by reversing the z-normalization.
  """
  return data * sigma + mu

def train_test_split(data, test_size=0.1):  
  """
  Splits data into training and test sets.
  data: tensor of [nb_samples, timesteps, input_dim]
  """
  ntrn = round(data.shape[0] * (1 - test_size))

  X_train, y_train = data[:ntrn,:-1,:], data[:ntrn,-1,:]
  X_test, y_test = data[ntrn:,:-1,:], data[ntrn:,-1,:]

  return (X_train, y_train), (X_test, y_test)

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
  plt.legend(['train loss', 'validation loss'], loc='upper right')
  plt.show()
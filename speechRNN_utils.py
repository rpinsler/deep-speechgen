# from __future__ import division

import math
import os
import numpy as np
import theano.tensor as T
import subprocess
import theano
from keras.layers.core import Layer
from keras.objectives import binary_crossentropy, mse

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_spectrogram_data(nsamples=0, path="data/train/spectrogram_80_40/"):
    """
    Loads all the spectrogram training data
    """
    # python ../fbank_features/signal2logspec.py -p
    data = []
    i = 0
    for file in os.listdir(path):
        if file.endswith(".logspec.npy"):
            if (nsamples != 0) & (nsamples == i):
                break
            i += 1

            data.append(np.load(path + file))
    return data


def load_ahocoder_data(nsamples=0, use_delta=True, path_lf0="data/train/lf0/",
                       path_mcp="data/train/mcp/", path_mfv="data/train/mfv/"):
    """
    Loads all the ahocoder training data
    """

    data = []
    d = use_delta*'d'
    i = 0
    for file in os.listdir(path_lf0):
        if file.endswith(".lf0" + d):
            if (nsamples != 0) & (nsamples == i):
                break
            i += 1

            lf0 = np.fromfile(path_lf0+file, 'float32')
            lf0 = np.reshape(lf0, (-1, 1+2*use_delta))
            mcp = np.fromfile(path_mcp+file.split(".")[0]+".mcp"+d, 'float32')
            mcp = np.reshape(mcp, (-1, mcp.size//lf0.shape[0]))
            mfv = np.fromfile(path_mfv+file.split(".")[0]+".mfv"+d, 'float32')
            mfv = np.reshape(mfv, (-1, 1+2*use_delta))
            idx = np.where(lf0 == -1e10)
            lf0[idx] = mfv[idx] = 'NaN'
            data.append(np.hstack((mcp, lf0, mfv)))  # 40x mcp, 1x lf0, 1x mfv

    return data


def save_ahocoder_pred(pred, file_name="", path_lf0="data/pred/lf0/",
                       path_mcp="data/pred/mcp/", path_mfv="data/pred/mfv/"):
    """
    Saves prediction of a single test sample
    """
    pred = np.array(pred, 'float32')
    mcp_pred = pred[:, 0:-2].flatten()
    lf0_pred, mfv_pred = pred[:, -2:-1], pred[:, -1:]
    idx = np.where(np.logical_and(np.isnan(lf0_pred), np.isnan(mfv_pred)).flatten())
    lf0_pred[idx] = -1e10
    mfv_pred[idx] = 0

    lf0_file = open(path_lf0 + file_name + '.lf0', 'wb')
    lf0_pred.tofile(lf0_file)
    lf0_file.close()

    mcp_file = open(path_mcp + file_name + '.mcp', 'wb')
    mcp_pred.tofile(mcp_file)
    mcp_file.close()

    mfv_file = open(path_mfv + file_name + '.mfv', 'wb')
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
    Extracts voiced/unvoiced flag as additional feature,
    indicating whether the timestep is voiced and therefore
    the values for lf0 and mfv are properly defined.
    """

    idx_lf0, idx_mfv = 40, 41
    if use_delta:
        idx_lf0 *= 3
        idx_mfv *= 3
    vuv = np.logical_not(np.logical_and(np.isnan(data[:, idx_lf0]),
                                        np.isnan(data[:, idx_mfv])))
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
        interpolate(data[:, i])
    return data


def replace_uv(data, vuv=None):
    if vuv is None:
        vuv = data[:, -1]
        data = data[:, :-1]

    data[np.logical_not(vuv), 40] = np.nan
    data[np.logical_not(vuv), 41] = np.nan
    return data


def get_feature_dim():
    DIM_MCP = 40
    DIM_LF0 = 1
    DIM_MFV = 1
    return DIM_MCP, DIM_LF0, DIM_MFV


def split_features(data, use_delta=True, use_vuv=False):
    dim_mcp, dim_lf0, dim_mfv = get_feature_dim()
    if use_delta:
        mcp = data[:, :dim_mcp]
        mcpd = data[:, dim_mcp:dim_mcp*2]
        mcpdd = data[:, dim_mcp*2:dim_mcp*3]
        off = dim_mcp*3
        lf0 = data[:, off:off+dim_lf0]
        lf0d = data[:, off+dim_lf0:off+dim_lf0*2]
        lf0dd = data[:, off+dim_lf0*2:off+dim_lf0*3]
        off = off+dim_lf0*3
        mfv = data[:, off:off+dim_mfv]
        mfvd = data[:, off+dim_mfv:off+dim_mfv*2]
        mfvdd = data[:, off+dim_mfv*2:off+dim_mfv*3]
    else:
        mcp = data[:, :dim_mcp]
        lf0 = data[:, dim_mcp:dim_mcp+dim_lf0]
        mfv = data[:, dim_mcp+dim_lf0:dim_mcp+dim_lf0+dim_mfv]
        mcpd = mcpdd = lf0d = lf0dd = mfvd = mfvdd = None
    if use_vuv:
        vuv = data[:, -1]
    else:
        vuv = None
    return (mcp, mcpd, mcpdd), (lf0, lf0d, lf0dd), (mfv, mfvd, mfvdd), vuv


def split_samples(samples, timesteps=100, shift=10):
    """
    Splits all samples into multiple sub-samples with fixed length.
    """
    data = []
    for sample in samples:
        nsplits = (sample.shape[0] - timesteps) // shift
        for s in range(nsplits):
            data.append(sample[s*shift:timesteps+s*shift, :])

    return data


def normalize_data(data, mu=None, sigma=None):
    """
    Normalizes each feature of the data using z-normalization.
    """

    if (mu is None) & (sigma is None):
        mu = data.mean((0, 1))
        sigma = data.std((0, 1))

    data -= mu
    data /= sigma

    return data, mu, sigma


def denormalize_data(data, mu, sigma):
    """
    Denormalizes each feature of the data by reversing the z-normalization.
    """
    return data * sigma + mu


def scale_output(training_data, output):
    maximum = np.max(training_data, axis=(0, 1))
    minimum = np.min(training_data, axis=(0, 1))
    return 0.01 + (output-minimum)*(0.99-0.01)/(maximum-minimum)


def train_test_split(data, pred_len=1, test_size=0.1):
    """
    Splits data into training and test sets.
    data: tensor of [nb_samples, timesteps, input_dim]
    """
    ntrn = round(data.shape[0] * (1 - test_size))

    X_train, y_train = data[:ntrn, :-pred_len, :], data[:ntrn, -pred_len:, :]
    X_test, y_test = data[ntrn:, :-pred_len, :], data[ntrn:, -pred_len:, :]
    return (X_train, y_train), (X_test, y_test)


def mlpg(mean, variance, approach, path_delta="SPTK-3.8/windows/delta",
         path_accel="SPTK-3.8/windows/accel"):
    # mkdir SPTK-3.8/windows
    # cd SPTK-3.8/windows
    # echo "-0.5 0 0.5" | x2x +af > delta
    # echo "0.25 -0.5 0.25" | x2x +af > accel
    fname, M = save_mlpg_pred(mean, variance, approach)
    cmd = 'mlpg -l '+str(M)+' -d '+path_delta+' -d '+path_accel+' '+fname
    proc = subprocess.Popen(cmd, cwd=os.getcwd(),
                            stdout=subprocess.PIPE, shell=True)
    pred = np.fromstring(proc.stdout.read(), dtype="float32")
    pred = np.reshape(pred, (-1, 42))
    return pred


def save_mlpg_pred(mean, variance, approach, path="data/pred/mlpg/"):
    M = mean.shape[1]/3
    # flatten time-step major (features change fastest)
    pred = np.vstack((mean, variance)).T.flatten()
    pred = np.array(pred, 'float32')
    fname = path + 'pred_' + approach + '.mlpg'
    pred_file = open(fname, 'wb')
    pred.tofile(pred_file)
    pred_file.close()
    return fname, M


def get_layer_outputs(layer, model, input, train=False):
    """
    Returns output of particular layer of a network
    """
    fn = theano.function([model.layers[0].input],
                         model.layers[layer].get_output(train=train))
    return fn(input)


def get_layer_weights(layer, model):
    """
    Returns weights of particular layer of a network
    """
    return model.layers[layer].get_weights()


def get_loss(model, X_test, y_test, idx):
    y_true = T.matrix()
    y_pred = T.matrix()
    out = get_layer_outputs(-1, model, X_test[idx:idx+1, :, :])
    return gmm_loss(y_true, y_pred).eval({y_true: y_test[idx:idx+1, :], y_pred: out})


def plot_lc(hist, to_file=None):
    """
    Plots the learning curve of a network
    """
    plt.figure()
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(['train loss', 'validation loss'], loc='upper right')

    if to_file is None:
        plt.show()
    else:
        plt.savefig(to_file, dpi=100)


def evaluate(y_true, y_pred):
    vuv_true = y_true[:, -1]
    mcp_true, lf0_true = y_true[:, :-3], y_true[:, -3:-2]
    mcp_pred, lf0_pred = y_pred[:, :-3], y_pred[:, -3:-2]
    vuv_pred = y_pred[:, -1]
    err = class_error(vuv_true, np.round(vuv_pred))
    return mmcd(mcp_true, mcp_pred), rmse(lf0_true, lf0_pred), err


def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true-y_pred)))


def class_error(y_true, y_pred):
    return 1 - np.sum(np.equal(y_true, y_pred))/float(len(y_true))


def mmcd(y_true, y_pred):
    alpha = 10*math.sqrt(2)/np.log(10)
    return alpha*np.mean(rmse(y_true, y_pred))


class GMMActivation(Layer):
    """
    GMM-like activation function.
    Assumes that input has (D+2)*M dimensions, where D is the dimensionality
    of the target data. The first M*D features are treated as means, the next
    M features as standard devs and the last M features as mixture components
    of the GMM.
    """
    def __init__(self, M, **kwargs):
        super(GMMActivation, self).__init__(**kwargs)
        self.M = M

    def get_output(self, train=False):
        X = self.get_input(train)
        D = T.shape(X)[1]/self.M - 2
        # leave mu values as they are since they're unconstrained
        # scale sigmas with exp, s.t. all values are non-negative
        X = T.set_subtensor(X[:, D*self.M:(D+1)*self.M],
                            T.exp(X[:, D*self.M:(D+1)*self.M]))
        # scale alphas with softmax, s.t. all values are in [0,1] and sum to 1
        X = T.set_subtensor(X[:, (D+1)*self.M:(D+2)*self.M],
                            T.nnet.softmax(X[:, (D+1)*self.M:(D+2)*self.M]))
        # apply sigmoid to vuv bit
        X = T.set_subtensor(X[:, -1], T.nnet.sigmoid(X[:, -1]))
        return X

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "M": self.M}
        base_config = super(GMMActivation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def gmm_loss(y_true, y_pred):
    """
    GMM loss function.
    Assumes that y_pred has (D+2)*M dimensions and y_true has D dimensions.
    The first M*D features are treated as means, the next M features as
    standard devs and the last M features as mixture components of the GMM.
    """
    def loss(m, M, D, y_true, y_pred):
        mu = y_pred[:, D*m:(m+1)*D]
        sigma = y_pred[:, D*M+m]
        alpha = y_pred[:, (D+1)*M+m]
        return (alpha/sigma) * T.exp(-T.sum(T.sqr(mu-y_true), -1)/(2*sigma**2))

    D = T.shape(y_true)[1] - 1
    M = (T.shape(y_pred)[1] - 1)/(D+2)
    seq = T.arange(M)
    result, _ = theano.scan(fn=loss, outputs_info=None, sequences=seq,
                            non_sequences=[M, D, y_true[:, :-1],
                                           y_pred[:, :-1]])
    # add loss for vuv bit
    vuv_loss = binary_crossentropy(y_true[:, -1], y_pred[:, -1])
    # vuv_loss = 0
    return -T.log(result.sum(0) + 1e-7) - vuv_loss


class LinearVUVActivation(Layer):
    def __init__(self, **kwargs):
        super(LinearVUVActivation, self).__init__(**kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        # apply sigmoid to vuv bit
        X = T.set_subtensor(X[:, -1], T.nnet.sigmoid(X[:, -1]))
        return X

    def get_config(self):
        config = {"name": self.__class__.__name__}
        base_config = super(LinearVUVActivation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def mse_crossentropy(y_true, y_pred):
    vuv_loss = binary_crossentropy(y_true[:, -1], y_pred[:, -1])
    return mse(y_true[:, :-1], y_pred[:, :-1]) * vuv_loss


def gmm_sample(X, M):
    D = np.shape(X)[0]/M - 2
    alphas = X[(D+1)*M:(D+2)*M]
    if sum(alphas) > 1.0:  # shouldn't happen but is possible due to numeric errors
        alphas /= sum(alphas)

    m = np.argmax(np.random.multinomial(1, alphas, 1))
    mu = X[D*m:(m+1)*D]
    sigma = X[D*M+m]
    return np.random.normal(mu, sigma), sigma, alphas


def bernoulli_sample(X):
    return int(np.random.rand() < X)


def sample(X, M):
    gmm, sigma, alphas = gmm_sample(X[:-1], M)
    vuv = np.asarray(bernoulli_sample(X[-1]))
    return np.hstack((gmm, vuv)), sigma, alphas


def load_speech_results(approach, path='data/pred/wav16/'):
    mcp = np.load(path + "pred_" + approach + ".logspec.npy")
    vuv = np.load(path + approach + ".vuv.npy")

    try:
        alphas = np.load(path + approach + ".alpha.npy")
    except IOError:
        alphas = None

    return mcp, vuv, alphas

# TODO for first paper submission (25.01.2016, 23:59 GMT+1):
# 1. run basic MDN
# 5. try prediction and analyze error (discriminative)
#    -> save results, maybe have to improve method
# 6. generate spectrogram only
#    -> try other fbank params,
#    -> regularization

# TODO for final presentations (Friday, 12.02.2016, 14h-18h):
# 1. finalize presentation

# TODO for second paper submission (Monday, 22.02.2016, 23:59 GMT+1):
# 1. update paper according to feedback
# 2. try more advanced stuff


# TODO:
# try scikit-learn for parameter optimization
# check activations, at least of the last layer

# open issues:
# why normalize outputs to lie between [0.01, 0.99]
# postfiltering in cepstral domain?

# things to try out:
# deeper architecture
# regularization, e.g. Dropout, BatchNormalization, L1/L2, structured reg
# given timesteps 1:t, predict t+1:t+n
# given timesteps 1:t, predict 2:t+1
# different initialization
# EarlyStopping
# gradient clipping
# RNN-specific tricks

# practical considerations from Goodfellow/Bengio:
# optimizer: SGD w/ momentum w/ decaying learning rate
# (linearly, exponentially, by a factor of 2-10)
# regularization: early stopping, dropout, batch normalization
# unsupervised pre-training (if it's known to be helpful)
# hyperparams: manually vs. automatically
# <hyperparam>: <increases capacity when>
# no of hidden units: increased
# learning rate: tuned optimally
# weight decay coefficient: decreased
# dropout rate: decreased
# monitor histograms of activations and gradient (see details in book)

from __future__ import print_function
from __future__ import absolute_import

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, TimeDistributedDense, Flatten
from keras.layers.noise import GaussianNoise
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1l2, l2

import pickle
import numpy as np
import speechRNN_utils as sp
from pylab import rcParams
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# originally 112639 frames for 7 seconds at sample rate of 16KHz
# 3872 (16bit), 1936 (32bit),  968 (64bit) extracted vectors
# lframe=80     Frame shift (samples) -> 80/16000 = 0.005 -> 5ms
# window size could be 196 frames -> 196/16000 = 0.01225 -> 12.25ms
# 7*16000/(196-80) = 968 -> suggests 64bit vector representation
# data load suggests 32bit vector representation though


def load(n=0, cut=0, use_delta=False, timesteps=100, shift=10):
    # pre-processing of data:
    # - downsample data from 44.1KHz to 16KHz: python downsample.py
    # - extract audio features: sudo ./ahocoder.sh
    # - extract derivatives of audio features: sudo ./extractdelta.sh

    data = sp.load_ahocoder_data(n, use_delta=use_delta)
    interp_data = []
    vuv_data = []
    for sample in data:
        sample = sample[cut:sample.shape[0]-cut, :]
        vuv = sp.get_vuv_flag(sample, use_delta=use_delta)
        sample = sp.interp_uv(sample, use_delta=use_delta)
        interp_data.append(sample)
        vuv_data.append(vuv)

    data = sp.split_samples(interp_data, timesteps, shift)
    vuv = sp.split_samples(vuv_data, timesteps, shift)
    interp_data = vuv_data = None  # free some space
    data = np.dstack(data).transpose((2, 0, 1))
    vuv = np.dstack(vuv).transpose((2, 0, 1))
    data = np.concatenate((data, vuv), -1)
    return data


def preprocess(data, mu=None, sigma=None):
    data, mu, sigma = sp.normalize_data(data, mu, sigma)
    return data, mu, sigma


def build(timesteps, input_dim, pred_len=1, M=None):
    if M is None:  # linear output layer
        model = Sequential()
        model.add(LSTM(128, return_sequences=True,
                       input_shape=(timesteps, input_dim)))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=False))
        model.add(Dropout(0.2))
        # model.add(TimeDistributedDense(input_dim))
        # model.add(Flatten())
        model.add(Dense(input_dim))
        model.add(sp.LinearVUVActivation())
        # model.compile(loss='mse', optimizer=RMSprop(clipvalue=10))
        model.compile(loss=sp.mse_crossentropy, optimizer=RMSprop(clipvalue=10))
    else:  # GMM output layer
        model = Sequential()
        # model.add(GaussianNoise(0.01, input_shape=(timesteps, input_dim)))
        model.add(LSTM(128, return_sequences=True,
                       input_shape=(timesteps, input_dim)))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense((input_dim-1+2)*M+1))
        # model.add(Dense((input_dim+2)*M))
        model.add(sp.GMMActivation(M))
        model.compile(loss=sp.gmm_loss, optimizer=RMSprop(clipvalue=10))
        # model.compile(loss=sp.gmm_loss, optimizer='adam')
        # model.compile(loss=sp.gmm_loss, optimizer=Adam(clipvalue=10))
        # model.compile(loss=sp.gmm_loss, optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=10))

    return model


def predict(X_test, model, pred_len=1, gen_len=0, M=None):
    generated = []
    alphas = []
    sigmas = []
    nb_samples, timesteps, input_dim = X_test.shape

    # sample start sequence from data
    idx = np.random.randint(nb_samples)
    signal = X_test[idx, :, :].reshape(1, timesteps, input_dim)

    for n in range(0, gen_len, pred_len):
        if (n % 500) == 0:
            print("Predictions completed: " + str(n+1) + "/" + str(gen_len))

        preds = model.predict(signal, verbose=0)[0]
        if M is not None:
            preds, sigma, weights = sp.sample(preds, M)
            alphas.append(weights)
            sigmas.append(sigma)
        generated.append(preds)
        signal = np.concatenate((signal[:, pred_len:, :],
                                 preds.reshape((1, pred_len, input_dim))), 1)

    print("Predictions completed: " + str(n+1) + "/" + str(gen_len))
    pred = np.vstack(generated)
    if M is not None:
        alphas = np.vstack(alphas)
        sigmas = np.vstack(sigmas)
    else:
        alphas = None
        sigmas = None

    return pred, idx, sigmas, alphas


def postprocess(data, approach, pred_len, use_delta=False,
                X_train=None, sigma=None, sigma_gmm=None):
    if use_delta:
        (mcp, mcpd, mcpdd), (lf0, lf0d, lf0dd), (mfv, mfvd, mfvdd), vuv = \
            sp.split_features(data, use_vuv=True)
        data = np.hstack((mcp, lf0, mfv, mcpd, lf0d, mfvd,
                          mcpdd, lf0dd, mfvdd))
        if sigma_gmm is None:
            variance = np.repeat(np.expand_dims(np.square(sigma), 0),
                                 pred_len, 0)
        else:
            variance = np.repeat(np.square(sigma_gmm), data.shape[1], 1)

        # data = sp.mlpg(data, variance, approach)
        data = np.hstack((mcp, lf0, mfv))
        data = sp.replace_uv(data, vuv)
    else:
        data = sp.replace_uv(data)

    return data


def visualize_result(pred, approach, test_sample=None,
                     gen_speech=True, vuv=None, alphas=None):

    if test_sample is not None:
        plt.figure()
        plt.xlabel('t')
        plt.ylabel('mcp')
        plt.imshow(test_sample[:, :40].T,
                   aspect='auto', interpolation='nearest', origin='lower')
        plt.savefig("img/orig_"+approach+".png", dpi=100, bbox_inches='tight')

        plt.figure()
        plt.xlabel('t')
        plt.ylabel('mcp')
        plt.imshow(np.vstack((test_sample[:, :40], pred[:, :40])).T,
                   aspect='auto', interpolation='nearest', origin='lower')
        # plt.vlines(test_sample.shape[0], 0, 38, linestyles='dashdot')
        plt.axvline(x=test_sample.shape[0], linewidth=2, color='black')
        plt.savefig("img/origpred_"+approach+".png", dpi=100, bbox_inches='tight')

    if alphas is None and not gen_speech:
        return  # no need to plot the rest

    nplots = 1 + (alphas is not None) + gen_speech

    plt.figure()
    ax1 = plt.subplot2grid((2+nplots, 5), (0, 0), rowspan=3, colspan=5)

    # f, axarr = plt.subplots(nplots, sharex=True)
    ax1.set_ylabel('mcp')
    ax1.imshow(pred[:, :40].T, aspect='auto', interpolation='nearest', origin='lower')
    # axarr[0].savefig("img/pred_"+approach+".png", dpi=100)

    if alphas is not None:
        # plt.figure()
        # plt.imshow(alphas.T, interpolation="nearest", aspect="auto")
        if alphas.shape[0] == pred.shape[0]:
            sharex = ax1
        else:
            sharex = None
        ax2 = plt.subplot2grid((2+nplots, 5), (3, 0), rowspan=1, colspan=5, sharex=sharex)
        ax2.imshow(alphas.T, aspect='auto', interpolation="nearest")
        # plt.colorbar()
        # plt.axis('off')
        # plt.xlabel('t')
        ax2.set_ylabel('alpha')
        ax2.get_yaxis().set_ticks([])
        # plt.savefig("img/weight_act_"+approach+".png", dpi=100)

    if gen_speech:
        if vuv is None:
            vuv = pred[:, -1]
        vuv = np.expand_dims(vuv, 0)

        if vuv.shape[0] == pred.shape[0]:
            sharex = ax1
        else:
            sharex = None
        # plt.figure()
        ax3 = plt.subplot2grid((2+nplots, 5), (2+nplots-1, 0), rowspan=1, colspan=5, sharex=sharex)
        ax3.imshow(vuv, aspect='auto', interpolation="nearest")
        # plt.xlabel('t')
        # plt.axis('off')
        ax3.set_ylabel('vuv')
        ax3.get_yaxis().set_ticks([])
        # plt.savefig("img/vuvpred_"+approach+".png", dpi=100)

    plt.xlabel('t')
    plt.tight_layout()
    plt.savefig("img/enhpred_"+approach+".png", dpi=100, bbox_inches='tight')


def viz_speech_results(approach, npredictions):
    for i in range(1, npredictions+1):
        pred, vuv, alphas = sp.load_speech_results(str(i)+"_"+approach)
        visualize_result(pred, str(i)+"_"+approach, vuv=vuv, alphas=alphas)


def debug(model):
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

    # keep track of variance of activations

approach = "2lstm_lin"
gen_speech = True
debug_mode = False

load_weights = False
nb_rawsamples = 0
timesteps = 100
shift = 10
batch_size = 128
nb_epoch = 100
pred_len = 1
gen_len = 2000
cut = 50  # no. timesteps removed at beginning and end of each sample
use_delta = False
M = None  # no of GMM components, or None
patience = 10
npredictions = 5

if not gen_speech:
    use_delta = False
if M == 0:
    raise ValueError("M should be greater than zero or None.")

if debug_mode:  # reduce computations to allow local debugging
    nb_rawsamples = 1
    nb_epoch = 1
    gen_len = 10
    npredictions = 1

#### print params
print("Approach: " + approach)
print("Generate speech: " + str(gen_speech))
print("Load weights: " + str(load_weights))
print("Samples: " + str(nb_rawsamples))
print("Timestamps: " + str(timesteps))
print("Shift: " + str(shift))
print("Batch size: " + str(batch_size))
print("Epochs: " + str(nb_epoch))
print("Prediction length: " + str(pred_len))
print("Generation length: " + str(gen_len))
print("Cut: " + str(cut))
print("Use delta: " + str(use_delta))
print("M: " + str(M))
print("Patience: " + str(patience))

print("Load data")
if gen_speech:
    data = load(nb_rawsamples, cut, use_delta, timesteps, shift)
else:
    data = sp.load_spectrogram_data(nb_rawsamples)
    data = sp.split_samples(data, timesteps, shift)
    data = np.dstack(data).transpose((2, 0, 1))
    # vuv = np.expand_dims(np.ones(data.shape[0:2]), 2)
    # data = np.concatenate((data, vuv), -1)

(X_train, y_train), (X_test, y_test) = sp.train_test_split(data, pred_len)
nb_samples, timesteps, input_dim = X_train.shape
data = None  # free some space

print("Preprocess data")
# exclude vuv bit from normalization
if gen_speech or M is not None:
    X_train[:, :, :-1], mu, sigma = preprocess(X_train[:, :, :-1])
    y_train[:, :, :-1], _, _ = preprocess(y_train[:, :, :-1], mu, sigma)
    X_test[:, :, :-1], _, _ = preprocess(X_test[:, :, :-1], mu, sigma)
    y_test[:, :, :-1], _, _ = preprocess(y_test[:, :, :-1], mu, sigma)
else:
    X_train, mu, sigma = preprocess(X_train)
    y_train, _, _ = preprocess(y_train, mu, sigma)
    X_test, _, _ = preprocess(X_test, mu, sigma)
    y_test, _, _ = preprocess(y_test, mu, sigma)

y_train = y_train[:, 0, :]  # only do single step prediction for now
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)

print("Build model")
early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
checkpointer = ModelCheckpoint(
    filepath="weights/" + "weights_" + approach + ".hdf5",
    verbose=1, save_best_only=True)
callbacks = [early_stopping, checkpointer]
model = build(timesteps, input_dim, pred_len=pred_len, M=M)

if load_weights:
    print("Load weights")
    # json = open("models/"+"model_"+approach+".json").read()
    # custom_objects = {"GMMActivation": sp.GMMActivation,
    #                   "gmm_loss": sp.gmm_loss}
    # model = model_from_json(json, custom_objects)
    model.load_weights("weights/" + "weights_" + approach + ".hdf5")
else:
    print("Train model")
    hist = model.fit(X_train, y_train, batch_size=batch_size,
                     nb_epoch=nb_epoch, validation_split=0.1,
                     callbacks=callbacks)
    sp.plot_lc(hist, to_file="img/lc_" + approach + ".png")
    # plot(model, to_file="img/model_" + approach + ".png")

    with open('eval/hist_' + approach + '.np', 'wb') as handle:
        pickle.dump(hist.history, handle)

    open("models/"+"model_"+approach+".json", 'w').write(model.to_json())

for i in range(1, npredictions+1):
    print("Predict")
    pred, test_idx, sigma_gmm, alphas = predict(X_test, model, pred_len=1,
                                                gen_len=gen_len, M=M)
    fname = "pred_" + str(i) + "_" + approach
    pred_file = open("data/pred/raw/" + fname + '.np', 'wb')
    pred.tofile(pred_file)
    pred_file.close()

    if y_test.ndim == 3:  # only makes sense for larger trajectories
        if gen_speech:
            mmcd, rmse, class_err = sp.evaluate(y_test[test_idx, :, :],
                                                pred[:pred_len, :])
            metrics = [mmcd, rmse, class_err]
        else:
            mmcd = sp.mmcd(y_test[test_idx, :, :], pred[:pred_len, :])
            metrics = [mmcd]
        metrics_file = open("eval/metrics_" + str(i) + "_" + approach + '.np', 'wb')
        metrics.tofile(metrics_file)
        metrics_file.close()

    if gen_speech:
        pred[:, -1] = np.round(pred[:, -1])
        pred[:, :-1] = sp.denormalize_data(pred[:, :-1], mu, sigma)
        # pred[:, :-1] = sp.scale_output(X_train[:, :, :-1], pred[:, :-1])
        test_sample = sp.denormalize_data(X_test[test_idx, :, :-1], mu, sigma)
    else:
        if M is None:
            pred = sp.denormalize_data(pred, mu, sigma)
            test_sample = sp.denormalize_data(X_test[test_idx, :, :], mu, sigma)
        else:
            pred[:, :-1] = sp.denormalize_data(pred[:, :-1], mu, sigma)
            test_sample = sp.denormalize_data(X_test[test_idx, :, :-1], mu, sigma)

        with open('data/pred/spectrogram/' + str(i) + "_" + approach +
                  '.logspec.npy', 'wb') as handle:
            pickle.dump(pred, handle)

        matplotlib.rcParams.update({'font.size': 18})
        visualize_result(pred, str(i) + "_" + approach, test_sample,
                         gen_speech, alphas=alphas)

    if gen_speech:
        with open('data/pred/wav16/' + str(i) + "_" + approach +
                  '.vuv.npy', 'wb') as handle:
            pickle.dump(pred[:, -1], handle)
        with open('data/pred/wav16/' + str(i) + "_" + approach +
                  '.alpha.npy', 'wb') as handle:
            pickle.dump(alphas, handle)
        pred = postprocess(pred, str(i) + "_" + approach, pred_len, use_delta=use_delta,
                           X_train=X_train, sigma=sigma, sigma_gmm=sigma_gmm)
        sp.save_ahocoder_pred(pred, fname)

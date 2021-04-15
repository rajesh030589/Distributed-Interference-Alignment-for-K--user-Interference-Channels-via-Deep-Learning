import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from comm import PAMDemod, PAMMod
from tqdm import tqdm
from TIMTIN_n import Distributed_timtin as distn
from TIMTIN_n import Distributed_timtin_predefined as dist_predefined
from TIMTIN_n import Distributed_timtin_choose as dist_choose
import scipy.io as sio


def generate_beamvectors(K, A, n, SNR):
    snr = 10 ** (SNR / 10)
    # snr = 2 * r * snr
    snr = n * snr
    Vp, Up, Rp = distn(n, 1, K, A, snr)
    for k in range(K):
        if np.dot(np.transpose(Vp[k]), Up[k]) < 0:
            Up[k] = -1 * Up[k]
    return Vp, Up, Rp


def generate_beamvectors_predefined(K, A, n, r, SNR, V1, R1):

    snr = 10 ** (SNR / 10)
    snr = 2 * r * snr

    Vp, Up, Rp = dist_predefined(n, 1, K, A, snr, V1, R1)
    for k in range(K):
        if np.dot(np.transpose(Vp[k]), Up[k]) < 0:
            Up[k] = -1 * Up[k]
    return Vp, Up, Rp


def choose_initial_vectors(K, A, n, r, SNR, Vlist, Rlist):

    snr = 10 ** (SNR / 10)
    snr = 2 * r * snr

    V, R = dist_choose(n, 1, K, A, snr, Vlist, Rlist)

    return V, R


def generate_randomvectors(K, n, I):

    Vlist = []
    Rlist = []

    for _ in range(I):

        V = []
        R = []
        for _ in range(K):
            R.append(-np.random.rand(1))
            Vtemp = np.zeros((n, 1))

            for m in range(1):
                v = np.random.rand(n)
                e = np.linalg.norm(v)
                Vtemp[:, m] = v / e
            V.append(Vtemp)

        Vlist.append(V)
        Rlist.append(R)

    return Vlist, Rlist


def prepare_data(K, m, n, r, V, U, R, SNR, L):
    snr = 10 ** (SNR / 10)
    snr = n * snr

    # Transmitter
    Message = []
    SignalIn = []
    TxOut = []

    for k in range(K):
        if m == "Gaussian":
            M = 0
            X = np.random.normal(0, 1, (L, 1))
        else:
            M = np.random.randint(0, 2 ** m, size=L)
            X = PAMMod(M, m, 1)
        VX = X * np.sqrt(snr ** R[k])
        if n != 1:
            VX = np.matmul(VX, np.transpose(V[:, k : k + 1]))

        Message.append(M)
        SignalIn.append(X)
        TxOut.append(VX)

    # Channel
    noise_variance = 1 / (np.sqrt(snr))

    Noise = []
    for k in range(K):
        noise = noise_variance * np.random.randn(TxOut[k].shape[0], TxOut[k].shape[1])
        Noise.append(noise)

    return Message, SignalIn, TxOut, Noise

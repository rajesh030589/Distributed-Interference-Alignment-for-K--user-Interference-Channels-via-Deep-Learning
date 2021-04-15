import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from comm import PAMDemod, PAMMod
from tqdm import tqdm
from MAX_SINR_MODEL import Distributed_maxSINR as distn
from MAX_SINR_MODEL import Distributed_maxSINR_choose as dist_choose
from MAX_SINR_MODEL import Distributed_maxSINR_predfined as dist_predefined
import scipy.io as sio
from math import pi
from sklearn.feature_selection import f_regression, mutual_info_regression


def generate_beamvectors(K, A, n, SNR):

    snr = 10 ** (SNR / 10)
    snr = n * snr

    Vp, Up, Rp = distn(n, 1, K, A, snr)
    for k in range(K):
        if np.dot(np.transpose(Vp[k]), Up[k]) < 0:
            Up[k] = -1 * Up[k]
    return Vp, Up, Rp


def generate_beamvectors_predefined(K, A, n, r, SNR, V1, R1):

    snr = 10 ** (SNR / 10)
    # snr = 2 * r * snr
    snr = 2 * snr

    Vp, Up, Rp, Rate = dist_predefined(n, 1, K, A, snr, V1, R1)

    for k in range(K):
        if np.dot(np.transpose(Vp[k]), Up[k]) < 0:
            Up[k] = -1 * Up[k]
    return Vp, Up, Rp, Rate


def choose_initial_vectors(K, A, n, r, SNR, Vlist, Rlist):

    snr = 10 ** (SNR / 10)
    # snr = 2 * r * snr
    snr = 2 * snr

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
    # snr = 2 * r * snr
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


def maxSINR_comm(K, m, n, r, H, SNR, V, U, Rp, L):

    # Transmitter
    _, SignalIn, TxOut, Noise = prepare_data(K, m, n, r, V, U, Rp, SNR, L)

    snr = 10 ** (SNR / 10)
    # snr = 2 * r * snr
    snr = n * snr

    # Channel
    RxIn = []
    for k in range(K):
        Y = 0
        for k1 in range(K):
            Y += H[k, k1] * TxOut[k1]
        Y = Y + Noise[k]
        RxIn.append(Y)

    SignalOut = []
    for k in range(K):
        UY = RxIn[k]
        if n != 1:
            UY = np.matmul(UY, U[:, k : k + 1])
        UY = UY / np.sqrt(snr ** Rp[k])

        SignalOut.append(UY)

    return SignalIn, SignalOut


def compute_error(K, m, SignalIn, SignalOut):

    BER = np.zeros(K)
    SER = np.zeros(K)
    for k in range(K):
        X = SignalIn[k]
        M = PAMDemod(X, m, 1).astype(int)

        Y = SignalOut[k]
        N = PAMDemod(Y, m, 1).astype(int)

        # SER
        SER[k] = np.count_nonzero(M - N) / (len(SignalIn[0]))

        # BER
        Mc = np.array([tuple(int(c) for c in "{:02b}".format(i)) for i in M])
        M1 = np.array([tuple(int(c) for c in "{:02b}".format(i)) for i in N])
        BER[k] = np.sum(abs(M1 - Mc)) / (m * len(SignalIn[0]))

    return BER, SER


def compute_RATE(K, SignalIn, SignalOut):

    # RATE
    RATE = []
    for k in range(K):
        RATE.append(np.log2(1 + (1 / np.mean((SignalIn[k] - SignalOut[k]) ** 2))))

    return np.sum(RATE)


def compute_capacity(K, m, SignalIn, SignalOut, L1):

    C = np.zeros(K)
    for k in range(K):

        tau = np.zeros((2 ** m, 2 ** m))
        for i in range(L1):
            X = SignalIn[i][k]
            M = PAMDemod(X, m, 1).astype("int")

            Y = SignalOut[i][k]
            N = PAMDemod(Y, m, 1).astype("int")

            tau[M, N] += 1
            # for j in range(len(N)):
            #     tau[int(M[j]), int(N[j])] += 1

        for i in range(2 ** m):
            tau[i, :] = tau[i, :] / np.sum(tau[i, :])

        C[k] = compute_entropy(tau)

    return C


def compute_entropy(tau):

    # Compute H(Y|X)
    tau_n = np.where(tau < 0.00001, 0.00001, tau)
    HYX = -np.mean(np.sum(np.multiply(tau, np.log2(tau_n)), axis=1))

    tau_n1 = np.mean(tau, axis=0)
    tau_n2 = np.where(tau_n1 < 0.00001, 0.00001, tau_n1)
    HY = -np.sum(np.multiply(tau_n1, np.log2(tau_n2)))
    # print("HY : ", HY)
    # print("HYX : ", HYX)
    return HY - HYX


def compute_MSE(K, SignalIn, SignalOut):

    # RATE
    MSE = np.zeros(K)
    for k in range(K):
        MSE[k] = np.mean((SignalIn[k] - SignalOut[k]) ** 2)

    return MSE


def maxSINR_main(K, m, n, r, SNR, H, A):

    L = 1000

    Vp, Up, Rp = generate_beamvectors(K, A, n, SNR)
    if n != 1:
        V = np.transpose(np.array(Vp))[0, :, :]
        U = np.transpose(np.array(Up))[0, :, :]

    SignalIn, SignalOut = maxSINR_ocmm(K, m, n, r, H, SNR, V, U, Rp, L)
    MSE = compute_MSE(K, SignalIn, SignalOut)

    if m == "Gaussian":
        return MSE
    else:
        BER, SER = compute_error(K, m, SignalIn, SignalOut)
        return MSE, BER, SER


def maxSINR_vectors_given(K, m, n, r, SNR, H, V, U, Rp):

    L = 1000

    SignalIn, SignalOut = maxSINR_comm(K, m, n, r, H, SNR, V, U, Rp, L)
    MSE = compute_MSE(K, SignalIn, SignalOut)

    if m == "Gaussian":
        return MSE
    else:
        BER, SER = compute_error(K, m, SignalIn, SignalOut)
        return MSE, BER, SER


def get_mi_estimate(K, m, n, r, SNR, H, V, U, Rp):
    L = 10000

    SignalIn, SignalOut = maxSINR_comm(K, m, n, r, H, SNR, V, U, Rp, L)
    MI = np.zeros(K)
    for k in range(K):
        X = SignalIn[k].reshape(-1, 1)
        Y = np.squeeze(SignalOut[k])
        M = mutual_info_regression(X, Y)
        MI[k] = M

    return MI


def maxSINR_ClosedForm(K, m, n, l, r, SNR, H, V, U, Rp):
    # def maxSINR_ClosedForm(K, m, n, l, r, SNR, H, A):

    # Vp, Up, Rp = generate_beamvectors(K, A, n, SNR)
    snr = n * (10 ** (SNR / 10))
    # V = np.transpose(np.array(Vp))[0, :, :]
    # U = np.transpose(np.array(Up))[0, :, :]

    # VV = {"V": V, "U": U, "R": Rp}
    # pickle.dump(VV, open("Vector.pkl", "wb"))

    MSE = np.zeros(K)
    for k in range(K):
        x = 0
        for k1 in range(K):
            if k == k1:
                x += (
                    1
                    - H[k, k1]
                    * np.matmul(np.transpose(U[:, k : k + 1]), V[:, k1 : k1 + 1])
                ) ** 2
            else:
                x += (
                    H[k, k1]
                    * np.matmul(np.transpose(U[:, k : k + 1]), V[:, k1 : k1 + 1])
                    * (np.sqrt(snr ** Rp[k1]))
                    / (np.sqrt(snr ** Rp[k]))
                ) ** 2
        x = x + snr ** (-1 - Rp[k])

        MSE[k] = x

    return MSE


def maxSINR_main_predefined(K, m, n, r, SNR, H, A, V1, R1, BER_needed):

    if BER_needed:
        L = 10000
    else:
        L = 1000

    Vp, Up, Rp, _ = generate_beamvectors_predefined(K, A, n, r, SNR, V1, R1)
    SignalIn, SignalOut = maxSINR_comm(K, m, n, r, H, SNR, Vp, Up, Rp, L)
    MSE = compute_MSE(K, SignalIn, SignalOut)

    if BER_needed:
        BER = compute_error(K, m, SignalIn, SignalOut)
        return MSE, BER
    else:
        return MSE


def maxSINR_main_predefined_entropy(K, m, n, r, SNR, H, A, V1, R1, L):

    L1 = L
    L2 = 10000

    Vp, Up, Rp, _ = generate_beamvectors_predefined(K, A, n, r, SNR, V1, R1)

    SignalIn = []
    SignalOut = []
    for _ in range(L1):
        SIn, SOut = maxSINR_comm(K, m, n, r, H, SNR, Vp, Up, Rp, L2)
        SignalIn.append(SIn)
        SignalOut.append(SOut)
    CAP = compute_capacity(K, m, SignalIn, SignalOut, L1)

    return CAP


def maxSINR_main_beam_entropy(K, m, n, r, SNR, H, A, Vp, Rp, Up, L):

    L1 = L
    L2 = 100000

    SignalIn = []
    SignalOut = []
    for _ in range(L1):
        SIn, SOut = maxSINR_comm(K, m, n, r, H, SNR, Vp, Up, Rp, L2)

        SignalIn.append(SIn)
        SignalOut.append(SOut)
    CAP = compute_capacity(K, m, SignalIn, SignalOut, L1)

    return CAP  # , Vp, Up, Rp


def maxSINR_ClosedForm_MI(K, m, n, l, r, SNR, H, A):

    snr = n * (10 ** (SNR / 10))

    Vp, Up, Rp = generate_beamvectors(K, A, n, SNR)

    V = np.transpose(np.array(Vp))[0, :, :]
    U = np.transpose(np.array(Up))[0, :, :]
    R = Rp

    Rate = np.zeros(K)
    for k in range(K):
        B = 0
        for k1 in range(K):
            if k1 != k:
                B += (
                    (snr ** R[k1])
                    * (H[k, k1] ** 2)
                    * np.matmul(V[:, k1 : k1 + 1], np.transpose(V[:, k1 : k1 + 1]))
                )

        B = B + (1 / snr) * np.eye(n)

        a = H[k, k] * np.matmul(np.transpose(U[:, k : k + 1]), V[:, k : k + 1])
        SINR_NUM = (snr ** R[k]) * np.matmul(a, np.transpose(a))
        SINR_DEN = np.matmul(
            np.matmul(np.transpose(U[:, k : k + 1]), B), U[:, k : k + 1]
        )

        Rate[k] = 0.5 * np.log(1 + SINR_NUM / SINR_DEN)

    return np.sum(Rate)


def get_Tx_output(K, m, n, l, r, snr, V, R):

    # Transmitter
    Message = []
    SignalIn = []
    TxOut = []

    for k in range(K):
        if m == "Gaussian":
            M = 0
            X = np.random.normal(0, 1, (1000, 1))
        else:
            M = np.random.randint(0, 2 ** m, size=1000)
            X = PAMMod(M, m, 1)
        VX = X * np.sqrt(snr ** R[k])
        if n != 1:
            VX = np.matmul(VX, np.transpose(V[:, k : k + 1]))

        Message.append(M)
        SignalIn.append(X)
        TxOut.append(VX)

    return SignalIn, TxOut

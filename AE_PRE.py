import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
from comm import PAMMod, PAMDemod
import scipy.io as sio
from MAX_SINR import generate_beamvectors
from AE_PRE_MODEL import (
    model_init,
    update_weights_encoder,
    test_encoder,
    update_weights_decoder,
    update_weights_end,
    get_output,
)
from MAX_SINR import prepare_data
from MAX_SINR import compute_MSE
from sklearn.feature_selection import f_regression, mutual_info_regression

# import entropy_estimators as ee


def save_weights(K, filename, TxNet, RxNet, Normalize, U, V, R, MSE, check):

    Tx = []
    Rx = []
    Norm = []

    for k in range(K):
        Tx.append(TxNet[k].get_weights())
        Rx.append(RxNet[k].get_weights())
        Norm.append(Normalize[k].get_weights())

    if check:
        try:
            Network = pickle.load(open(filename, "rb"))
            MSE_old = Network["MSE"]
            print("Old MSE :", MSE_old[-1])
            if MSE_old[-1] > MSE[-1]:
                NETWORK = {
                    "TX": Tx,
                    "RX": Rx,
                    "NORM": Norm,
                    "MSE": MSE,
                    "V": V,
                    "U": U,
                    "R": R,
                }
                pickle.dump(NETWORK, open(filename, "wb"))
                print("model saved")
            else:
                print("model not saved")
        except:
            NETWORK = {
                "TX": Tx,
                "RX": Rx,
                "NORM": Norm,
                "MSE": MSE,
                "V": V,
                "U": U,
                "R": R,
            }
            pickle.dump(NETWORK, open(filename, "wb"))
            print("new model saved")
    else:
        NETWORK = {
            "TX": Tx,
            "RX": Rx,
            "NORM": Norm,
            "MSE": MSE,
            "V": V,
            "U": U,
            "R": R,
        }
        pickle.dump(NETWORK, open(filename, "wb"))
        print("new model saved")


def ae_load_model(K, filename, m, n, r, l, SNR, H, A):

    snr = 10 ** (np.array(SNR) / 10)
    snr = n * snr

    n_b = 1
    b_s = 1000

    try:
        Network = pickle.load(open(filename, "rb"))
        V = Network["V"]
        U = Network["U"]
        Rp = Network["R"]
    except:
        Vp, Up, Rp = generate_beamvectors(K, A, n, SNR)
        V = np.transpose(np.array(Vp))[0, :, :]
        U = np.transpose(np.array(Up))[0, :, :]

    _, SignalIn, TxOut, Noise = prepare_data(K, m, n, r, V, U, Rp, SNR, b_s)

    TxNet, RxNet, Normalize = model_init(
        K,
        m,
        n,
        l,
        r,
        n_b,
        b_s,
        H,
        SignalIn,
        Noise,
        TxOut,
        snr,
    )
    try:
        Network = pickle.load(open(filename, "rb"))
        for k in range(K):
            TxNet[k].set_weights(Network["TX"][k])
            RxNet[k].set_weights(Network["RX"][k])
            Normalize[k].set_weights(Network["NORM"][k])
    except:
        print("Untrained Model Loaded")
        pass

    return TxNet, RxNet, Normalize, V, U, Rp


def train_save_model(
    K,
    filename,
    filename1,
    m,
    n,
    r,
    l,
    SNR,
    H,
    A,
    n1,
    n2,
    n3,
    save=False,
    check=True,
):

    TxNet, RxNet, Normalize, V, U, R = ae_load_model(
        K, filename1, m, n, r, l, SNR, H, A
    )

    snr = 10 ** (np.array(SNR) / 10)
    snr = n * snr

    b_s = 1000

    _, SignalIn, TxOut, Noise = prepare_data(K, m, n, r, V, U, R, SNR, b_s)

    MSE = []
    for _ in range(n1):
        TxNet, Normalize, mse = update_weights_encoder(
            TxNet, Normalize, K, SignalIn, TxOut, 0.005
        )
        print(mse)
        MSE.append(mse)

    for _ in range(n2):

        RxNet, mse = update_weights_decoder(
            TxNet, Normalize, RxNet, K, SignalIn, H, Noise, 0.001
        )
        # RxNet, mse = update_weights_decoder(
        #     RxNet, K, SignalIn, H, Noise, 0.001, V, R, SNR
        # )
        print(mse)
        MSE.append(mse)

    for _ in range(n3):
        TxNet, RxNet, Normalize, mse = update_weights_end(
            TxNet, RxNet, Normalize, K, SignalIn, H, Noise, 0.001
        )
        print(mse)
        MSE.append(mse)

    if save:
        save_weights(K, filename, TxNet, RxNet, Normalize, U, V, R, MSE, check)

    # # Tx, Rx = get_output(TxNet, RxNet, Normalize, K, SignalIn, Noise, H)

    # MSE = np.zeros(K)
    # for k in range(K):
    #     MSE[k] = np.mean((Rx[k] - SignalIn[k]) ** 2)

    # print(10 * np.log10(MSE))
    # get_mse_ber(K, m, n, r, l, SNR, H, A, filename)


def get_mse_ber(K, m, n, r, l, SNR, H, A, filename):

    TxNet, RxNet, Normalize, V, U, R = ae_load_model(K, filename, m, n, r, l, SNR, H, A)
    L1 = 10
    L2 = 1000

    if m == "Gaussian":
        MSE = np.zeros((L1, K))
    else:
        MSE = np.zeros((L1, K))
        BER = np.zeros((L1, K))
        SER = np.zeros((L1, K))

    for j in range(L1):
        MessageIn, SignalIn, _, Noise = prepare_data(K, m, n, r, V, U, R, SNR, L2)
        # _, Rx = get_output(TxNet, RxNet, Normalize, K, SignalIn, Noise, H, V, U, R, SNR)
        _, Rx = get_output(TxNet, RxNet, Normalize, K, SignalIn, Noise, H)

        for k in range(K):
            if m == "Gaussian":
                MSE[j, k] = np.mean((Rx[k] - SignalIn[k]) ** 2)
            else:
                MSE[j, k] = np.mean((Rx[k] - SignalIn[k]) ** 2)
                msg_det = PAMDemod(np.array(Rx[k]), m, 1).astype(int)
                # SER

                SER[j, k] = np.sum((msg_det != MessageIn[k]).astype(int)) / L2

                # BER

                Mc = np.array(
                    [tuple(int(c) for c in "{:02b}".format(i)) for i in MessageIn[k]]
                )
                M1 = np.array(
                    [tuple(int(c) for c in "{:02b}".format(i)) for i in msg_det]
                )
                BER[j, k] = np.sum(abs(M1 - Mc)) / (m * L2)

    if m == "Gaussian":
        return np.squeeze(np.mean(MSE, axis=0))
    else:
        return (
            np.squeeze(np.mean(MSE, axis=0)),
            np.squeeze(np.mean(BER, axis=0)),
            np.squeeze(np.mean(SER, axis=0)),
        )


def check_noise_dist(K, m, n, r, l, SNR, H, A, filename):

    TxNet, RxNet, Normalize, V, U, R = ae_load_model(K, filename, m, n, r, l, SNR, H, A)
    L1 = 10
    L2 = 1000

    filt_noise = np.zeros(L1 * L2)
    t = 0
    for _ in range(L1):
        _, SignalIn, _, Noise = prepare_data(K, m, n, r, V, U, R, SNR, L2)
        _, Rx = get_output(TxNet, RxNet, Normalize, K, SignalIn, Noise, H)
        for t1 in range(1000):
            filt_noise[t] = np.array(Rx[0] - SignalIn[0])[t1, 0]
            t += 1
    return filt_noise


def get_mi_estimate(K, m, n, r, l, SNR, H, A, filename):

    TxNet, RxNet, Normalize, V, U, R = ae_load_model(K, filename, m, n, r, l, SNR, H, A)
    L1 = 10
    L2 = 1000

    X_data = np.zeros((L1 * L2, K))
    Y_data = np.zeros((L1 * L2, K))

    MI = np.zeros(K)

    for j in range(L1):
        _, SignalIn, _, Noise = prepare_data(K, m, n, r, V, U, R, SNR, L2)
        _, Rx = get_output(TxNet, RxNet, Normalize, K, SignalIn, Noise, H)

        for k in range(K):

            for t1 in range(L2):
                X_data[1000 * j + t1, k] = np.array(SignalIn[k])[t1, 0]
                Y_data[1000 * j + t1, k] = np.array(Rx[k])[t1, 0]

    for k in range(K):

        X = X_data[:, k].reshape(-1, 1)
        Y = np.squeeze(Y_data[:, k])
        M = mutual_info_regression(X, Y)
        MI[k] = M

    return MI


def get_Tx_Output(K, m, n, r, l, SNR, H, A, filename):
    TxNet, RxNet, Normalize, V, _, R = ae_load_model(K, filename, m, n, r, l, SNR, H, A)

    print(V[0])
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

    noise_variance = 1 / (np.sqrt(snr))

    Noise = []
    for k in range(K):
        noise = noise_variance * np.random.randn(TxOut[k].shape[0], TxOut[k].shape[1])
        Noise.append(noise)

    Tx, _ = get_output(TxNet, RxNet, Normalize, K, SignalIn, Noise, H)

    return SignalIn, Tx
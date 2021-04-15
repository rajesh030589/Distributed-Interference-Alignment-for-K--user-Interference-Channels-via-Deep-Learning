import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
from AE_END_MODEL import training_model, predict_data, model_init
from comm import PAMMod, PAMDemod
import scipy.io as sio
import time
from sklearn.feature_selection import f_regression, mutual_info_regression


def prepare_data(K, m, n, l, snr, n_b, b_s):

    # Generate the stage 1 data set samples
    Message = []
    SignalIn = []
    for _ in range(n_b):
        M = []
        X = []
        for _ in range(K):
            if m == "Gaussian":
                msg = 0
                X1 = np.random.normal(0, 1, (b_s * l, 1))
            else:
                msg = np.random.randint(2 ** m, size=b_s * l)
                X1 = PAMMod(msg, m, 1).reshape(b_s, l)
            M.append(msg)
            X.append(X1)
        Message.append(M)
        SignalIn.append(X)

    noise_var = 1 / np.sqrt(snr)
    Noise = []
    for _ in range(n_b):
        N = []
        for _ in range(K):
            N.append(noise_var * np.random.randn(b_s, n * l))
        Noise.append(N)

    return Message, SignalIn, Noise


def ae_train_saved_model(
    K,
    m,
    n,
    r,
    l,
    SNR,
    H,
    A,
    n1,
    dist,
    file_to_save,
    saved_file,
    check,
    new_model,
    save=True,
    log_mse=False,
):

    snr = 10 ** (np.array(SNR) / 10)
    snr = n * snr

    TxNet, RxNet, Normalize = ae_load_model(
        K, saved_file, m, n, r, l, SNR, H, A, new_model
    )

    n_b = 1
    b_s = 1000
    epochs = n1
    lr_init = 0.001

    _, Signal, Noise = prepare_data(K, m, n, l, snr, n_b, b_s)

    TxNet, RxNet, Normalize, MSE = training_model(
        TxNet,
        RxNet,
        Normalize,
        K,
        m,
        n,
        l,
        r,
        n_b,
        b_s,
        epochs,
        lr_init,
        H,
        Signal,
        Noise,
        snr,
        dist,
        log_mse,
    )

    if save:
        save_weights(K, file_to_save, TxNet, RxNet, Normalize, MSE, check)

    return TxNet, RxNet, Normalize


def get_mse_ber(K, m, n, r, l, SNR, H, A, file_name=None):

    snr = 10 ** (np.array(SNR) / 10)
    snr = n * snr

    TxNet, RxNet, Normalize = ae_load_model(K, file_name, m, n, r, l, SNR, H, A)

    if m == "Gaussian":
        MSE = compute_mse_ber(TxNet, RxNet, Normalize, K, m, n, l, H, snr)
        return MSE
    else:
        MSE, BER, SER = compute_mse_ber(TxNet, RxNet, Normalize, K, m, n, l, H, snr)
        return MSE, BER, SER


def get_achievability(K, m, n, r, l, SNR, H, A, file_name=None, dist=False):

    snr = 10 ** (np.array(SNR) / 10)
    snr = n * snr

    TxNet, RxNet, Normalize = ae_load_model(K, file_name, m, n, r, l, SNR, H, A, False)
    I = compute_capacity(TxNet, RxNet, Normalize, K, m, n, l, H, snr)
    return I


def ae_load_model(K, filename, m, n, r, l, SNR, H, A, new_model=False):

    snr = 10 ** (np.array(SNR) / 10)
    # snr = 2 * r * snr
    snr = n * snr

    n_b = 1
    b_s = 1000

    _, Signal, Noise = prepare_data(K, m, n, l, snr, n_b, b_s)

    TxNet, RxNet, Normalize = model_init(
        K,
        m,
        n,
        l,
        r,
        H,
        Signal,
        Noise,
        snr,
    )
    if not new_model:
        try:
            Network = pickle.load(open(filename, "rb"))
            for k in range(K):
                TxNet[k].set_weights(Network["TX"][k])
                RxNet[k].set_weights(Network["RX"][k])
                Normalize[k].set_weights(Network["NORM"][k])
        except:
            print("Untrained Model Loaded")
            pass

    return TxNet, RxNet, Normalize


def save_weights(K, filename, TxNet, RxNet, Normalize, MSE, check=True):

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
                NETWORK = {"TX": Tx, "RX": Rx, "NORM": Norm, "MSE": MSE}
                pickle.dump(NETWORK, open(filename, "wb"))
                print("model saved")
            else:
                print("model not saved")
        except:
            NETWORK = {"TX": Tx, "RX": Rx, "NORM": Norm, "MSE": MSE}
            pickle.dump(NETWORK, open(filename, "wb"))
            print("new model saved")

    else:
        NETWORK = {"TX": Tx, "RX": Rx, "NORM": Norm, "MSE": MSE}
        pickle.dump(NETWORK, open(filename, "wb"))
        print("new model saved")


def compute_mse_ber(TxNet, RxNet, Normalize, K, m, n, l, H, snr):
    L1 = 2
    L2 = 1000

    if m == "Gaussian":
        MSE = np.zeros((L1, K))
    else:
        MSE = np.zeros((L1, K))
        BER = np.zeros((L1, K))
        SER = np.zeros((L1, K))
    for j in range(L1):
        MessageIn, Signal, Noise = prepare_data(K, m, n, l, snr, 1, L2)
        SignalIn, _, Output = predict_data(
            TxNet, RxNet, Normalize, K, Signal[0], H, Noise[0]
        )

        for k in range(K):

            if m == "Gaussian":
                MSE[j, k] = np.mean((Output[k] - Signal[0][k]) ** 2)
            else:
                MSE[j, k] = np.mean((Output[k] - Signal[0][k]) ** 2)

                msg_det = PAMDemod(np.array(Output[k]), m, 1).astype(int)

                # SER
                SER[j, k] = np.sum((msg_det != MessageIn[0][k]).astype(int)) / L2

                # BER

                Mc = np.array(
                    [tuple(int(c) for c in "{:02b}".format(i)) for i in MessageIn[0][k]]
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


def compute_capacity(TxNet, RxNet, Normalize, K, m, n, l, H, snr):

    L = 1000
    C = np.zeros(K)

    Message, Signal, Noise = prepare_data(K, m, n, l, snr, 1, L)
    _, Output = predict_data(TxNet, RxNet, Normalize, K, Signal[0], H, Noise[0])

    for k in range(K):
        msg_det = PAMDemod(np.array(Output[k]), m, 1)
        tau = np.zeros((2 ** m, 2 ** m))
        for i in range(len(msg_det)):
            tau[int(Message[0][k][i]), int(msg_det[i])] += 1

        for i in range(2 ** m):
            tau[i, :] = tau[i, :] / np.sum(tau[i, :])
        C[k] = get_information_capacity(tau)

    return C


def get_information_capacity(tau):

    # Compute H(Y|X)
    tau_n = np.where(tau < 0.00001, 0.00001, tau)
    HYX = -np.mean(np.sum(np.multiply(tau, np.log2(tau_n)), axis=1))

    tau_n1 = np.mean(tau, axis=0)
    tau_n2 = np.where(tau_n1 < 0.00001, 0.00001, tau_n1)
    HY = -np.sum(np.multiply(tau_n1, np.log2(tau_n2)))
    return HY - HYX


def get_output_symbols(K, filename, m, n, r, l, SNR, H, A):

    snr = 10 ** (np.array(SNR) / 10)
    # snr = 2 * r * snr
    snr = 2 * snr

    TxNet, RxNet, Normalize = ae_load_model(K, filename, m, n, r, l, SNR, H, A)

    n_b = 1

    Message = []
    SignalIn = []
    for _ in range(n_b):
        M = []
        X = []
        for _ in range(K):
            msg = 0
            # msg = np.arange(2 ** m)
            X1 = np.random.normal(0, 1, (1000, 1))
            # X1 = PAMMod(msg, m, 1).reshape(2 ** m, l)
            M.append(msg)
            X.append(X1)
        Message.append(M)
        SignalIn.append(X)

    noise_var = 1 / np.sqrt(snr)
    Noise = []

    for _ in range(n_b):
        N = []
        for _ in range(K):
            # N.append(noise_var * np.random.randn(2 ** m, n * l))
            N.append(noise_var * np.random.randn(1000, n * l))
        Noise.append(N)

    TxSignal, RxSignal = predict_data(
        TxNet, RxNet, Normalize, K, SignalIn[0], H, Noise[0]
    )

    # return TxSignal, RxSignal
    return SignalIn, TxSignal, RxSignal


def get_Tx_Output(K, m, n, r, l, SNR, H, A, saved_file):

    snr = 10 ** (np.array(SNR) / 10)
    snr = n * snr

    TxNet, RxNet, Normalize = ae_load_model(K, saved_file, m, n, r, l, SNR, H, A, False)

    MessageIn, Signal, Noise = prepare_data(K, m, n, l, snr, 1, 1000)
    SignalIn, Tx, _ = predict_data(TxNet, RxNet, Normalize, K, Signal[0], H, Noise[0])

    return SignalIn, Tx


def get_mi_estimate(K, m, n, r, l, SNR, H, A, filename):

    snr = 10 ** (np.array(SNR) / 10)
    # snr = 2 * r * snr
    snr = n * snr

    TxNet, RxNet, Normalize = ae_load_model(K, filename, m, n, r, l, SNR, H, A)
    L1 = 10
    L2 = 1000

    X_data = np.zeros((L1 * L2, K))
    Y_data = np.zeros((L1 * L2, K))

    MI = np.zeros(K)

    for j in range(L1):
        MessageIn, Signal, Noise = prepare_data(K, m, n, l, snr, 1, L2)
        SignalIn, _, Rx = predict_data(
            TxNet, RxNet, Normalize, K, Signal[0], H, Noise[0]
        )

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


# def ae_main_test_entropy(K, filename, m, n, r, l, SNR, H, A):

#     snr = 10 ** (np.array(SNR) / 10)
#     # snr = 2 * r * snr
#     snr = 2 * snr

#     n_b = 1
#     b_s = 1000
#     epochs = 10

#     _, Signal, Noise = prepare_data(K, m, n, l, snr, n_b, b_s)
#     TxNet, RxNet, Normalize, _ = training(
#         K,
#         m,
#         n,
#         l,
#         r,
#         n_b,
#         b_s,
#         epochs,
#         H,
#         Signal,
#         Noise,
#         snr,
#     )
#     Network = pickle.load(open(filename, "rb"))
#     for k in range(K):
#         TxNet[k].set_weights(Network["TX"][k])
#         RxNet[k].set_weights(Network["RX"][k])
#         Normalize[k].set_weights(Network["NORM"][k])

#     I = compute_capacity(TxNet, RxNet, Normalize, K, m, n, l, H, snr)
#     return I


# def ae_save_model(K, filename, m, n, r, l, SNR, H, A, dist):

#     saved_model = True
#     snr = 10 ** (np.array(SNR) / 10)
#     # snr = 2 * r * snr
#     snr = 2 * snr

#     n_b = 1
#     b_s = 1000
#     epochs = 1000

#     _, Signal, Noise = prepare_data(K, m, n, l, snr, n_b, b_s)

#     if saved_model:
#         TxNet, RxNet, Normalize, _ = training_model(
#             K,
#             m,
#             n,
#             l,
#             r,
#             n_b,
#             b_s,
#             10,
#             H,
#             Signal,
#             Noise,
#             snr,
#         )
#         Network = pickle.load(open(filename, "rb"))
#         for k in range(K):
#             TxNet[k].set_weights(Network["TX"][k])
#             RxNet[k].set_weights(Network["RX"][k])
#             Normalize[k].set_weights(Network["NORM"][k])
#         training_saved_model(
#             TxNet,
#             RxNet,
#             Normalize,
#             K,
#             m,
#             n,
#             l,
#             r,
#             n_b,
#             b_s,
#             epochs,
#             H,
#             Signal,
#             Noise,
#             snr,
#             dist,
#         )

#     else:
#         TxNet, RxNet, Normalize, MSE = training(
#             K, m, n, l, r, n_b, b_s, epochs, H, Signal, Noise, snr, dist
#         )

#     save_weights(K, filename, TxNet, RxNet, Normalize, MSE)


# def get_output_symbols(K, filename, m, n, r, l, SNR, H, A):
#     snr = 10 ** (np.array(SNR) / 10)
#     # snr = 2 * r * snr
#     snr = 2 * snr

#     epochs = 10
#     n_b = 1

#     Message = []
#     SignalIn = []
#     for _ in range(n_b):
#         M = []
#         X = []
#         for _ in range(K):
#             msg = np.arange(2 ** m)
#             X1 = PAMMod(msg, m, 1).reshape(2 ** m, l)
#             M.append(msg)
#             X.append(X1)
#         Message.append(M)
#         SignalIn.append(X)

#     noise_var = 1 / np.sqrt(snr)
#     Noise = []
#     for _ in range(n_b):
#         N = []
#         for _ in range(K):
#             N.append(noise_var * np.random.randn(2 ** m, n * l))
#         Noise.append(N)

#     TxNet, RxNet, Normalize, _ = training_model(
#         K,
#         m,
#         n,
#         l,
#         r,
#         n_b,
#         2 ** m,
#         epochs,
#         H,
#         SignalIn,
#         Noise,
#         snr,
#     )

#     Network = pickle.load(open(filename, "rb"))
#     for k in range(K):
#         TxNet[k].set_weights(Network["TX"][k])
#         RxNet[k].set_weights(Network["RX"][k])
#         Normalize[k].set_weights(Network["NORM"][k])
#     print("Model loaded with MSE: ", Network["MSE"][-1])
#     TxSignal, RxSignal = predict_data(
#         TxNet, RxNet, Normalize, K, SignalIn[0], H, Noise[0]
#     )

#     return TxSignal, RxSignal


# def get_test_mse_ber(K, m, n, r, l, SNR, H, A, file_name=None, dist=False):

#     snr = 10 ** (np.array(SNR) / 10)
#     snr = n * snr

#     n_b = 1
#     b_s = 1000

#     _, Signal, Noise = prepare_data(K, m, n, l, snr, n_b, b_s)

#     TxNet, RxNet, Normalize = model_init(
#         K,
#         m,
#         n,
#         l,
#         r,
#         H,
#         Signal,
#         Noise,
#         snr,
#     )

#     Network = pickle.load(open(file_name, "rb"))
#     for k in range(K):
#         TxNet[k].set_weights(Network["TX"][k])
#         RxNet[k].set_weights(Network["RX"][k])
#         Normalize[k].set_weights(Network["NORM"][k])
#     print("Train MSE ", Network["MSE"][-1])
#     epochs = 500
#     lr_init = 0.0001

#     _, Signal, Noise = prepare_data(K, m, n, l, snr, n_b, b_s)

#     TxNet, RxNet, Normalize, MSE = training_model(
#         TxNet,
#         RxNet,
#         Normalize,
#         K,
#         m,
#         n,
#         l,
#         r,
#         n_b,
#         b_s,
#         epochs,
#         lr_init,
#         H,
#         Signal,
#         Noise,
#         snr,
#         dist,
#     )
#     # save_weights(K, file_name, TxNet, RxNet, Normalize, MSE, check=False)
#     MSE, BER = compute_mse_ber(TxNet, RxNet, Normalize, K, m, n, l, H, snr)
#     MSE, BER = compute_mse_ber(TxNet, RxNet, Normalize, K, m, n, l, H, snr)

#     MSE, BER = compute_mse_ber(TxNet, RxNet, Normalize, K, m, n, l, H, snr)
#     print(10 * np.log10(MSE))
#     return MSE, BER

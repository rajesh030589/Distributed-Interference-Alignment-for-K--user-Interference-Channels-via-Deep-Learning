from comm import (
    convert_channel,
    get_specific_channel,
    get_filename,
    get_channel_file_name,
    get_perturbed_channel_file_name,
)
import numpy as np
import pickle
from tqdm import tqdm
from AE_TIMTIN_PRE import train_save_model as training_model
import multiprocessing as mp
import math
import itertools
import matplotlib.pyplot as plt
from AE_TIMTIN_PRE import get_mse_ber, check_noise_dist, get_mi_estimate, get_Tx_Output


# np.random.seed(10)


def start_training_model(K, beta, SNR, C, save, check):

    n1 = 500
    n2 = 200
    n3 = 200

    m = "Gaussian"
    n = 2

    c_type = "sym specific"
    strong = 0.9
    weak = 0

    channel_name = get_channel_file_name(K, c_type, strong, weak, beta, C)
    Channel = pickle.load(open(channel_name, "rb"))
    A = Channel["A"]

    if m == "Gaussian":
        r = 1
    else:
        r = m / n

    l = 1

    H = convert_channel(A, K, SNR)
    snr = 2 * (10 ** (SNR / 10))
    file_to_save = get_filename(
        K,
        m,
        n,
        r,
        l,
        snr,
        c_type,
        strong,
        weak,
        beta,
        C,
        "Saved_Models",
        "AE_TIMTIN_PRE",
    )
    saved_file = get_filename(
        K,
        m,
        n,
        r,
        l,
        snr,
        c_type,
        strong,
        weak,
        beta,
        C,
        "Saved_Models",
        "AE_TIMTIN_PRE",
    )
    training_model(
        K,
        file_to_save,
        saved_file,
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
        save,
        check,
    )


def get_mse_ber_results(K, beta, SNR, C):

    m = "Gaussian"
    n = 2

    c_type = "sym specific"
    strong = 0.9
    weak = 0

    if m == "Gaussian":
        r = 1
    else:
        r = m / n
    l = 1

    channel_name = get_channel_file_name(K, c_type, strong, weak, beta, C)
    Channel = pickle.load(open(channel_name, "rb"))
    A = Channel["A"]

    H = convert_channel(A, K, SNR)
    snr = 2 * (10 ** (SNR / 10))
    model_file = get_filename(
        K,
        m,
        n,
        r,
        l,
        snr,
        c_type,
        strong,
        weak,
        beta,
        C,
        "Saved_Models",
        "AE_TIMTIN_PRE",
    )

    if m == "Gaussian":
        MSE_SNR = get_mse_ber(K, m, n, r, l, SNR, H, A, model_file)
    else:
        MSE_SNR, BER_SNR, SER_SNR = get_mse_ber(K, m, n, r, l, SNR, H, A, model_file)
    file_name = get_filename(
        K, m, n, 1, l, SNR, c_type, strong, weak, beta, C, "MSE_Data", "AE_TIMTIN_PRE"
    )

    if m == "Gaussian":
        MSE = {"MSE": MSE_SNR}
        pickle.dump(MSE, open(file_name, "wb"))
    else:

        MSE_BER_SER = {
            "MSE": MSE_SNR,
            "BER": BER_SNR,
            "SER": SER_SNR,
        }
        pickle.dump(MSE_BER_SER, open(file_name, "wb"))

    return MSE_SNR


def get_robust_mse_ber_results(K, beta, SNR, C):

    m = "Gaussian"
    n = 2

    c_type = "sym specific"
    strong = 0.9
    weak = 0

    if m == "Gaussian":
        r = 1
    else:
        r = m / n
    l = 1

    channel_name = get_channel_file_name(K, c_type, strong, weak, beta, C)
    Channel = pickle.load(open(channel_name, "rb"))
    A = Channel["A"]

    H = convert_channel(A, K, SNR)
    snr = 2 * (10 ** (SNR / 10))
    model_file = get_filename(
        K,
        m,
        n,
        r,
        l,
        snr,
        c_type,
        strong,
        weak,
        beta,
        C,
        "Saved_Models",
        "AE_TIMTIN_PRE",
    )

    channel_name = get_perturbed_channel_file_name(K, c_type, strong, weak, beta, C, 0)
    Channel = pickle.load(open(channel_name, "rb"))
    A1 = Channel["A"]

    H1 = convert_channel(A1, K, SNR)

    if m == "Gaussian":
        MSE_SNR = get_mse_ber(K, m, n, r, l, SNR, H1, A1, model_file)
    else:
        MSE_SNR, BER_SNR, SER_SNR = get_mse_ber(K, m, n, r, l, SNR, H, A, model_file)
    file_name = get_filename(
        K,
        m,
        n,
        1,
        l,
        SNR,
        c_type,
        strong,
        weak,
        beta,
        C,
        "Robust_MSE_Data",
        "AE_TIMTIN_PRE",
    )

    if m == "Gaussian":
        MSE = {"MSE": MSE_SNR}
        pickle.dump(MSE, open(file_name, "wb"))
    else:

        MSE_BER_SER = {
            "MSE": MSE_SNR,
            "BER": BER_SNR,
            "SER": SER_SNR,
        }
        pickle.dump(MSE_BER_SER, open(file_name, "wb"))

    return MSE_SNR


def MI_estimator(K, beta, SNR, C):

    m = "Gaussian"
    n = 2

    c_type = "sym specific"
    strong = 0.9
    weak = 0

    if m == "Gaussian":
        r = 1
    else:
        r = m / n
    l = 1

    channel_name = get_channel_file_name(K, c_type, strong, weak, beta, C)
    Channel = pickle.load(open(channel_name, "rb"))
    A = Channel["A"]

    H = convert_channel(A, K, SNR)

    snr = n * (10 ** (SNR / 10))

    filename = get_filename(
        K,
        m,
        n,
        r,
        l,
        snr,
        c_type,
        strong,
        weak,
        beta,
        C,
        "Saved_Models",
        "AE_TIMTIN_PRE",
    )

    MI = get_mi_estimate(K, m, n, r, l, SNR, H, A, filename)

    file_name = get_filename(
        K, m, n, 1, l, SNR, c_type, strong, weak, beta, C, "MI_Data", "AE_TIMTIN_PRE"
    )

    MI_ES = {
        "MI": MI,
    }
    pickle.dump(MI_ES, open(file_name, "wb"))
    return MI


def get_Tx_data(K, beta, SNR, C):

    m = "Gaussian"
    n = 2
    r = 1
    l = 1

    c_type = "sym specific"
    strong = 0.9
    weak = 0

    snr = n * (10 ** (SNR / 10))

    channel_name = get_channel_file_name(K, c_type, strong, weak, beta, C)
    Channel = pickle.load(open(channel_name, "rb"))
    A = Channel["A"]

    H = convert_channel(A, K, SNR)

    filename = get_filename(
        K,
        m,
        n,
        r,
        l,
        snr,
        c_type,
        strong,
        weak,
        beta,
        C,
        "Saved_Models",
        "AE_TIMTIN_PRE",
    )

    X1, VX1 = get_Tx_Output(K, m, n, r, l, SNR, H, A, filename)

    file_name = get_filename(
        K, m, n, r, l, SNR, c_type, strong, weak, beta, C, "Const_Data", "AE_TIMTIN_PRE"
    )

    XY = {"X": X1, "Y": VX1}
    pickle.dump(XY, open(file_name, "wb"))


# K = 5
# beta = 0.5
# C = 0
# start_training_model(K, beta, 50, C, True, False)
# get_mse_ber_results(K, beta, [50])
# plot_noise_dist(K, beta, 50)
# MI_estimator(5, 0.9, [50])
# # # AE_multiprocess_save_model()
# for K in [3, 4, 5]:
#     for beta in [0.5, 0.9]:
#         AE_multiprocess(K, beta, [10, 30, 50])
# # AE_multiprocess_save_model()
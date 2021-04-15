from comm import (
    convert_channel,
    get_specific_channel,
    get_channel_file_name,
    get_filename,
    get_perturbed_channel_file_name,
)
import numpy as np
import pickle
from tqdm import tqdm
from AE_END import ae_train_saved_model as ae_train_save
from AE_END import (
    get_achievability,
    get_mse_ber,
    get_output_symbols,
    get_mi_estimate,
    get_Tx_Output,
)
import multiprocessing as mp
import math
import itertools
import matplotlib.pyplot as plt


def train_save_model(K, beta, SNR, C, dist, log_mse=False, new_model=False):

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

    n1 = 100

    H = convert_channel(A, K, SNR)
    snr = 2 * (10 ** (SNR / 10))

    string = "AE_END"
    if dist:
        string = string + "_DIST"
    if log_mse:
        string = string + "_LOG"

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
        string,
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
        string,
    )
    ae_train_save(
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
        False,
        new_model,
        True,
        log_mse,
    )


def compute_data(K, beta, SNR, C, dist, log_mse):

    m = "Gaussian"
    n = 2
    r = 1

    c_type = "sym specific"
    strong = 0.9
    weak = 0

    l = 1

    channel_name = get_channel_file_name(K, c_type, strong, weak, beta, C)
    Channel = pickle.load(open(channel_name, "rb"))
    A = Channel["A"]

    string = "AE_END"
    if dist:
        string = string + "_DIST"
    if log_mse:
        string = string + "_LOG"

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
        string,
    )
    if m == "Gaussian":
        MSE_SNR = get_mse_ber(K, m, n, r, l, SNR, H, A, model_file)
    else:
        MSE_SNR, BER_SNR, SER_SNR = get_mse_ber(K, m, n, r, l, SNR, H, A, model_file)

    file_name = get_filename(
        K, m, n, r, l, SNR, c_type, strong, weak, beta, C, "MSE_Data", string
    )

    if m == "Gaussian":
        MSE = {
            "MSE": MSE_SNR,
        }
        pickle.dump(MSE, open(file_name, "wb"))
    else:
        MSE_BER_SER = {
            "MSE": MSE_SNR,
            "BER": BER_SNR,
            "SER": SER_SNR,
        }
        pickle.dump(MSE_BER_SER, open(file_name, "wb"))

    return MSE_SNR


def compute_robust_data(K, beta, SNR, C, dist, log_mse):

    m = "Gaussian"
    n = 2
    r = 1

    c_type = "sym specific"
    strong = 0.9
    weak = 0

    l = 1

    channel_name = get_channel_file_name(K, c_type, strong, weak, beta, C)
    Channel = pickle.load(open(channel_name, "rb"))
    A = Channel["A"]

    string = "AE_END"
    if dist:
        string = string + "_DIST"
    if log_mse:
        string = string + "_LOG"

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
        string,
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
        K, m, n, r, l, SNR, c_type, strong, weak, beta, C, "Robust_MSE_Data", string
    )

    if m == "Gaussian":
        MSE = {
            "MSE": MSE_SNR,
        }
        pickle.dump(MSE, open(file_name, "wb"))
    else:
        MSE_BER_SER = {
            "MSE": MSE_SNR,
            "BER": BER_SNR,
            "SER": SER_SNR,
        }
        pickle.dump(MSE_BER_SER, open(file_name, "wb"))

    return MSE_SNR


def MI_estimator(K, beta, SNR, C, dist, log_mse):

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

    string = "AE_END"
    if dist:
        string = string + "_DIST"
    if log_mse:
        string = string + "_LOG"

    channel_name = get_channel_file_name(K, c_type, strong, weak, beta, C)
    Channel = pickle.load(open(channel_name, "rb"))
    A = Channel["A"]

    H = convert_channel(A, K, SNR)

    snr = n * (10 ** (SNR / 10))

    filename = get_filename(
        K, m, n, r, l, snr, c_type, strong, weak, beta, C, "Saved_Models", string
    )

    MI = get_mi_estimate(K, m, n, r, l, SNR, H, A, filename)

    file_name = get_filename(
        K, m, n, 1, l, SNR, c_type, strong, weak, beta, C, "Const_Data", string
    )

    MI_ES = {
        "MI": MI,
    }
    pickle.dump(MI_ES, open(file_name, "wb"))
    return MI


def get_Tx_data(K, beta, SNR, C, dist, log_mse):

    m = "Gaussian"
    n = 2
    r = 1
    l = 1

    c_type = "sym specific"
    strong = 0.9
    weak = 0

    string = "AE_END"
    if dist:
        string = string + "_DIST"
    if log_mse:
        string = string + "_LOG"

    snr = n * (10 ** (SNR / 10))

    channel_name = get_channel_file_name(K, c_type, strong, weak, beta, C)
    Channel = pickle.load(open(channel_name, "rb"))
    A = Channel["A"]

    H = convert_channel(A, K, SNR)

    filename = get_filename(
        K, m, n, r, l, snr, c_type, strong, weak, beta, C, "Saved_Models", string
    )

    X1, VX1 = get_Tx_Output(K, m, n, r, l, SNR, H, A, filename)

    file_name = get_filename(
        K, m, n, r, l, SNR, c_type, strong, weak, beta, C, "Const_Data", string
    )

    XY = {"X": X1, "Y": VX1}
    pickle.dump(XY, open(file_name, "wb"))


def main_function():
    # train_save_model(5, 0.9, [30])
    compute_data(5, 0.9, [30])
    # AE_Constellation()


if __name__ == "__main__":
    main_function()

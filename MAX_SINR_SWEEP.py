import pickle
import numpy as np
from tqdm import tqdm
import math
import multiprocessing as mp
import itertools
from comm import (
    create_channel,
    get_specific_channel,
    convert_channel,
    get_filename,
    get_channel_file_name,
    get_perturbed_channel_file_name,
)
from MAX_SINR import maxSINR_main_beam_entropy as maxsinr_comm_entropy
from MAX_SINR import maxSINR_main_predefined as maxsinr_comm_predefined
from MAX_SINR import maxSINR_main as maxsinr_comm
from MAX_SINR import maxSINR_vectors_given as maxsinr_comm_given
from MAX_SINR import generate_beamvectors
from MAX_SINR import generate_randomvectors, choose_initial_vectors
from MAX_SINR import maxSINR_ClosedForm_MI, get_Tx_output, get_mi_estimate
from MAX_SINR import maxSINR_ClosedForm
from AE_END import get_output_symbols
import matplotlib.pyplot as plt


def compute_mi_estimate(K, beta, SNR, C):

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

    H = convert_channel(A, K, SNR)

    snr = n * (10 ** (SNR / 10))

    filename = get_filename(
        K, m, n, r, l, snr, c_type, strong, weak, beta, C, "Saved_Models", "AE_PRE"
    )

    Network = pickle.load(open(filename, "rb"))
    V = Network["V"]
    U = Network["U"]
    Rp = Network["R"]

    MI = get_mi_estimate(K, m, n, r, SNR, H, V, U, Rp)

    file_name = get_filename(
        K, m, n, 1, l, SNR, c_type, strong, weak, beta, C, "MI_Data", "MaxSINR"
    )

    MI_ES = {
        "MI": MI,
    }
    pickle.dump(MI_ES, open(file_name, "wb"))
    return MI


# def compute_cf_mi_estimate(K, beta, SNR_list):

#     m = "Gaussian"
#     n = 2
#     r = 1

#     c_type = "sym specific"
#     strong = 0.9
#     weak = 0

#     l = 1

#     I = 10

#     Ch = [0]
#     MI = np.zeros((K, I, len(SNR_list)))

#     for i in tqdm(range(I)):
#         A = get_specific_channel(c_type, K, beta, strong, weak)
#         if i in Ch:
#             for j in range(len(SNR_list)):
#                 H = convert_channel(A, K, SNR_list[j])
#                 MI[:, i, j] = maxSINR_ClosedForm_MI(K, m, n, l, r, SNR_list[j], H, A)

#     for j in range(len(SNR_list)):
#         file_name = get_filename(
#             K, m, n, 1, l, SNR_list[j], c_type, strong, weak, beta, "MaxSINR_CF_MI"
#         )

#         mi = np.squeeze(MI[:, :, j])
#         MI_ES = {
#             "MI": mi,
#         }
#         pickle.dump(MI_ES, open(file_name, "wb"))


def compute_data(K, beta, SNR, C):

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

    H = convert_channel(A, K, SNR)

    snr = n * (10 ** (SNR / 10))

    filename = get_filename(
        K, m, n, r, l, snr, c_type, strong, weak, beta, C, "Saved_Models", "AE_PRE"
    )

    Network = pickle.load(open(filename, "rb"))
    V = Network["V"]
    U = Network["U"]
    Rp = Network["R"]

    if m == "Gaussian":
        # MSE_SNR = maxsinr_comm(K, m, n, r, SNR, H, A)
        MSE_SNR = maxSINR_ClosedForm(K, m, n, l, r, SNR, H, V, U, Rp)
        # MSE_SNR = maxSINR_ClosedForm(K, m, n, l, r, SNR, H, A)
        # MSE_SNR = maxsinr_comm_given(K, m, n, r, SNR, H, V, U, Rp)
    else:
        MSE_SNR, BER_SNR, SER_SNR = maxsinr_comm(K, m, n, r, SNR, H, A)

    file_name = get_filename(
        K, m, n, r, l, SNR, c_type, strong, weak, beta, C, "MSE_Data", "MaxSINR"
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


def compute_robust_data(K, beta, SNR, C, C1):

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

    H = convert_channel(A, K, SNR)

    snr = n * (10 ** (SNR / 10))

    filename = get_filename(
        K, m, n, r, l, snr, c_type, strong, weak, beta, C, "Saved_Models", "AE_PRE"
    )

    Network = pickle.load(open(filename, "rb"))
    V = Network["V"]
    U = Network["U"]
    Rp = Network["R"]

    channel_name = get_perturbed_channel_file_name(K, c_type, strong, weak, beta, C, 0)
    Channel = pickle.load(open(channel_name, "rb"))
    A1 = Channel["A"]

    H1 = convert_channel(A1, K, SNR)

    if m == "Gaussian":
        # MSE_SNR = maxsinr_comm(K, m, n, r, SNR, H, A)
        MSE_SNR = maxSINR_ClosedForm(K, m, n, l, r, SNR, H1, V, U, Rp)
        # MSE_SNR = maxSINR_ClosedForm(K, m, n, l, r, SNR, H, A)
        # MSE_SNR = maxsinr_comm_given(K, m, n, r, SNR, H, V, U, Rp)
    else:
        MSE_SNR, BER_SNR, SER_SNR = maxsinr_comm(K, m, n, r, SNR, H, A)

    file_name = get_filename(
        K, m, n, r, l, SNR, c_type, strong, weak, beta, C, "Robust_MSE_Data", "MaxSINR"
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


def get_Tx_data(K, beta, SNR, C):

    m = "Gaussian"
    n = 2
    r = 1
    l = 1

    c_type = "sym specific"
    strong = 0.9
    weak = 0

    snr = n * (10 ** (SNR / 10))

    filename = get_filename(
        K, m, n, r, l, snr, c_type, strong, weak, beta, C, "Saved_Models", "AE_PRE"
    )

    Network = pickle.load(open(filename, "rb"))
    V = Network["V"]
    Rp = Network["R"]

    print(V[0])

    X1, VX1 = get_Tx_output(K, m, n, l, r, SNR, V, Rp)

    file_name = get_filename(
        K, m, n, r, l, SNR, c_type, strong, weak, beta, C, "Const_Data", "MaxSINR"
    )

    XY = {"X": X1, "Y": VX1}
    pickle.dump(XY, open(file_name, "wb"))


# def main_function():
# compute_data(5, 0.9, [40])
# compute_mi_estimate(5, 0.9, [50])
# compute_cf_mi_estimate(5, 0.9, [50])
# plot_scatter_Tx_plot(5, 0.9, 40)


if __name__ == "__main__":
    # specific_case()
    # main_function()
    pass

# def specific_case():
#     K = 3
#     beta = 0.5
#     j0 = 0
#     Ch = 0

#     c_type = "sym specific"
#     strong = 0.9
#     weak = 0
#     r = 1
#     l = 1

#     I = 30
#     n = K - 2
#     m = r * n

#     SNR_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

#     for i in tqdm(range(I)):
#         A = get_specific_channel(c_type, K, beta, strong, weak)
#         Vlist, Rlist = generate_randomvectors(K, n, 100)
#         if i == Ch:
#             SNR = 50
#             V1, R1 = choose_initial_vectors(K, A, n, r, SNR, Vlist, Rlist)
#             for j in range(len(SNR_list)):
#                 if j == j0:
#                     H = convert_channel(A, K, SNR_list[j])
#                     mse, ber, ser = maxsinr_comm_predefined(
#                         K, m, n, r, SNR_list[j], H, A, V1, R1, True
#                     )
#     file = get_filename(
#         K, m, n, r, l, SNR_list[j0], c_type, strong, weak, beta, "MAXSINR"
#     )
#     MSE_BER = pickle.load(open(file, "rb"))
#     MSE_BER["MSE"][Ch] = mse
#     MSE_BER["BER"][Ch] = ber

#     MSE_BER1 = {
#         "MSE": MSE_BER["MSE"],
#         "BER": MSE_BER["BER"],
#     }
#     pickle.dump(MSE_BER1, open(file, "wb"))

# def mp_function():
# K_list = [3]
# m_list = [2, 3, 4, 5, 6]
# r_list = [1, 2]

# paramlist = list(itertools.product(K_list, beta_list))

# with mp.Pool(mp.cpu_count()) as pool:
#     pool.map(maxSINR_multiprocess, paramlist)
# # K_list = [3]

# K_list = [3]
# n_list = [2]
# paramlist = list(itertools.product(m_list, n_list))

# with mp.Pool(mp.cpu_count()) as pool:
#     pool.map(maxSINR_multiprocess_predefined_entropy, paramlist)

# maxSINR_multiprocess_m_entropy(
#     [
#         3,
#         2,
#     ]
# )

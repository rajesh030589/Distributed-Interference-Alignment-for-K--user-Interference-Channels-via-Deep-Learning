import math
import numpy as np
from math import pi
import pickle

# from sympy import *


def generateGrayarr(n):

    arr = list()
    X = np.zeros(2 ** n)
    # start with one-bit pattern
    arr.append("0")
    arr.append("1")

    i = 2
    j = 0
    while True:
        if i >= 1 << n:
            break
        for j in range(i - 1, -1, -1):
            arr.append(arr[j])
        for j in range(i):
            arr[j] = "0" + arr[j]
        for j in range(i, 2 * i):
            arr[j] = "1" + arr[j]
        i = i << 1
    for i in range(len(arr)):
        X[i] = int(arr[i], 2)
    return X


def PSKMod(msg_array, m, n):

    mod_seq = generateGrayarr(m)
    sym_seq1 = [np.math.cos(pi / 4 + ((2 * pi) / (2 ** m)) * i) for i in range(2 ** m)]
    sym_seq2 = [np.math.sin(pi / 4 + ((2 * pi) / (2 ** m)) * i) for i in range(2 ** m)]

    X = np.zeros((len(msg_array), n))
    for i in range(len(msg_array)):
        X[i, 0] = sym_seq1[np.where(mod_seq == msg_array[i])[0][0]]
        X[i, 1] = sym_seq2[np.where(mod_seq == msg_array[i])[0][0]]
    return X


def PSKDemod(RxSig, m, n):

    mod_seq = generateGrayarr(m)
    sym_seq1 = [np.math.cos(pi / 4 + ((2 * pi) / (2 ** m)) * i) for i in range(2 ** m)]
    sym_seq2 = [np.math.sin(pi / 4 + ((2 * pi) / (2 ** m)) * i) for i in range(2 ** m)]
    msg_det = np.zeros(len(RxSig))
    for i in range(len(RxSig)):
        dist = np.zeros(2 ** m)
        for j in range(2 ** m):
            dist[j] = (RxSig[i, 0] - sym_seq1[j]) ** 2 + (
                RxSig[i, 1] - sym_seq2[j]
            ) ** 2
        msg_det[i] = mod_seq[np.argmin(dist)]
    return msg_det


def PAMMod(msg_array, m, n):

    if not (int(m / n) - m / n == 0):
        raise AttributeError("m should be greater than equal to n for baseline")

    mod_seq = generateGrayarr(int(m / n))

    mod_seq1 = np.zeros(len(mod_seq), dtype=int)
    for i in range(len(mod_seq1)):
        mod_seq1[i] = np.where(mod_seq == i)[0][0]
    l = int(m / n)
    odd_seq = np.linspace(2 ** l - 1, -(2 ** l) + 1, 2 ** l)
    A = np.sqrt(3 / ((2 ** l) ** 2 - 1))
    L = len(msg_array)

    if n == 1:
        X2 = np.zeros((L, 1))
        X2[:, 0] = odd_seq[mod_seq1[msg_array]] * A
        return X2

    X1 = np.zeros((L, n))
    for i in range(L):
        msg = msg_array[i]
        for j in range(n):
            b = int(bin(msg)[2:].zfill(m)[j * l : (j + 1) * l], 2)
            X1[i, j] = odd_seq[np.where(mod_seq == b)] * (A / np.sqrt(n))
    return X1


def PAMDemod(RxSig, m, n):
    if not (int(m / n) - m / n == 0):
        raise AttributeError("m should be greater than equal to n for baseline")

    mod_seq = generateGrayarr(int(m / n))
    l = int(m / n)
    odd_seq = np.linspace(2 ** l - 1, -(2 ** l) + 1, 2 ** l)
    A = np.sqrt(3 / ((2 ** l) ** 2 - 1)) / np.sqrt(n)
    odd_seq1 = odd_seq * A

    L = len(RxSig)
    msg_det = np.zeros(L)
    if n == 1:
        ODD_seq = np.matmul(np.ones((L, 1)), odd_seq1.reshape(1, len(odd_seq)))
        MSG_seq = np.matmul(RxSig.reshape(L, 1), np.ones((1, len(odd_seq1))))
        DIST_seq = np.abs(ODD_seq - MSG_seq)

        msg_det1 = mod_seq[np.argmin(DIST_seq, axis=1)]

        return msg_det1
    for i in range(L):
        num = 0
        for j in range(n):
            a = mod_seq[np.argmin(abs(odd_seq * A - RxSig[i, j]))]
            num = num + 2 ** (l * (n - j - 1)) * a
        msg_det[i] = num
    return msg_det


# def capacity_PAM(m, N0):
#     n = 1
#     M = 2 ** m

#     l = int(m / n)
#     odd_seq = np.linspace(2 ** l - 1, -(2 ** l) + 1, 2 ** l)
#     A = np.sqrt(3 / ((2 ** l) ** 2 - 1)) / np.sqrt(n)
#     odd_seq1 = odd_seq * A

#     y, i = symbols("y,i")

#     fy = 0
#     for i in range(M):
#         fy += (
#             (1 / M)
#             * (1 / sqrt(2 * pi * N0))
#             * exp(-((y - odd_seq1[i]) ** 2) / (2 * N0))
#         )
#     integrate(-fy * log(fy), (y, -oo, oo))


def create_channel(channel_type, SNR, K, beta=0.5, strong=0.9, weak=0):

    A = get_specific_channel(channel_type, K, beta, strong, weak)
    H = np.eye(K)
    for i in range(K):
        for j in range(K):
            alpha0 = A[i, j]
            SIR = SNR * (1 - alpha0)
            sir = 10 ** (SIR / 10)
            H[i, j] = np.sqrt(1 / sir)
    return H, A


def convert_channel(A, K, SNR):
    H = np.eye(K)
    for i in range(K):
        for j in range(K):
            alpha0 = A[i, j]
            SIR = SNR * (1 - alpha0)
            sir = 10 ** (SIR / 10)
            H[i, j] = np.sqrt(1 / sir)
    return H


def get_specific_channel(channel_type, K, beta, strong, weak):

    if channel_type == "test_3":
        return np.array(
            [
                [1, 0.9, 0.1],
                [0.1, 1, 0.9],
                [0.1, 0.9, 1],
            ]
        )
    if channel_type == "test_5":
        return np.array(
            [
                [1, 0.9, 0.9, 0.9, 0],
                [0.9, 1, 0.9, 0.9, 0.9],
                [0, 0.9, 1, 0.9, 0.9],
                [0, 0.9, 0.9, 1, 0.9],
                [0, 0, 0.9, 0, 1],
            ]
        )
    if channel_type == "test_4":
        return np.array(
            [
                [1.0, 0.9, 0.9, 0.0],
                [0.9, 1.0, 0.0, 0.0],
                [
                    0.9,
                    0.0,
                    1.0,
                    0.0,
                ],
                [
                    0.9,
                    0.0,
                    0.9,
                    1.0,
                ],
            ]
        )

    A = np.eye(K)

    if channel_type == "sym specific":

        I = np.arange(K * (K - 1))
        np.random.shuffle(I)
        n = int(beta * K * (K - 1))
        Strong = I[:n]

        m = 0
        for i in range(K):
            for j in range(K):
                if i != j:
                    if m in Strong:
                        A[i, j] = strong
                    else:
                        A[i, j] = weak
                    m = m + 1

        return A

    if channel_type == "asym specific":

        I = np.arange(K * (K - 1))
        np.random.shuffle(I)
        n = int(beta * K * (K - 1))
        Strong = I[:n]

        m = 0
        for i in range(K):
            for j in range(K):
                if i != j:
                    if m in Strong:
                        A[i, j] = strong + (1 - strong) * np.random.rand()
                    else:
                        A[i, j] = weak * np.random.rand()
                    m = m + 1

        return A

    if channel_type == "asym":
        for i in range(K):
            for j in range(K):
                if i != j:
                    A[i, j] = strong * np.random.rand()
        return A

    if channel_type == "sym control":

        for i in range(K):
            for j in range(K):
                if i != j:
                    if np.random.rand() > beta:
                        A[i, j] = strong
                    else:
                        A[i, j] = weak
        return A

    if channel_type == "asym control":

        for i in range(K):
            for j in range(K):
                if i != j:
                    if np.random.rand() > beta:
                        A[i, j] = strong + (1 - strong) * np.random.rand()
                    else:
                        A[i, j] = weak * np.random.rand()
        return A

    if channel_type == "sym":
        for i in range(K):
            for j in range(K):
                if i != j:
                    A[i, j] = strong
        return A


def qfunc(x):
    return 0.5 - 0.5 * math.erf(x / np.sqrt(2))


def get_filename(K, m, n, r, l, SNR, c_type, strong, weak, beta, C, string1, string2):
    file1 = string1 + "/" + string2 + "_"
    file2 = (
        str(K)
        + "_"
        + str(m)
        + "_"
        + str(n)
        + "_"
        + str(r)
        + "_"
        + str(l)
        + "_"
        + str(SNR)
        + "_"
        + c_type
        + "_"
        + str(strong)
        + "_"
        + str(weak)
        + "_"
        + str(beta)
        + "_"
        + str(C)
    )
    return file1 + file2 + ".pkl"


def get_channel_file_name(K, c_type, strong, weak, beta, i):

    file1 = "Channels/"
    file2 = (
        str(K)
        + "_"
        + c_type
        + "_"
        + str(strong)
        + "_"
        + str(weak)
        + "_"
        + str(beta)
        + "_"
        + str(i)
    )
    file_name = file1 + file2 + ".pkl"

    return file_name


def get_perturbed_channel_file_name(K, c_type, strong, weak, beta, i, j):

    file1 = "Perturbed_Channels/"
    file2 = (
        str(K)
        + "_"
        + c_type
        + "_"
        + str(strong)
        + "_"
        + str(weak)
        + "_"
        + str(beta)
        + "_"
        + str(i)
        + "_"
        + str(j)
    )
    file_name = file1 + file2 + ".pkl"

    return file_name


def save_channels(K, c_type, strong, weak, beta, C):

    file_name = get_channel_file_name(K, c_type, strong, weak, beta, C)
    A = get_specific_channel(c_type, K, beta, strong, weak)

    CH = {"A": A}
    pickle.dump(CH, open(file_name, "wb"))


def save_perturbed_channel(K, c_type, strong, weak, beta, C, C1):

    channel_name = get_channel_file_name(K, c_type, strong, weak, beta, C)
    Channel = pickle.load(open(channel_name, "rb"))
    A = Channel["A"]

    for i in range(K):
        for j in range(K):
            if i != j:
                A[i, j] = A[i, j] + 0.05 * np.random.rand()

    file_name = get_perturbed_channel_file_name(K, c_type, strong, weak, beta, C, C1)
    A = get_specific_channel(c_type, K, beta, strong, weak)

    CH = {"A": A}
    pickle.dump(CH, open(file_name, "wb"))
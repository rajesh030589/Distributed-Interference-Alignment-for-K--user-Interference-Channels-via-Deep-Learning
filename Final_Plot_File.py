from AE_TIMTIN_PRE_SWEEP import start_training_model as training1
from AE_END_SWEEP import train_save_model as training2
from AE_PRE_SWEEP import start_training_model as training3

from MAX_SINR_SWEEP import compute_data as get1
from AE_END_SWEEP import compute_data as get2
from AE_PRE_SWEEP import get_mse_ber_results as get3
from AE_TIMTIN_PRE_SWEEP import get_mse_ber_results as get4

from comm import get_filename, get_channel_file_name

import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
from pylab import rc

plt.rcParams.update(
    {"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]}
)

rc("axes", linewidth=2)
mpl.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

K = 5


def get_file(beta, C, SNR, string1, string2):
    file_name = get_filename(
        K,
        "Gaussian",
        2,
        1,
        1,
        SNR,
        "sym specific",
        0.9,
        0,
        beta,
        C,
        string1,
        string2,
    )

    return file_name


# 1 Average Plot
Beta = np.linspace(0.5, 1, 6)
plt.figure()
string_list = [
    "MaxSINR",
    "AE_PRE",
    "AE_TIMTIN_PRE",
    "AE_END",
    "AE_END_DIST",
]
string_list1 = [
    r"MaxSINR",
    r"AE PRE",
    r"AE TIMTIN PRE",
    r"AE END",
    r"AE END DIST",
]

marker_list = ["o", "+", "*", ">", "<", "^", "v"]

for i in range(len(string_list)):
    MSE_Data = []
    for beta in Beta:
        M2 = []
        for C in range(3):
            file1 = get_file(beta, C, 50, "MSE_Data", string_list[i])
            MSE = pickle.load(open(file1, "rb"))
            m1 = MSE["MSE"]
            M2.append(np.mean(10 * np.log10(m1)))

        MSE_Data.append(np.mean(M2))
    plt.plot(
        Beta,
        MSE_Data,
        label=string_list1[i],
        marker=marker_list[i],
        markersize=8,
        linewidth=2,
    )
plt.legend()
plt.grid(True)
plt.xlim([0.5, 1])
plt.ylim([-30, 5])
plt.xlabel(r"\textbf{Channel Interference,} $\boldsymbol{\beta}$")
plt.ylabel(r"\textbf{MSE (dB)}")
# plt.savefig("Figures/Final_Avg.png")
# plt.show()


# Scatter Plot
plt.figure()
for i in range(1, len(string_list)):
    for beta in Beta:
        for C in range(3):
            file1 = get_file(beta, C, 50, "MSE_Data", string_list[0])
            MSE = pickle.load(open(file1, "rb"))
            m1 = MSE["MSE"]
            m1 = -np.mean(10 * np.log10(m1))
            file1 = get_file(beta, C, 50, "MSE_Data", string_list[i])
            MSE = pickle.load(open(file1, "rb"))
            m2 = MSE["MSE"]
            m2 = -np.mean(10 * np.log10(m2))

            plt.scatter(m1, m2, s=100 * beta ** 4)
# plt.legend()
plt.plot(np.linspace(-5, 30, 100), np.linspace(-5, 30, 100))
plt.ylabel(r"\textbf{AE, -MSE(dB)} ")
plt.xlabel(r"\textbf{MaxSINR, -MSE(dB)}")
plt.grid(True)
plt.ylim([-5, 30])
plt.xlim([-5, 30])
# plt.savefig("Figures/Final_Scatter.png")
# plt.show()


# MI Plot
plt.figure()
p = np.zeros(6)
for i in range(len(string_list)):
    MSE_Data = []
    t = 0
    for beta in Beta:
        M1 = []
        for C in range(3):
            file1 = get_file(beta, C, 50, "MI_Data", string_list[i])
            MSE = pickle.load(open(file1, "rb"))
            m1 = MSE["MI"]
            M1.append(np.mean(m1))

        if i == 0:
            MSE_Data.append(np.min(M1))
            p[t] = np.argmin(M1)
        else:
            p = p.astype(int)
            MSE_Data.append(M1[p[t]])
        t = t + 1
    plt.plot(
        Beta,
        MSE_Data,
        label=string_list1[i],
        marker=marker_list[i],
        markersize=8,
        linewidth=2,
    )
plt.legend()
plt.grid(True)
# plt.xlim([0.5, 1])
plt.ylim([0, 5])
plt.xlabel(r"\textbf{Channel Interference,} $\boldsymbol{\beta}$")
plt.ylabel(r"\textbf{Mutual Inforamation per User}")
# plt.savefig("Figures/Final_MI.png")
# plt.show()


# Constellation
Beta = np.linspace(0.5, 1, 6)
string_list = [
    "MaxSINR",
    "AE_PRE",
    "AE_TIMTIN_PRE",
]


C = 1
beta = 0.8
channel_name = get_channel_file_name(K, "sym specific", 0.9, 0, beta, C)
Channel = pickle.load(open(channel_name, "rb"))
A = Channel["A"]

fig, ax = plt.subplots(1, 3, sharey=True)


for i in range(3):

    file2 = get_file(beta, C, 50, "Const_Data", string_list[i])
    XY = pickle.load(open(file2, "rb"))
    X = XY["X"]
    Y = XY["Y"]

    for k in range(K):
        ax[i].scatter(X[k], Y[k][:, 0], label="User " + str(k))
    ax[i].grid(True)
    ax[i].set_xlim([-4, 4])
    ax[i].set_ylim([-4, 4])
    ax[i].set_xlabel(r"$W_i$")
    ax[i].set_title(string_list1[i])
    ax[0].set_ylabel(r"$\boldsymbol{X_i}$")
    handles, labels = ax[i].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", mode="expand", ncol=5)
# plt.savefig("Figures/Final_Const1.png")
C = 1
beta = 0.9
channel_name = get_channel_file_name(K, "sym specific", 0.9, 0, beta, C)
Channel = pickle.load(open(channel_name, "rb"))
A = Channel["A"]

fig, ax = plt.subplots(1, 3, sharey=True)


for i in range(3):

    file2 = get_file(beta, C, 50, "Const_Data", string_list[i])
    XY = pickle.load(open(file2, "rb"))
    X = XY["X"]
    Y = XY["Y"]

    for k in range(K):
        ax[i].scatter(X[k], Y[k][:, 0], label="User " + str(k))
    ax[i].grid(True)
    ax[i].set_xlim([-4, 4])
    ax[i].set_ylim([-4, 4])
    ax[i].set_xlabel(r"$W_i$")
    ax[i].set_title(string_list1[i])
    ax[0].set_ylabel(r"$\boldsymbol{X_i}$")
    handles, labels = ax[i].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", mode="expand", ncol=5)
# plt.savefig("Figures/Final_Const.png")

# plt.show()
plt.show()

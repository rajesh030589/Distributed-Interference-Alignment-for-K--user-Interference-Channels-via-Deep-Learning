from AE_TIMTIN_PRE_SWEEP import start_training_model as training1
from AE_END_SWEEP import train_save_model as training2
from AE_PRE_SWEEP import start_training_model as training3

from MAX_SINR_SWEEP import compute_robust_data as get_mse1
from AE_END_SWEEP import compute_robust_data as get_mse2
from AE_PRE_SWEEP import get_robust_mse_ber_results as get_mse3
from AE_TIMTIN_PRE_SWEEP import get_robust_mse_ber_results as get_mse4

from MAX_SINR_SWEEP import compute_mi_estimate as get_mi1
from AE_END_SWEEP import MI_estimator as get_mi2
from AE_PRE_SWEEP import MI_estimator as get_mi3
from AE_TIMTIN_PRE_SWEEP import MI_estimator as get_mi4

from MAX_SINR_SWEEP import get_Tx_data as get_data1
from AE_END_SWEEP import get_Tx_data as get_data2
from AE_PRE_SWEEP import get_Tx_data as get_data3
from AE_TIMTIN_PRE_SWEEP import get_Tx_data as get_data4

from comm import (
    get_filename,
    get_channel_file_name,
    save_channels,
    save_perturbed_channel,
)

from comm import get_filename

import numpy as np
import matplotlib.pyplot as plt
import pickle

gen_channels = False
gen_robust_mse_data = False


K = 5
num_runs = 3

# Save Channels
""" 
In order generate new set of perturbed channels run the generating channels section and comment out rest
of the code.
"""

if gen_channels:
    c_type = "sym specific"
    strong = 0.9
    weak = 0

    for C in range(num_runs):
        for beta in np.linspace(0.5, 1, 6):
            save_perturbed_channel(K, c_type, strong, weak, beta, C, 0)


# Saving MSE data
"""
In order generate MSE Data from saved models run the generate MSE Data section and comment out rest
of the code.
# """


Beta = np.linspace(0.5, 1, 6)

if gen_robust_mse_data:
    for beta in Beta:
        for C in range(num_runs):
            # MaxSINR
            get_mse1(K, beta, 50, C, 0)

            # AE-END
            get_mse2(K, beta, 50, C, False, False)

            # AE-END distributed
            get_mse2(K, beta, 50, C, True, False)

            # Pre train maxsinr
            get_mse3(K, beta, 50, C)

            # Pre train timtin
            get_mse4(K, beta, 50, C)


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


plt.figure()
string_list = [
    "MaxSINR",
    "AE_END",
    "AE_END_DIST",
    "AE_PRE",
    "AE_TIMTIN_PRE",
]

marker_list = ["o", "+", "*", ">", "<", "^", "v"]

for i in range(len(string_list)):
    MSE_Data = []
    for beta in Beta:
        for C in range(3):
            file1 = get_file(beta, C, 50, "Robust_MSE_Data", string_list[i])
            MSE = pickle.load(open(file1, "rb"))
            m1 = MSE["MSE"]
            MSE_Data.append(np.mean(10 * np.log10(m1)))

    plt.plot(MSE_Data, label=string_list[i], marker=marker_list[i])
plt.legend()
plt.grid(True)
plt.ylabel("MSE per User (dB)")
plt.show()

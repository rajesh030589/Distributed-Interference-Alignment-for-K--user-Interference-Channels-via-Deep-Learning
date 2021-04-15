from AE_TIMTIN_PRE_SWEEP import start_training_model as training1
from AE_END_SWEEP import train_save_model as training2
from AE_PRE_SWEEP import start_training_model as training3

from MAX_SINR_SWEEP import compute_data as get_mse1
from AE_END_SWEEP import compute_data as get_mse2
from AE_PRE_SWEEP import get_mse_ber_results as get_mse3
from AE_TIMTIN_PRE_SWEEP import get_mse_ber_results as get_mse4

from MAX_SINR_SWEEP import compute_mi_estimate as get_mi1
from AE_END_SWEEP import MI_estimator as get_mi2
from AE_PRE_SWEEP import MI_estimator as get_mi3
from AE_TIMTIN_PRE_SWEEP import MI_estimator as get_mi4

from MAX_SINR_SWEEP import get_Tx_data as get_data1
from AE_END_SWEEP import get_Tx_data as get_data2
from AE_PRE_SWEEP import get_Tx_data as get_data3
from AE_TIMTIN_PRE_SWEEP import get_Tx_data as get_data4

from comm import get_filename, get_channel_file_name, save_channels

from comm import get_filename

import numpy as np
import matplotlib.pyplot as plt
import pickle

"""

This code generates the required data and models saves them and which are later used to plot the figures.
"""

K = 5
num_runs = 3

# Save Channels
""" 
In order generate new set of channels run the generating channels section and comment out rest
of the code.
"""
c_type = "sym specific"
strong = 0.9
weak = 0


# CAUTION: If channels are generated then entire process of saving models
# and generating new data has to be repeated.
gen_channel = False
gen_model = False
gen_mse_data = False
gen_mi_data = False
gen_tx_data = False

if gen_channel:
    for beta in np.linspace(0.5, 1, 6, endpoint=True):
        for C in range(num_runs):
            save_channels(K, c_type, strong, weak, 0, num_runs)


# Generating Models
""" 
In order generate new models run the generating models section and comment out rest
of the code.
"""
if gen_model:
    for C in range(num_runs):
        for beta in np.linspace(0.5, 1, 6):

            # AE Pretraining with TIMTIN
            training1(K, beta, 50, C, True, False)

            # AE End to End Joint
            training2(K, beta, 50, C, True, True, True)

            # AE End to End Distributive
            training2(K, beta, 50, C, True)

            # AE Pretrining to MaxSINR
            training3(K, beta, 50, C, True, False)

# Saving MSE data
""" 
In order generate MSE Data from saved models run the generate MSE Data section and comment out rest
of the code.
"""

Beta = np.linspace(0.5, 1, 6)
if gen_mse_data:
    for beta in Beta:
        for C in range(num_runs):
            # MaxSINR
            get_mse1(K, beta, 0, C)

            # AE-END
            get_mse2(K, beta, 50, C, False, False)

            # AE-END distributed
            get_mse2(K, beta, 50, C, True, False)

            # Pre train maxsinr
            get_mse3(K, beta, 50, C)

            # Pre train timtin
            get_mse4(K, beta, 50, C)

# Saving MI Data
""" 
In order generate MI Data from saved models run the generate MI Data section and comment out rest
of the code.
"""
Beta = np.linspace(0.5, 1, 6)

if gen_mi_data:
    for beta in Beta:
        for C in range(num_runs):

            # MaxSINR
            get_mi1(K, beta, 50, C)

            # AE-END
            get_mi2(K, beta, 50, C, False, False)

            # AE-END
            get_mi2(K, beta, 50, C, True, False)

            # Pre train maxsinr
            get_mi3(K, beta, 50, C)

            # Pre train timtin
            get_mi4(K, beta, 50, C)

# Constellation Data
""" 
In order to generate TX Output Data from saved models run the generate TX Data section and comment out rest
of the code.
"""

if gen_tx_data:
    beta_val = 0.8

    # AE-END
    get_data2(K, beta_val, 50, 0, False, False)

    # AE-END-DIST
    get_data2(K, beta_val, 50, 0, True, False)

    # Pre train maxsinr
    get_data3(K, beta_val, 50, 0)

    # Pre train timtin
    get_data4(K, beta_val, 50, 0)

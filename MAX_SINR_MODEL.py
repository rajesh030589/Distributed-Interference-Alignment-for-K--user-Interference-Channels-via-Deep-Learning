from scipy.io import loadmat
from scipy.linalg import null_space
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io as sio


class transmitter:
    def __init__(self, channel_use, N_users, n_streams):
        self.n = channel_use
        self.K = N_users
        self.m = n_streams

    def random_beam(self):
        # Initial random power allocation and beam vector selection
        V = []
        R = []
        for _ in range(self.K):
            R.append(-np.random.rand(self.m))
            Vtemp = np.zeros((self.n, self.m))

            for m in range(self.m):
                v = np.random.rand(self.n)
                e = np.linalg.norm(v)
                Vtemp[:, m] = v / e
            V.append(Vtemp)

        return V, R


class receiver:
    def __init__(self, channel_use, N_users, n_streams):

        self.n = channel_use
        self.K = N_users
        self.m = n_streams

    def compute_covariance(self, P):

        QD = []
        QI = []

        for rx_k in range(self.K):
            QD_t = []
            QI_t = []
            for m in range(self.m):
                a = P ** self.DesP[rx_k][m]
                b = np.matmul(self.DesV[rx_k][m], np.transpose(self.DesV[rx_k][m]))
                QD_t.append(a * b)

                QItemp = np.zeros((self.n, self.n))
                for i in range(len(self.IntP[rx_k][m])):
                    a = P ** self.IntP[rx_k][m][i]
                    b = np.matmul(
                        self.IntV[rx_k][m][i], np.transpose(self.IntV[rx_k][m][i])
                    )
                    QItemp = QItemp + a * b

                # Interference Covariance matrix
                QI_t.append(QItemp)

            QD.append(QD_t)
            QI.append(QI_t)

        return QD, QI

    """
    This function determines the recieve beam former and the power allocation
    for the reciprocal channel for a user receiver
    """

    def compute_recieve_beamvector(self, DesP, DesV, IntP, IntV, A, QI):

        R = np.zeros(self.m)
        U = np.zeros((self.n, self.m))

        for m in range(self.m):

            IntP_mod = IntP[m].copy()

            if IntP_mod:

                # Power Allocation Vectors

                MaxIntP = max(IntP_mod)
                R[m] = min(0, -max(MaxIntP))

                # Receive Beam formers
                Inverse = np.linalg.inv(QI[m] + np.eye(self.n))
                Num = np.matmul(Inverse, np.array(DesV[m]))
                Den = np.linalg.norm(Num)

                U[:, m : m + 1] = Num / Den

        return U, R

    """
    This function consolidates the received beamformers for all users
    """

    def equalize(self, A, P):

        U = []
        R = []

        _, QI = self.compute_covariance(P)
        for i in range(self.K):

            u, r = self.compute_recieve_beamvector(
                self.DesP[i], self.DesV[i], self.IntP[i], self.IntV[i], A, QI[i]
            )
            U.append(u)
            R.append(r)
        return U, R

    """
    This function finds the interfering beams and the interfering powers
    """

    def compute_recieve(self, V, R, A, P):

        # Collect the desired beam direction for each of the streams
        self.DesV = []
        self.DesP = []
        for rx_k in range(self.K):
            DesV_t = []
            DesP_t = []
            for m in range(self.m):
                DesV_t.append(V[rx_k][:, m : m + 1])
                DesP_t.append(R[rx_k][m] + A[rx_k][rx_k])
            self.DesV.append(DesV_t)
            self.DesP.append(DesP_t)

        # Collect the interference vectors for each of the desired beam
        # The desired streams are decoded and removed in lexicograhic sequence

        self.IntV = []
        self.IntP = []
        for rx_k in range(self.K):
            IntP_t = []
            IntV_t = []
            for m in range(self.m):
                IntP_t1 = []
                IntV_t1 = []

                for tx_k in range(self.K):
                    if rx_k == tx_k:
                        if self.m > 1:
                            for m1 in range(m + 1, self.m):
                                IntP_t1.append(R[tx_k][m1 : m1 + 1] + A[rx_k][tx_k])
                                IntV_t1.append(V[tx_k][:, m1 : m1 + 1])
                    else:
                        for m1 in range(self.m):
                            IntP_t1.append(R[tx_k][m1 : m1 + 1] + A[rx_k][tx_k])
                            IntV_t1.append(V[tx_k][:, m1 : m1 + 1])
                IntP_t.append(IntP_t1)
                IntV_t.append(IntV_t1)
            self.IntP.append(IntP_t)
            self.IntV.append(IntV_t)

        U, R = self.equalize(A, P)

        return U, R

    def sum_rate(self, U, P):

        QD, QI = self.compute_covariance(P)
        C = []
        for rx_k in range(self.K):
            for m in range(self.m):
                a1 = np.transpose(U[rx_k][:, m : m + 1])
                b1 = QD[rx_k][m]
                c1 = U[rx_k][:, m : m + 1]
                d1 = np.matmul(np.matmul(a1, b1), c1)

                if QI:
                    a2 = np.transpose(U[rx_k][:, m : m + 1])
                    b2 = QI[rx_k][m]
                    c2 = U[rx_k][:, m : m + 1]

                    d2 = 1 + np.matmul(np.matmul(a2, b2), c2)
                else:
                    d2 = 1

                C.append(np.log2(1 + d1 / d2))
        return sum(C) / self.n


def Distributed_maxSINR(n, m, K, A, P):

    # Tx Side
    Tx1 = transmitter(n, K, m)
    Rx1 = receiver(n, K, m)

    # Rx Side
    Rx2 = receiver(n, K, m)
    R_max = 0

    for _ in range(200):

        err_th = 1e-10
        err = 1

        # SNR
        R_old = 0
        count = 0

        [V1, R1] = Tx1.random_beam()

        while err > err_th and count < 10:
            count = count + 1

            # Compute the receive vectors
            [U2, R2] = Rx2.compute_recieve(V1, R1, A, P)

            R = Rx2.sum_rate(U2, P)  # Compute the sum rate

            # Compute the tx beam vectors for the reciprocal channel
            V2 = U2.copy()

            Ar = np.transpose(A)

            # Compute the receive vectors
            [U1, R1] = Rx1.compute_recieve(V2, R2, Ar, P)

            # Compute the beam vectors for the reciprocal channel
            V1 = U1.copy()

            err = abs(R - R_old)

            R_old = R

            if R > R_max:
                R_max = R
                R1_max = R1
                V1_max = V1
                U2_max = U2

    return V1_max, U2_max, R1_max


def Distributed_maxSINR_predfined(n, m, K, A, P, initV, initR):

    # randV : Pre-decided random beam vector
    # randR : Pre-decided random beam vector

    # Tx Side
    Rx1 = receiver(n, K, m)

    # Rx Side
    Rx2 = receiver(n, K, m)

    err_th = 1e-10
    err = 1

    # SNR
    R_old = 0
    count = 0

    V1 = initV.copy()
    R1 = initR.copy()

    while err > err_th and count < 10:
        count = count + 1

        # Compute the receive vectors
        [U2, R2] = Rx2.compute_recieve(V1, R1, A, P)

        R = Rx2.sum_rate(U2, P)  # Compute the sum rate

        # Compute the tx beam vectors for the reciprocal channel
        V2 = U2.copy()

        Ar = np.transpose(A)

        # Compute the receive vectors
        [U1, R1] = Rx1.compute_recieve(V2, R2, Ar, P)

        # Compute the beam vectors for the reciprocal channel
        V1 = U1.copy()

        err = abs(R - R_old)

        R_old = R

    return V1, U2, R1, R


def Distributed_maxSINR_choose(n, m, K, A, P, Vlist, Rlist):

    R_max = 0
    choice_t = 0
    for t in range(len(Vlist)):

        V1 = Vlist[t].copy()
        R1 = Rlist[t].copy()

        _, _, _, Rate = Distributed_maxSINR_predfined(n, m, K, A, P, V1, R1)

        if Rate > R_max:
            R_max = Rate
            choice_t = t

    return Vlist[choice_t], Rlist[choice_t]

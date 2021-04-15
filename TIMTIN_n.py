from scipy.io import loadmat
from scipy.linalg import null_space
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io as sio


def convert_reciprocal(A, K):
    Ar = []
    for k_rx in range(K):
        Art = []
        for k_tx in range(K):
            Art.append(A[k_tx][k_rx])
        Ar.append(Art)
    return Ar


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

    def intf_subspace(self, IntP, IntV):

        n_dim = min(self.K - 1, self.n - 1)
        MaxIntP = np.zeros(n_dim)
        MaxIntV = np.zeros((self.n, self.n - 1))
        MaxIntIdx = []
        for i in range(n_dim):
            maxIntidx = np.argmax(IntP)
            maxIntP = IntP[maxIntidx]
            MaxIntIdx.append(maxIntidx)
            IntP[maxIntidx] = -100
            MaxIntP[i] = maxIntP
            MaxIntV[:, i : i + 1] = IntV[maxIntidx]

        temp = np.zeros((self.n, 1))
        for i in range(n_dim, self.n - 1):
            temp[i + 1, 0] = 1
            MaxIntV[:, i : i + 1] = temp

        return MaxIntP, MaxIntV, MaxIntIdx[0]

        #     MaxIntIdx.append(np.argmax(IntP))
        # MaxIntP = np.zeros(self.n - 1)
        # MaxIntV = np.zeros((self.n, self.n - 1))
        # for i in range(len(MaxIntIdx)):
        #     MaxIntP[i] = IntP[MaxIntIdx[i]]
        #     MaxIntV[:, i : i + 1] = IntV[MaxIntIdx[i]]

        # return MaxIntP, MaxIntV, MaxIntIdx[0]

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

    def compute_recieve_beamvector(self, DesP, DesV, IntP, IntV, A):

        R = np.zeros(self.m)
        U = np.zeros((self.n, self.m))
        for m in range(self.m):

            IntP_mod = IntP[m].copy()
            IntV_mod = IntV[m].copy()

            """
                *) If the strongest interference is in the same direction as the
                desired beam direction and the desired power is high,
                    - the ZF direction would be null space for the remaining interferers
                    - the power allocation for the reciprocal beam would be maz interference received
                *) else
                    - the ZF direction would be the null space for the remaining interferers
                    - the power allocation would be the the maximum residual interference after zeroforcing
                """
            if IntP_mod:

                MaxIntP, MaxIntV, MaxIntIdx = self.intf_subspace(IntP_mod, IntV_mod)

                if (
                    np.max(
                        [
                            np.dot(np.transpose(DesV[m]), np.transpose(MaxIntV[:, i]))
                            for i in range(MaxIntV.shape[1])
                        ]
                    )
                    < 1e-5
                    and max(MaxIntP) < DesP[m]
                ):

                    # R[m] = 0
                    R[m] = min(0, -max(MaxIntP))
                    IntP_mod[MaxIntIdx] = -1

                    _, MaxIntV, _ = self.intf_subspace(IntP_mod, IntV_mod)
                    # to be modified and checked

                    p = null_space(np.transpose(MaxIntV))
                    if p.shape[1] > 1:
                        U[:, m : m + 1] = p[:, 0:1]
                    else:
                        U[:, m : m + 1] = p
                else:
                    p = null_space(np.transpose(MaxIntV))
                    if p.shape[1] > 1:
                        U[:, m : m + 1] = p[:, 0:1]
                    else:
                        U[:, m : m + 1] = p
                    v1 = np.matmul(
                        np.transpose(U[:, m]),
                        np.transpose(np.asmatrix(np.array(IntV_mod))),
                    )

                    x = []
                    for l in range(len(IntP_mod)):
                        if abs(v1[0, l]) > 1e-5:
                            x.append(IntP_mod[l])
                    if x:
                        MaxIntP = max(x)
                        R[m] = min(0, -MaxIntP)
                        # R[m] = 0
                    else:
                        continue
            else:
                continue

        return U, R

    """
    This function consolidates the received beamformers for all users
    """

    def equalize(self, A):

        U = []
        R = []
        for i in range(self.K):
            u, r = self.compute_recieve_beamvector(
                self.DesP[i], self.DesV[i], self.IntP[i], self.IntV[i], A
            )
            U.append(u)
            R.append(r)
        return U, R

    """
    This function finds the interfering beams and the interfering powers
    """

    def compute_recieve(self, V, R, A):

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

        U, R = self.equalize(A)

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


def Distributed_timtin(n, m, K, A, P):

    # Tx Side
    Tx1 = transmitter(n, K, m)
    Rx1 = receiver(n, K, m)

    # Rx Side
    Rx2 = receiver(n, K, m)
    R_max = 0

    for _ in range(50):

        err_th = 1e-10
        err = 1

        # SNR
        R_old = 0
        count = 0

        [V1, R1] = Tx1.random_beam()

        while err > err_th and count < 10:
            count = count + 1

            # Compute the receive vectors
            [U2, R2] = Rx2.compute_recieve(V1, R1, A)

            R = Rx2.sum_rate(U2, P)  # Compute the sum rate

            # Compute the tx beam vectors for the reciprocal channel
            V2 = U2.copy()

            Ar = convert_reciprocal(A, K)  # Reciprocal channel

            # Compute the receive vectors
            [U1, R1] = Rx1.compute_recieve(V2, R2, Ar)

            # Compute the beam vectors for the reciprocal channel
            V1 = U1.copy()

            err = abs(R - R_old)

            R_old = R

            if R > R_max:
                R_max = R
                R1_max = R1
                V1_max = V1
                U2_max = U2
                # print(R_max)

    return V1_max, U2_max, R1_max


def Distributed_timtin_choose(n, m, K, A, P, Vlist, Rlist):

    # Tx Side
    Rx1 = receiver(n, K, m)

    # Rx Side
    Rx2 = receiver(n, K, m)
    R_max = 0
    for t in range(len(Vlist)):

        err_th = 1e-10
        err = 1

        # SNR
        R_old = 0
        count = 0

        V1 = Vlist[t].copy()
        R1 = Rlist[t].copy()

        while err > err_th and count < 10:
            count = count + 1

            # Compute the receive vectors
            [U2, R2] = Rx2.compute_recieve(V1, R1, A)

            R = Rx2.sum_rate(U2, P)  # Compute the sum rate

            # Compute the tx beam vectors for the reciprocal channel
            V2 = U2.copy()

            Ar = convert_reciprocal(A, K)  # Reciprocal channel

            # Compute the receive vectors
            [U1, R1] = Rx1.compute_recieve(V2, R2, Ar)

            # Compute the beam vectors for the reciprocal channel
            V1 = U1.copy()

            err = abs(R - R_old)

            R_old = R

            if R > R_max:
                R_max = R
                choice_t = t

    return Vlist[choice_t], Rlist[choice_t]


def Distributed_timtin_predefined(n, m, K, A, P, initV, initR):

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
        [U2, R2] = Rx2.compute_recieve(V1, R1, A)

        R = Rx2.sum_rate(U2, P)  # Compute the sum rate

        # Compute the tx beam vectors for the reciprocal channel
        V2 = U2.copy()

        Ar = convert_reciprocal(A, K)  # Reciprocal channel

        # Compute the receive vectors
        [U1, R1] = Rx1.compute_recieve(V2, R2, Ar)

        # Compute the beam vectors for the reciprocal channel
        V1 = U1.copy()

        err = abs(R - R_old)

        R_old = R

    return V1, U2, R1

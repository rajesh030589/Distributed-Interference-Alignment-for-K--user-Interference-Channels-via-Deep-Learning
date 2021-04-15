import numpy as np
import tensorflow as tf
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx("float64")


class Normalizer(tf.keras.Model):
    def __init__(self, n, l):
        super(Normalizer, self).__init__()

        # n : length of each beam vector
        # l : no of beam vectors

        self.l = l * tf.constant(1.0, dtype=tf.float64)
        self.scale = tf.Variable(
            tf.ones(shape=(n * l,), dtype=tf.float64),
            dtype=tf.float64,
            trainable=True,
        )
        self.power_scale = tf.Variable(
            tf.constant(1.0, dtype=tf.float64),
            dtype=tf.float64,
            trainable=True,
            constraint=lambda x: tf.clip_by_value(x, 0, 1),
        )

    def call(self, inputs):
        inputs = self.normalize(inputs)
        scalar = (
            tf.multiply(inputs, self.scale)
            / tf.math.sqrt(tf.math.reduce_sum(tf.math.square(self.scale)))
        ) * tf.math.sqrt(self.l)
        return tf.multiply(scalar, self.power_scale)

    def normalize(self, x):

        mu_x, sig_x = tf.nn.moments(x, 0)
        x = (x - mu_x) / tf.math.sqrt(sig_x)
        return x


class Tx_System(tf.keras.Model):
    def __init__(self, n, l, Nodes):
        super(Tx_System, self).__init__()
        initializer = tf.keras.initializers.GlorotNormal()

        self.tx1 = tf.keras.layers.Dense(
            units=Nodes,
            activation="relu",
            use_bias=True,
            kernel_initializer=initializer,
            bias_initializer=initializer,
        )
        self.tx2 = tf.keras.layers.Dense(
            units=Nodes,
            use_bias=True,
            kernel_initializer=initializer,
            bias_initializer=initializer,
        )
        self.tx3 = tf.keras.layers.Dense(
            units=n * l, use_bias=False, kernel_initializer=initializer
        )

    def transmit(self, x):
        x = self.tx1(x)
        x = self.tx2(x)
        x = self.tx3(x)
        return x


class Rx_System(tf.keras.Model):
    def __init__(self, l, Nodes):
        super(Rx_System, self).__init__()
        initializer = tf.keras.initializers.GlorotNormal()

        self.rx1 = tf.keras.layers.Dense(
            units=Nodes,
            use_bias=True,
            activation="relu",
            kernel_initializer=initializer,
            bias_initializer=initializer,
        )
        self.rx2 = tf.keras.layers.Dense(
            units=Nodes,
            use_bias=True,
            kernel_initializer=initializer,
            bias_initializer=initializer,
        )
        self.rx3 = tf.keras.layers.Dense(
            units=l,
            use_bias=False,
            activation=None,
        )

    def receive(self, y):
        y = self.rx1(y)
        y = self.rx2(y)
        y = self.rx3(y)
        return y


def test_encoder(TxNet, Normalize, K, X, Y):

    Loss = []
    for k in range(K):
        TxSig = Normalize[k](TxNet[k].transmit(X[k]))
        l = tf.keras.losses.mse(TxSig, Y[k])
        Loss.append(l)

    return np.mean(Loss)


def update_weights_encoder(TxNet, Normalize, K, X, Y, lr_rate):

    Loss = []
    for k in range(K):
        with tf.GradientTape(persistent=True) as tape:
            TxSig = Normalize[k](TxNet[k].transmit(X[k]))
            l = tf.keras.losses.mse(TxSig, Y[k])
            Loss.append(l)
        Gradients1 = tape.gradient(l, TxNet[k].trainable_variables)
        Gradients2 = tape.gradient(l, Normalize[k].trainable_variables)
        optimizer = tf.keras.optimizers.Adam(lr_rate)
        optimizer.apply_gradients((zip(Gradients1, TxNet[k].trainable_variables)))
        optimizer.apply_gradients((zip(Gradients2, Normalize[k].trainable_variables)))

    return TxNet, Normalize, np.mean(Loss)


# def update_weights_decoder(RxNet, K, X, Y, lr_rate):

#     Loss = []
#     for k in range(K):
#         with tf.GradientTape() as tape:
#             RxOut = RxNet[k].receive(X[k])
#             l = tf.keras.losses.mse(RxOut[k], Y[k])
#             Loss.append(l)
#         Gradients = tape.gradient(l, RxNet[k].trainable_variables)
#         optimizer = tf.keras.optimizers.Adam(lr_rate)
#         optimizer.apply_gradients((zip(Gradients, RxNet[k].trainable_variables)))

#     return RxNet, np.mean(Loss)


def test_decoder(TxNet, Normalize, RxNet, K, X, Y):

    Loss = []
    for k in range(K):
        RxSig = Normalize[k](TxNet[k].transmit(X[k]))
        RxOut = RxNet[k].receive(RxSig)
        l = tf.keras.losses.mse(RxOut[k], Y[k])
        Loss.append(l)

    return np.mean(Loss)


# def update_weights_decoder(TxNet, Normalize, RxNet, K, X, H, Noise, lr_rate, V, R, SNR):

#     snr = 2 * (10 ** (SNR / 10))
#     # TxSignal = []
#     TxSignal1 = []
#     for k in range(K):
#         # TxSignal.append(Normalize[k](TxNet[k].transmit(X[k])))
#         TxSignal1.append(
#             np.matmul(X[k] * np.sqrt(snr ** R[k]), np.transpose(V[:, k : k + 1]))
#         )

#     Loss = []
#     for k in range(K):
#         with tf.GradientTape() as tape:
#             RxSig = 0
#             for k1 in range(K):
#                 RxSig = RxSig + TxSignal1[k1] * H[k, k1]
#             RxSig = RxSig + Noise[k]
#             Y = RxNet[k].receive(RxSig)
#             l = tf.keras.losses.mse(X[k], Y)
#             Loss.append(l)
#         Gradients = tape.gradient(l, RxNet[k].trainable_variables)
#         optimizer = tf.keras.optimizers.Adam(lr_rate)
#         optimizer.apply_gradients((zip(Gradients, RxNet[k].trainable_variables)))

#     return RxNet, np.mean(Loss)


def update_weights_decoder(TxNet, Normalize, RxNet, K, X, H, Noise, lr_rate):

    Loss = []
    for k in range(K):
        with tf.GradientTape() as tape:
            RxSig = 0
            for k1 in range(K):
                RxSig = RxSig + Normalize[k1](TxNet[k1].transmit(X[k1])) * H[k, k1]
            RxSig = RxSig + Noise[k]
            Y = RxNet[k].receive(RxSig)
            l = tf.keras.losses.mse(X[k], Y)
            Loss.append(l)
        Gradients = tape.gradient(l, RxNet[k].trainable_variables)
        optimizer = tf.keras.optimizers.Adam(lr_rate)
        optimizer.apply_gradients((zip(Gradients, RxNet[k].trainable_variables)))

    return RxNet, np.mean(Loss)


def update_weights_end(TxNet, RxNet, Normalize, K, X, H, Noise, lr_rate):

    Loss = []
    with tf.GradientTape(persistent=True) as tape:
        TxSignal = []
        for k in range(K):
            TxSignal.append(Normalize[k](TxNet[k].transmit(X[k])))

        Loss = []
        for k in range(K):
            RxSig = 0
            for k1 in range(K):
                RxSig = RxSig + TxSignal[k1] * H[k, k1]
            RxSig = RxSig + Noise[k]
            Y = RxNet[k].receive(RxSig)
            l = tf.keras.losses.mse(X[k], Y)
            # l = tf.reduce_mean(tf.keras.losses.mse(X[k], Y))
            # l = (10 / 2.303) * tf.math.log(l)
            Loss.append(l)

    for k in range(K):
        loss = Loss[k]
        Gradients1 = tape.gradient(loss, TxNet[k].trainable_variables)
        Gradients2 = tape.gradient(loss, Normalize[k].trainable_variables)
        Gradients3 = tape.gradient(loss, RxNet[k].trainable_variables)
        optimizer = tf.keras.optimizers.Adam(lr_rate)
        optimizer.apply_gradients((zip(Gradients1, TxNet[k].trainable_variables)))
        optimizer.apply_gradients((zip(Gradients2, Normalize[k].trainable_variables)))
        optimizer.apply_gradients((zip(Gradients3, RxNet[k].trainable_variables)))

    return TxNet, RxNet, Normalize, np.mean(Loss)


def model_init(
    K,
    m,
    n,
    l,
    r,
    n_b,
    b_s,
    H,
    Signal,
    Noise,
    TxOut,
    snr,
):

    # Network Model
    TxNet = []
    Normalize = []
    RxNet = []
    for _ in range(K):
        TxNet.append(Tx_System(n, l, 32))
        RxNet.append(Rx_System(l, 32))
        Normalize.append(Normalizer(n, l))

    # Stage 1 Training
    lr_rate = 0.001
    TxNet, Normalize, _ = update_weights_encoder(
        TxNet, Normalize, K, Signal, TxOut, lr_rate
    )

    # Stage 2 Training
    RxNet, _ = update_weights_decoder(
        TxNet, Normalize, RxNet, K, Signal, H, Noise, lr_rate
    )
    return TxNet, RxNet, Normalize


def get_output(TxNet, RxNet, Normalize, K, X, Noise, H):

    TxSignal = []
    for k in range(K):
        TxSignal.append(Normalize[k](TxNet[k].transmit(X[k])))

    RxSignal = []
    for k in range(K):
        RxSig = 0
        for k1 in range(K):
            RxSig = RxSig + TxSignal[k1] * H[k, k1]
        RxSig = RxSig + Noise[k]
        Y = RxNet[k].receive(RxSig)
        RxSignal.append(Y)

    return TxSignal, RxSignal


# def get_output(TxNet, RxNet, Normalize, K, X, Noise, H, V, U, R, SNR):
# def get_output(TxNet, RxNet, Normalize, K, X, Noise, H):

#     snr = 2 * (10 ** (SNR / 10))
#     TxSignal = []
#     # TxSignal1 = []
#     for k in range(K):
#         TxSignal.append(Normalize[k](TxNet[k].transmit(X[k])))
#         TxSignal1.append(
#             np.matmul(X[k] * np.sqrt(snr ** R[k]), np.transpose(V[:, k : k + 1]))
#         )

#     RxSignal = []
#     RxSignal1 = []
#     for k in range(K):
#         RxSig = 0
#         RxSig1 = 0
#         for k1 in range(K):
#             RxSig = RxSig + TxSignal[k1] * H[k, k1]
#             RxSig1 = RxSig1 + TxSignal1[k1] * H[k, k1]
#         RxSig = RxSig + Noise[k]
#         RxSig1 = RxSig1 + Noise[k]
#         Y = RxNet[k].receive(RxSig)
#         Y1 = np.matmul(RxSig1, U[:, k : k + 1])
#         Y1 = Y1 / np.sqrt(snr ** R[k])
#         RxSignal.append(Y)
#         RxSignal1.append(Y1)

#     return TxSignal1, RxSignal1


# def check_convergence(X, Y):
#     x1 = np.mean(X[:5])
#     x2 = np.mean(X[-5:])

#     x3 = max(X)
#     x4 = min(X)

#     if abs(x3 - x4) < 0.001 * x3:
#         return 1

#     if x1 < x2 and x1 > 0.005:
#         return 1

#     if abs(x1 - x2) < 0.001 * x1:
#         return 1

#     if min(Y[-20:]) > min(Y[-40:]):
#         return 1

#     if X[-1] < 1e-6:
#         return 1

#     return 0

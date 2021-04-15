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


def update_weights(TxNet, RxNet, Normalize, K, X, H, Noise, lr_rate, dist, log_mse):

    Loss = []
    l1 = 0
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
            if log_mse:
                l = tf.keras.losses.mse(X[k], Y)
                l = tf.reduce_mean(l)
                # l = -(10 / 2.303) * tf.math.log(1 + (1 / l))
                l = (10 / 2.303) * tf.math.log(l)
            else:
                l = tf.keras.losses.mse(X[k], Y)
            l1 += l
            Loss.append(l)

    for k in range(K):
        if dist:
            loss = Loss[k]
        else:
            loss = l1
        Gradients1 = tape.gradient(loss, TxNet[k].trainable_variables)
        Gradients2 = tape.gradient(loss, Normalize[k].trainable_variables)
        Gradients3 = tape.gradient(loss, RxNet[k].trainable_variables)
        optimizer = tf.keras.optimizers.Adam(
            lr_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
        )
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
    H,
    Signal,
    Noise,
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
    TxNet, RxNet, Normalize, _ = update_weights(
        TxNet, RxNet, Normalize, K, Signal[0], H, Noise[0], lr_rate, False, False
    )
    return TxNet, RxNet, Normalize


def training_model(
    TxNet,
    RxNet,
    Normalize,
    K,
    m,
    n,
    l,
    r,
    n_b,
    b_s,
    epochs,
    lr_rate_init,
    H,
    Signal,
    Noise,
    snr,
    dist,
    log_mse,
):

    # Stage 1 Training
    lr_rate = lr_rate_init
    MSE_loss = []
    for epoch in tqdm(range(epochs)):

        batch_loss = []
        for i in range(n_b):
            TxNet, RxNet, Normalize, mse_loss = update_weights(
                TxNet,
                RxNet,
                Normalize,
                K,
                Signal[i],
                H,
                Noise[i],
                lr_rate,
                dist,
                log_mse,
            )
            # T1, T2, T3 = predict_data(
            #     TxNet, RxNet, Normalize, K, Signal[i], H, Noise[i]
            # )
            batch_loss.append(mse_loss)
        MSE_loss.append(np.mean(batch_loss))
        print("Epoch %d: %2.5f" % (epoch, MSE_loss[-1]))

        # if epoch > 50:
        #     if check_convergence(MSE_loss[-10:], MSE_loss[-40:]):
        #         if lr_rate == 0.001:
        #             lr_rate = 0.0005
        #         elif lr_rate == 0.0005:
        #             lr_rate = 0.0001
        #         elif lr_rate == 0.0001:
        #             lr_rate = 0.00005
        #         else:
        #             break

    return TxNet, RxNet, Normalize, MSE_loss


def predict_data(TxNet, RxNet, Normalize, K, X, H, Noise):

    TxSignal = []
    for k in range(K):
        TxSignal.append(Normalize[k](TxNet[k].transmit(X[k])))

    Loss = []
    l1 = 0
    for k in range(K):
        RxSig = 0
        for k1 in range(K):
            RxSig = RxSig + TxSignal[k1] * H[k, k1]
        RxSig = RxSig + Noise[k]
        Y = RxNet[k].receive(RxSig)
        l = tf.keras.losses.mse(X[k], Y)
        l1 += l
        Loss.append(l)

    TxSignal = []
    RxSignal = []
    Loss1 = []
    for k in range(K):
        TxSig = Normalize[k](TxNet[k].transmit(X[k]))
        TxSignal.append(TxSig)

    for k in range(K):
        RxSig = 0
        for k1 in range(K):
            RxSig = RxSig + TxSignal[k1] * H[k, k1]
        RxSig = RxSig + Noise[k]
        Y = RxNet[k].receive(RxSig)
        RxSignal.append(Y)
        Loss1.append(tf.keras.losses.mse(RxSignal[k], X[k]))

    return X, TxSignal, RxSignal


def check_convergence(X, Y):
    x1 = np.mean(X[:5])
    x2 = np.mean(X[-5:])

    x3 = max(X)
    x4 = min(X)

    if abs(x3 - x4) < 0.001 * x3:
        return 1

    if x1 < x2 and x1 > 0.005:
        return 1

    if abs(x1 - x2) < 0.001 * x1:
        return 1

    if min(Y[-20:]) > min(Y[-40:]):
        return 1

    if X[-1] < 1e-6:
        return 1

    return 0

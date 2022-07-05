import tensorflow as tf
import numpy as np


class GaussianLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.fc_μ = tf.keras.layers.Dense(units=units)
        self.fc_σ = tf.keras.layers.Dense(units=units, activation='softplus')

    def call(self, inputs):
        μ = self.fc_μ(inputs)
        σ = self.fc_σ(inputs) + 1e-6
        return [μ, σ]


class Model(tf.keras.Model):
    def __init__(self, dim_x, dim_z):
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_in = self.dim_x + self.dim_z
        self.dim_out = self.dim_z
        self.lstm_cell = tf.keras.layers.LSTMCell(self.dim_in)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.gaussian = GaussianLayer(units=self.dim_out)

    def warmup(self, z, x):
        zx = tf.concat((z, x), axis=-1)
        zx = tf.reshape(zx, (-1, 1, self.dim_in))
        lstm_out, *state = self.lstm_rnn(zx)
        μ, σ = self.gaussian(lstm_out)
        return μ, σ, state

    def forward(self, z, x, states):
        zx = tf.concat((z, x), axis=-1)
        lstm_out, states = self.lstm_cell(zx, states=states)
        μ, σ = self.gaussian(lstm_out)
        return μ, σ, states

    @staticmethod
    def sample(μ, σ):
        ε = tf.random.normal(shape=tf.shape(μ))
        return μ + σ * ε

    def call(self, inputs):
        """
        :param inputs: shape(batch_size, seq_len, dim_x)
        """
        seq_len = inputs.shape[1]
        μ_preds = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        σ_preds = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        # init_x shape(batch_size, dim_x)
        init_x = inputs[:, 0, :]
        # init_z shape(batch_size, dim_z)
        init_z = tf.constant(np.zeros(shape=(init_x.shape[0], self.dim_z), dtype='float32'))
        # get warmup states
        μ, σ, states = self.warmup(z=init_z, x=init_x)
        # save result
        μ_preds, σ_preds = μ_preds.write(μ_preds.size(), μ), σ_preds.write(σ_preds.size(), σ)
        # loop over the rest steps in sequence
        for n in tf.range(1, seq_len):
            x = inputs[:, n, :]
            # z is sampled from probability distribution described by (μ, σ)
            z = self.sample(μ=μ, σ=σ)
            μ, σ, states = self.forward(z=z, x=x, states=states)
            # save result
            μ_preds, σ_preds = μ_preds.write(μ_preds.size(), μ), σ_preds.write(σ_preds.size(), σ)
        # shape(seq_len, batch_size, dim_z)
        μ_preds, σ_preds = μ_preds.stack(), σ_preds.stack()
        # shape(batch_size, seq_len, dim_z)
        μ_preds, σ_preds = tf.transpose(μ_preds, [1, 0, 2]), tf.transpose(σ_preds, [1, 0, 2])
        return [μ_preds, σ_preds]


class DeepAR(Model):

    def train_step(self, data):
        inp, tar = data
        with tf.GradientTape() as tape:
            μ, σ = self(inputs=inp, training=True)
            loss = self.loss_fn(y_true=tar, μ=μ, σ=σ)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss, "rmse": self.metric_fn(μ[:, 5:, :], tar[:, 5:, :])}

    def test_step(self, data):
        inp, tar = data
        μ, σ = self(inputs=inp)
        return {"loss": self.loss_fn(y_true=tar, μ=μ, σ=σ), "rmse": self.metric_fn(μ[:, 5:, :], tar[:, 5:, :])}

    def compile(self, optimizer, metric_fn):
        super(DeepAR, self).compile()
        self.optimizer = optimizer
        self.metric_fn = metric_fn
        self.loss_fn = self.gaussian_loss

    @staticmethod
    def gaussian_loss(y_true, μ, σ):
        return tf.reduce_mean(0.5 * tf.math.log(σ) + 0.5 * tf.math.divide(tf.math.square(y_true - μ), σ)) + 1e-6 + 6


def create_model(dim_x=4, dim_z=3):
    model = DeepAR(dim_x=dim_x, dim_z=dim_z)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        metric_fn=tf.keras.metrics.RootMeanSquaredError(),
    )
    return model

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import gen_array_ops


class BinaryTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name='binary_true_positives', **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives


class TF_proxy_anchor(tf.keras.layers.Layer):
    def __init__(self, nb_classes, sz_embedding, **kwargs):
        super(TF_proxy_anchor, self).__init__()
        self.nb_classes = tf.cast(nb_classes, tf.int32)
        self.sz_embedding = sz_embedding

        self.proxy = tf.compat.v1.get_variable(name='proxy',
                                               shape=[nb_classes, sz_embedding],
                                               initializer=tf.random_normal_initializer(),
                                               dtype=tf.float32,
                                               trainable=True)

    def get_config(self):
        config = super(TF_proxy_anchor, self).get_config()
        config.update({
            'nb_classes': self.nb_classes.numpy(),
            'proxy': self.proxy.numpy(),
            'sz_embedding': self.sz_embedding
        })
        return config

    def call(self, inputs, **kwargs):
        self.add_loss(self.custom_loss(inputs[0], inputs[1]))
        return inputs[0]

    def get_vars(self):
        return self.proxy

    def custom_loss(self, target, embeddings):
        oh_target = tf.squeeze(tf.one_hot(tf.cast(target, tf.int32), depth=self.nb_classes))
        embeddings_l2 = tf.cast(tf.nn.l2_normalize(embeddings, axis=1), tf.float32)
        proxy_l2 = tf.nn.l2_normalize(self.proxy, axis=1)

        pos_target = oh_target
        neg_target = 1.0 - pos_target

        pos_target = tf.cast(pos_target, tf.bool)
        neg_target = tf.cast(neg_target, tf.bool)

        cos = tf.matmul(embeddings_l2, proxy_l2, transpose_b=True)

        pos_mat = tf.where(pos_target, x=tf.exp(-32 * (cos - 0.1)), y=tf.zeros_like(pos_target, dtype=tf.float32))
        neg_mat = tf.where(neg_target, x=tf.exp(32 * (cos + 0.1)), y=tf.zeros_like(neg_target, dtype=tf.float32))

        n_valid_proxies = tf.math.count_nonzero(tf.reduce_sum(oh_target, axis=0), dtype=tf.dtypes.float32)

        pos_term = tf.reduce_sum(tf.math.log(1.0 + tf.reduce_sum(pos_mat, axis=0))) / n_valid_proxies
        neg_term = tf.reduce_sum(tf.math.log(1.0 + tf.reduce_sum(neg_mat, axis=0))) / tf.cast(self.nb_classes, tf.float32)
        loss = pos_term + neg_term
        return loss

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import gen_array_ops

class TF_proxy_anchor(tf.keras.layers.Layer):
    def __init__(self, nb_classes, batch_size, sz_embedding):
        super(TF_proxy_anchor, self).__init__()
        self.nb_classes = tf.cast(nb_classes, tf.float32)

        self.proxy = tf.compat.v1.get_variable(name='proxy', shape=[nb_classes, sz_embedding],
                                               initializer=tf.random_normal_initializer(),
                                               dtype=tf.float32,
                                               trainable=True)

    def call(self, inputs, **kwargs):
        self.add_loss(self.custom_loss(inputs[0], inputs[1]))
        return inputs[0]

    def get_vars(self):
        return self.proxy

    def distinct(self, a):
        _a = np.unique(a, axis=0)
        return _a

    def custom_loss(self, target, embeddings):
        cosine_similarity = embeddings
        class_num = cosine_similarity.get_shape().as_list()[1]
        P_one_hot = tf.one_hot(indices=tf.argmax(target, axis=1),
                               depth=class_num,
                               on_value=None,
                               off_value=None)
        N_one_hot = 1.0 - P_one_hot

        pos_exp = tf.exp(-32 * (cosine_similarity - 0.1))
        neg_exp = tf.exp(32 * (cosine_similarity + 0.1))

        P_sim_sum = tf.reduce_sum(pos_exp * P_one_hot, axis=0)
        N_sim_sum = tf.reduce_sum(neg_exp * N_one_hot, axis=0)

        num_valid_proxies = tf.math.count_nonzero(tf.reduce_sum(P_one_hot, axis=0),
                                                  dtype=tf.dtypes.float32)

        pos_term = tf.reduce_sum(tf.math.log(1.0 + P_sim_sum)) / num_valid_proxies
        neg_term = tf.reduce_sum(tf.math.log(1.0 + N_sim_sum)) / class_num
        loss = pos_term + neg_term

        return loss

    def tf_unique_2d(self, x):
        x_shape = tf.shape(x)  # (3,2)
        x1 = tf.tile(x, [1, x_shape[0]])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]
        x2 = tf.tile(x, [x_shape[0], 1])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]

        x1_2 = tf.reshape(x1, [x_shape[0] * x_shape[0], x_shape[1]])
        x2_2 = tf.reshape(x2, [x_shape[0] * x_shape[0], x_shape[1]])
        cond = tf.reduce_all(tf.equal(x1_2, x2_2), axis=1)
        cond = tf.reshape(cond, [x_shape[0], x_shape[0]])  # reshaping cond to match x1_2 & x2_2
        cond_shape = tf.shape(cond)
        cond_cast = tf.cast(cond, tf.int32)  # convertin condition boolean to int
        cond_zeros = tf.zeros(cond_shape, tf.int32)  # replicating condition tensor into all 0's

        # CREATING RANGE TENSOR
        r = tf.range(x_shape[0])
        r = tf.add(tf.tile(r, [x_shape[0]]), 1)
        r = tf.reshape(r, [x_shape[0], x_shape[0]])

        # converting TRUE=1 FALSE=MAX(index)+1 (which is invalid by default) so when we take min it wont get selected & in end we will only take values <max(indx).
        f1 = tf.multiply(tf.ones(cond_shape, tf.int32), x_shape[0] + 1)
        f2 = tf.ones(cond_shape, tf.int32)
        cond_cast2 = tf.where(tf.equal(cond_cast, cond_zeros), f1, f2)  # if false make it max_index+1 else keep it 1

        # multiply range with new int boolean mask
        r_cond_mul = tf.multiply(r, cond_cast2)
        r_cond_mul2 = tf.reduce_min(r_cond_mul, axis=1)
        r_cond_mul3, unique_idx = tf.unique(r_cond_mul2)
        r_cond_mul4 = tf.subtract(r_cond_mul3, 1)

        # get actual values from unique indexes
        op = tf.gather(x, r_cond_mul4)

        return (op)

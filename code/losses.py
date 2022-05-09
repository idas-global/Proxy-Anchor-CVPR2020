from pytorch_metric_learning import miners, losses
import tensorflow as tf
# use tensorflow graph with tf 1.xx version


class TF_proxy_anchor(tf.keras.layers.Layer):
    def __init__(self, nb_classes, batch_size, sz_embedding):
        super(TF_proxy_anchor, self).__init__()
        self.nb_classes = tf.cast(nb_classes, tf.float32)

        self.proxy = tf.compat.v1.get_variable(name='proxy', shape=[batch_size, sz_embedding],
                                               initializer=tf.random_normal_initializer(),
                                               dtype=tf.float32,
                                               trainable=True)

    def call(self, inputs, **kwargs):
        self.add_loss(self.custom_loss(inputs[0], inputs[1]))
        return inputs[0]

    def get_vars(self):
        return self.proxy

    def custom_loss(self, target, embeddings):
        #n_unique = tf.cast(tf.shape(self.tf_unique_2d(target))[0], tf.float32)
        n_unique = 48
        embeddings_l2 = tf.nn.l2_normalize(embeddings, axis=1)
        proxy_l2 = tf.nn.l2_normalize(self.proxy, axis=1)

        pos_target = target
        neg_target = 1.0 - pos_target

        sim_mat = tf.matmul(embeddings_l2, proxy_l2, transpose_b=True)

        pos_mat = tf.matmul(tf.exp(-32 * (sim_mat - 0.1)), pos_target)
        neg_mat = tf.matmul(tf.exp(32 * (sim_mat - 0.1)), neg_target)

        # n_unique = batch_size // n_instance

        pos_term = tf.constant(1.0) / n_unique * tf.reduce_sum(
            tf.math.log(1.0 + tf.reduce_sum(pos_mat, axis=0)))
        neg_term = tf.constant(1.0) / self.nb_classes * tf.reduce_sum(tf.math.log(1.0 + tf.reduce_sum(neg_mat, axis=0)))

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

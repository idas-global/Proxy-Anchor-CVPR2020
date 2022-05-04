import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses
import numpy as np

import tensorflow as tf

# use tensorflow graph with tf 1.xx version

class TF_proxy_anchor:
    def __init__(self, model, nb_classes, batch_size, sz_embedding):
        self.nb_classes = nb_classes
        model.proxy = tf.compat.v1.get_variable(name='proxy', shape=[batch_size, sz_embedding],
                                               initializer=tf.random_normal_initializer(),
                                               dtype=tf.float32,
                                               trainable=True)
        self.proxy = model.proxy

    def proxy_anchor_loss(self, target, embeddings):
        n_unique = self.tf_unique_2d(target).shape[0]
        embeddings_l2 = tf.nn.l2_normalize(embeddings, axis=1)
        proxy_l2 = tf.nn.l2_normalize(self.proxy, axis=1)

        pos_target = target
        neg_target = 1.0 - pos_target

        sim_mat = tf.matmul(embeddings_l2, proxy_l2, transpose_b=True)

        pos_mat = tf.matmul(tf.exp(-32 * (sim_mat - 0.1)), pos_target)
        neg_mat = tf.matmul(tf.exp(32 * (sim_mat - 0.1)), neg_target)

        # n_unique = batch_size // n_instance

        pos_term = 1.0 / n_unique * tf.reduce_sum(
            tf.math.log(1.0 + tf.reduce_sum(pos_mat, axis=0)))
        neg_term = 1.0 / self.nb_classes * tf.reduce_sum(tf.math.log(1.0 + tf.reduce_sum(neg_mat, axis=0)))

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

def proxy_anchor_loss(y_true, y_pred):
    margin = 0.1
    alpha = 32
    print('Got Here')
    cosine_similarity = y_pred
    class_num = cosine_similarity.get_shape().as_list()[1]
    P_one_hot = tf.one_hot(indices=tf.argmax(y_true, axis=1),
                           depth=class_num,
                           on_value=None,
                           off_value=None)
    N_one_hot = 1.0 - P_one_hot

    pos_exp = tf.exp(-alpha * (cosine_similarity - margin))
    neg_exp = tf.exp(alpha * (cosine_similarity + margin))

    P_sim_sum = tf.reduce_sum(pos_exp * P_one_hot, axis=0)
    N_sim_sum = tf.reduce_sum(neg_exp * N_one_hot, axis=0)

    num_valid_proxies = tf.math.count_nonzero(tf.reduce_sum(P_one_hot, axis=0),
                                              dtype=tf.dtypes.float32)

    pos_term = tf.reduce_sum(tf.math.log(1.0 + P_sim_sum)) / num_valid_proxies
    neg_term = tf.reduce_sum(tf.math.log(1.0 + N_sim_sum)) / class_num
    loss = pos_term + neg_term

    return loss


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T)
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, softmax_scale = self.scale)

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin) 
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'semihard')
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings = False)
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
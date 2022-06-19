import tensorflow as tf
import numpy as np
def binary_focal_loss(cfg):
    alpha = tf.constant(cfg["train"]["bfl"]["alpha"], dtype=tf.float32)
    gamma = tf.constant(cfg["train"]["bfl"]["gamma"], dtype=tf.float32)
    epsilon = tf.constant(cfg["train"]["bfl"]["epsilon"], dtype=tf.float32)
    n_classes = cfg["train"]["gender_num"]
    # 得到y_true和y_pred
    def cal_loss(logits, label):
        y_true = tf.one_hot(label, n_classes)
        probs = tf.nn.sigmoid(logits)
        y_pred = tf.clip_by_value(probs, epsilon, 1. - epsilon)
        # 得到调节因子weight和alpha
        ## 先得到y_true和1-y_true的概率【这里是正负样本的概率都要计算哦！】
        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred)
        ## 然后通过p_t和gamma得到weight
        weight = tf.pow((tf.ones_like(y_true) - p_t), gamma)
        ## 再得到alpha，y_true的是alpha，那么1-y_true的是1-alpha
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        # 最后就是论文中的公式，相当于：- alpha * (1-p_t)^gamma * log(p_t)
        focal_loss = - alpha_t * weight * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    return cal_loss

def softmax_focal_loss(cfg):
    alpha = tf.constant(cfg["train"]["sfl"]["alpha"], dtype=tf.float32)
    gamma = tf.constant(cfg["train"]["sfl"]["gamma"], dtype=tf.float32)
    epsilon = tf.constant(cfg["train"]["sfl"]["epsilon"], dtype=tf.float32)
    # y_true and y_pred
    def cal_loss(logits, label):
        probs = tf.nn.softmax(logits)
        y_pred = tf.clip_by_value(probs, epsilon, 1. - epsilon)
        # weight term and alpha term【因为y_true是只有1个元素为1其他元素为0的one-hot向量，所以对于每个样本，只有y_true位置为1的对应类别才有weight，其他都是0】这也是为什么网上有的版本会用到tf.gather函数，这个函数的作用就是只把有用的这个数取出来，可以省略一些0相关的运算。
        label = np.squeeze(np.array(label)).astype(np.float32) 
        weight = tf.multiply(label, tf.pow(tf.subtract(1., y_pred), gamma))
        if alpha != 0.0:  # 我这实现中的alpha只是起到了调节loss倍数的作用（调节倍数对训练没影响，因为loss的梯度才是影响训练的关键），要想起到调节类别不均衡的作用，要替换成数组，数组长度和类别总数相同，每个元素表示对应类别的权重。另外[这篇](https://blog.csdn.net/Umi_you/article/details/80982190)博客也提到了，alpha在多分类Focal loss中没作用，也就是只能调节整体loss倍数，不过如果换成数组形式的话，其实是可以达到缓解类别不均衡问题的目的。
            alpha_t = tf.cast(label * alpha, tf.float32) + (tf.ones_like(label) - label) * tf.cast((1.0 - alpha), tf.float32)
        else:
            alpha_t = tf.ones_like(label)
        # origin x ent，这里计算原始的交叉熵损失
        xent = tf.multiply(label, -tf.math.log(y_pred))
        # focal x ent，对交叉熵损失进行调节，“-”号放在上一行代码了，所以这里不需要再写“-”了。
        focal_xent = tf.multiply(alpha_t, tf.multiply(weight, xent))
        # in this situation, reduce_max is equal to reduce_sum，因为经过y_true选择后，每个样本只保留了true label对应的交叉熵损失，所以使用max和使用sum是同等作用的。
        reduced_fl = tf.reduce_max(focal_xent, axis=1)
        return tf.reduce_mean(reduced_fl)
    return cal_loss

def get_loss(cfg):
    age_loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    gender_loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    if cfg.train.gender_loss == "binary_focal":
        gender_loss_object = binary_focal_loss(cfg)
    if cfg.train.age_loss == "softmax_focal":
        age_loss_object = softmax_focal_loss(cfg)
    return gender_loss_object, age_loss_object



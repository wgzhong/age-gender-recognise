import tensorflow as tf
import numpy as np

#公式：L(pt) = -αt(1-pt)^γ log(pt)，
# pt=p and αt=α  when y=1 ,pt=1-p and αt=1-α when y=-1或者0 视情况而定
def binary_focal_loss(cfg):
    alpha = tf.constant(cfg["train"]["bfl"]["alpha"], dtype=tf.float32)
    gamma = tf.constant(cfg["train"]["bfl"]["gamma"], dtype=tf.float32)
    epsilon = tf.constant(cfg["train"]["bfl"]["epsilon"], dtype=tf.float32)
    def focal_loss(y_true, y_probs):
        positive_pt = tf.where(tf.equal(y_true, 1), y_probs, tf.ones_like(y_probs))
        negative_pt = tf.where(tf.equal(y_true, 0), 1-y_probs, tf.ones_like(y_probs))
        
        loss =  -alpha * tf.pow(1-positive_pt, gamma) * tf.math.log(tf.clip_by_value(positive_pt, epsilon, 1.)) - \
            (1-alpha) * tf.pow(1-negative_pt, gamma) * tf.math.log(tf.clip_by_value(negative_pt,  epsilon, 1.))

        return tf.reduce_sum(loss)
    return focal_loss

def get_loss(cfg):
    loss_object = tf.keras.losses.BinaryFocalCrossentropy(from_logits=True)
    if cfg.train.loss == "binary_focal":
        loss_object = binary_focal_loss(cfg)
    return loss_object



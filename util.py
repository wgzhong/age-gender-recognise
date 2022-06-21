import tensorflow as tf
from tensorflow import keras

class BinaryFocalLossAccuracy(keras.metrics.Metric):
    def __init__(self, name='BinaryFocalLossAccuracy', **kwargs):
        super(BinaryFocalLossAccuracy, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        values = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])
        self.tp.assign_add(tf.reduce_sum(values))
 
    def result(self):
        return self.tp / self.total

class SoftmaxFocalLossAccuracy(keras.metrics.Metric):
    def __init__(self, name='SoftmaxFocalLossAccuracy', **kwargs):
        super(SoftmaxFocalLossAccuracy, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        values = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])
        self.tp.assign_add(tf.reduce_sum(values))
 
    def result(self):
        return self.tp / self.total





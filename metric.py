import tensorflow as tf
from tensorflow import keras

class BinaryFocalLossAccuracy(keras.metrics.Metric):
    def __init__(self, name='BinaryFocalLossAccuracy', **kwargs):
        super(BinaryFocalLossAccuracy, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
    def update_state(self, y_true, y_pred, sample_weight=None):
        gender = tf.where(y_pred[:,0]>0.5, 1, 0)
        p_age = tf.argmax(y_true[0:,1:], axis=-1, output_type=tf.int32)
        age = tf.argmax(y_pred[0:,1:], axis=-1, output_type=tf.int32)
        a1 = tf.equal(gender, y_true[:,0])
        a2 = tf.equal(age, p_age)
        a3 = tf.cast(a1&a2, dtype="int32")
        # self.tp.assign_add(tf.reduce_mean(tf.cast(a1&a2, dtype='float32')))
        num = tf.cast(tf.size(y_true)/tf.size(y_true[0]), dtype="int32")
        self.total.assign_add(num)
        self.tp.assign_add(tf.reduce_sum(a3))
 
    def result(self):
        return self.tp / self.total

        
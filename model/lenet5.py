import tensorflow as tf
from tensorflow.keras import datasets, layers, models
class Lenet5(tf.keras.Model):
    def __init__(self, cfg):
        super(Lenet5, self).__init__()
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(6, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(120, activation='relu'))
        self.model.add(layers.Dropout( rate=0.5, name="Dropout"))
        self.model.add(layers.Dense(84, activation='relu'))
        self.model.add(layers.Dense(cfg.train.num_classes, activation='sigmoid'))
    def call(self, input):
        x=self.model(input)
        return x
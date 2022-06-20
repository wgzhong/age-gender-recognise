import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from model.mobilenetv3 import *
from model.loss import *
from config.config import *
from data.pa100k import *
import hydra
    
@hydra.main(config_path="./config/", config_name="config.yaml")
def main(cfg):
    epochs = cfg.train.epochs
    model = mobilenetv3_small(cfg)
    optimizer = get_optimizer(cfg)
    gender_loss_object, age_loss_object = get_loss(cfg)
    train_dataset = ImageSequence(cfg, "train")
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training = True)
            genger_loss = gender_loss_object(logits[0], y[0])
            age_loss = age_loss_object(logits[1], y[1:])
            loss = genger_loss + age_loss
        # # Use the gradient tape to automatically retrieve
        # # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss, model.trainable_weights)
        # # Run one step of gradient descent by updating
        # # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss
    
    for epoch in range(epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss = train_step(x_batch_train, y_batch_train)
            if step % 400 == 0:
                print("epoch is %d: training loss (for one batch) at step %d: %.4f" % (epoch, step, float(loss)))
            train_dataset.on_epoch_end()
 
    



if __name__ == '__main__':
    main()
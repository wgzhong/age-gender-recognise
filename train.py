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
    batch_size = cfg.train.batch_size
    model = mobilenetv3_small(cfg)
    optimizer = get_optimizer(cfg)
    gender_loss_object, age_loss_object = get_loss(cfg)
    train_dataset = ImageSequence(cfg, "train")
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training = True)
                genger_loss = gender_loss_object(logits[0], y_batch_train[0])
                print(logits[1].shape)
                age_loss = age_loss_object(logits[1], y_batch_train[1:])
            loss = genger_loss + age_loss
            # # Use the gradient tape to automatically retrieve
            # # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss, model.trainable_weights)
            # # Run one step of gradient descent by updating
            # # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            # # # Log every 200 batches.
            # if step % 10 == 0:
            #     print(
            #         "Training loss (for one batch) at step %d: %.4f"
            #         % (step, float(loss))
            #     )
            #     print("Seen so far: %s samples" % ((step + 1) * batch_size))

 
    



if __name__ == '__main__':
    main()
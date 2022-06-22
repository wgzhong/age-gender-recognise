import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from model.mobilenetv3 import *
from model.loss import *
from config.config import *
from data.pa100k import *
from util import *
import hydra
    
@hydra.main(config_path="./config/", config_name="config.yaml")
def main(cfg):
    epochs = cfg.train.epochs
    model = mobilenetv3_small(cfg)
    optimizer = get_optimizer(cfg)
    gender_loss_object, age_loss_object = get_loss(cfg)
    #train
    train_gender_loss = tf.keras.metrics.Mean('train_gender_loss', dtype=tf.float32)
    train_age_loss = tf.keras.metrics.Mean('train_age_loss', dtype=tf.float32)
    train_gender_accuracy = BinaryFocalLossAccuracy('train_gender_accuracy')
    train_age_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_age_accuracy')
    #val
    val_gender_loss = tf.keras.metrics.Mean('val_gender_loss', dtype=tf.float32)
    val_age_loss = tf.keras.metrics.Mean('val_age_loss', dtype=tf.float32)
    val_gender_accuracy = BinaryFocalLossAccuracy('val_gender_accuracy')
    val_age_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('val_age_accuracy')
    #datasets
    train_dataset = ImageSequence(cfg, "train")
    val_dataset = ImageSequence(cfg, "val")
    #summary
    train_log_dir = 'logs/gradient_tape/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_log_dir = 'logs/gradient_tape/val'
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    #cycle
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training = True)
            genger_loss = gender_loss_object(tf.one_hot(y[0], 2), logits[0])
            age_loss = age_loss_object(y[1], logits[1])
            loss = genger_loss + age_loss
        # # Use the gradient tape to automatically retrieve
        # # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss, model.trainable_weights)
        # # Run one step of gradient descent by updating
        # # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_gender_loss(genger_loss)
        train_age_loss(age_loss)
        train_gender_accuracy(y[0], logits[0])
        # train_age_accuracy(logits[1], y[1])

        return loss
    
    @tf.function
    def val_step(x, y):
        print("val")

    for epoch in range(epochs):
        train_dataset.on_epoch_end()
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss = train_step(x_batch_train, y_batch_train)
            template = 'Epoch {}, step {}, gender_loss: {}, age_loss: {}, total Loss: {}, gender_accuracy: {}'
            print(template.format(epoch + 1, step,
                                    train_gender_loss.result(),
                                    train_age_loss.result(),
                                    loss,
                                    train_gender_accuracy.result() * 100,
                                    ))
        with train_summary_writer.as_default():
            tf.summary.scalar('train_age_loss', train_age_loss.result(), step=epoch)
            tf.summary.scalar('train_gender_loss', train_gender_loss.result(), step=epoch)
            tf.summary.scalar('train_gender_accuracy', train_gender_accuracy.result(), step=epoch)
        
        # for step, (x_batch_tval, y_batch_val) in enumerate(val_dataset):
        #     loss = val_step(x_batch_tval, y_batch_val)
        # with val_summary_writer.as_default():
        #     tf.summary.scalar('val_gender_loss', val_age_loss.result(), step=epoch)
        #     tf.summary.scalar('val_gender_loss', val_gender_loss.result(), step=epoch)
        #     tf.summary.scalar('val_gender_accuracy', val_gender_accuracy.result(), step=epoch)

        train_gender_loss.reset_states()
        train_age_loss.reset_states()
        train_gender_accuracy.reset_states()
        # train_age_accuracy.reset_states()
        val_gender_loss.reset_states()
        val_age_loss.reset_states()
        val_gender_accuracy.reset_states()
        # train_age_accuracy.reset_states()
    



if __name__ == '__main__':
    main()
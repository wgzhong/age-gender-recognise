import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import *
from model.mobilenetv3 import *
from model.loss import *
from config.config import *
from data.pa100k import *
from util import *
import hydra
import os

@hydra.main(config_path="./config/", config_name="config.yaml")
def main(cfg):
    save_model_dir="./save_model"
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    base_model = keras.applications.MobileNetV3Small(
        weights="imagenet",  
        input_shape=(224, 112, 3),
        include_top=False) 
    base_model.trainable = False
    # model = tf.keras.Sequential([
    # base_model,
    # GlobalAveragePooling2D(),
    # Dense(cfg["train"]["gender_num"])
    # ])

    inputs = keras.Input(shape=(224, 112, 3))
    x = base_model(inputs, training=False)

    x = GlobalAveragePooling2D(name='average_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001), )(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = BatchNormalization(name='bn_fc_01')(x)
    x = hswish()(x)

    gender = Dense(cfg["train"]["gender_num"])(x)
    age = Dense(cfg["train"]["age_num"])(x)
    model = keras.Model(inputs, [gender, age])
    model.summary()
    #datasets
    train_dataset = ImageSequence(cfg, "train")
    val_dataset = ImageSequence(cfg, "val")
    optimizer = get_optimizer(cfg)
    gender_loss_object, age_loss_object = get_loss(cfg)
    # model.compile(loss=[gender_loss_object], optimizer=optimizer, metrics=['binary_accuracy'])
    # model.fit(train_dataset, epochs=500, validation_data=val_dataset)

    #train
    train_gender_loss = tf.keras.metrics.Mean('train_gender_loss', dtype=tf.float32)
    train_age_loss = tf.keras.metrics.Mean('train_age_loss', dtype=tf.float32)
    train_total_loss = tf.keras.metrics.Mean('train_total_loss', dtype=tf.float32)
    train_gender_accuracy = tf.keras.metrics.CategoricalAccuracy("train_gender_accuracy")#BinaryFocalLossAccuracy('train_gender_accuracy')
    train_age_accuracy = tf.keras.metrics.CategoricalAccuracy('train_age_accura+cy')
    #val
    val_gender_loss = tf.keras.metrics.Mean('val_gender_loss', dtype=tf.float32)
    val_age_loss = tf.keras.metrics.Mean('val_age_loss', dtype=tf.float32)
    val_total_loss = tf.keras.metrics.Mean('val_total_loss', dtype=tf.float32)
    val_gender_accuracy = tf.keras.metrics.CategoricalAccuracy("val_gender_accuracy")#BinaryFocalLossAccuracy('val_gender_accuracy')
    val_age_accuracy = tf.keras.metrics.CategoricalAccuracy('val_age_accuracy')
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
            # tf.print(logits[0], tf.one_hot(y[0], 2), y[2])
            gender_loss = gender_loss_object(tf.one_hot(y[0], 2), logits[0])
            age_loss = age_loss_object(y[1], logits[1])
            loss = gender_loss + age_loss
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        #recoder
        train_gender_loss(gender_loss)
        train_age_loss(age_loss)
        train_total_loss(loss)
        train_gender_accuracy(tf.one_hot(y[0], 2), logits[0])
        train_age_accuracy(y[1], logits[1])
    
    @tf.function
    def val_step(x, y):
        logits = model(x, training = False)
        # tf.print(logits[0], logits[1], y[0], y[1], image_path)
        gender_loss = gender_loss_object(tf.one_hot(y[0], 2), logits[0])
        age_loss = age_loss_object(y[1], logits[1])
        loss = gender_loss + age_loss
        val_gender_loss(gender_loss)
        val_age_loss(age_loss)
        val_total_loss(loss)
        val_gender_accuracy(tf.one_hot(y[0], 2), logits[0])
        val_age_accuracy(y[1], logits[1])

    for epoch in range(cfg.train.epochs):
        train_dataset.on_epoch_end()
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            train_step(x_batch_train, y_batch_train)
        template = 'Epoch {}, gender_loss: {}, age_loss: {}, total Loss: {}, gender_accuracy: {}, age_accuracy: {}'
        print(template.format(epoch + 1,
                            train_gender_loss.result(),
                            train_age_loss.result(),
                            train_total_loss.result(),
                            train_gender_accuracy.result() * 100,
                            train_age_accuracy.result() * 100
                            ))
        with train_summary_writer.as_default():
            tf.summary.scalar('train_age_loss', train_age_loss.result(), step=epoch)
            tf.summary.scalar('train_gender_loss', train_gender_loss.result(), step=epoch)
            tf.summary.scalar('train_total_loss', train_total_loss.result(), step=epoch)
            tf.summary.scalar('train_gender_accuracy', train_gender_accuracy.result(), step=epoch)
            tf.summary.scalar('train_age_accuracy', train_age_accuracy.result(), step=epoch)
        
        if epoch%5==0 and epoch>0:
            val_dataset.on_epoch_end()
            for step, (x_batch_tval, y_batch_val) in enumerate(val_dataset):
                val_step(x_batch_tval, y_batch_val)
            with val_summary_writer.as_default():
                tf.summary.scalar('val_gender_loss', val_age_loss.result(), step=epoch)
                tf.summary.scalar('val_gender_loss', val_gender_loss.result(), step=epoch)
                tf.summary.scalar('val_total_loss', val_total_loss.result(), step=epoch)
                tf.summary.scalar('val_gender_accuracy', val_gender_accuracy.result(), step=epoch)
                tf.summary.scalar('val_age_accuracy', val_age_accuracy.result(), step=epoch)
            val_template = 'val dataset: gender_loss: {}, age_loss: {}, total Loss: {}, gender_accuracy: {}, age_accuracy: {}'
            print(val_template.format(val_gender_loss.result(),
                            val_age_loss.result(),
                            val_total_loss.result(),
                            val_gender_accuracy.result() * 100,
                            val_age_accuracy.result() * 100
                            ))
            val_gender_loss.reset_states()
            val_age_loss.reset_states()
            val_total_loss.reset_states()
            val_gender_accuracy.reset_states()
            val_age_accuracy.reset_states()
        if epoch % 50 == 0 and epoch>0:
            model.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch), save_format='tf')


        train_gender_loss.reset_states()
        train_age_loss.reset_states()
        train_total_loss.reset_states()
        train_gender_accuracy.reset_states()
        train_age_accuracy.reset_states()
    # save weights
    model.save_weights(filepath=save_model_dir+"model", save_format='tf') #保存训练的权值



if __name__ == '__main__':
    main()
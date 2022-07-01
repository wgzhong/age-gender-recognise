import tensorflow as tf
from model.layers import LastStage
from tensorflow.keras.layers import *
from model.loss import *
from config.config import *
from data.pa100k import *
from metric import *
import hydra
import os

@hydra.main(config_path="./config/", config_name="config.yaml")
def main(cfg):
    save_model_dir="./save_model"
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    base_model = keras.applications.MobileNetV2(
        weights="imagenet",  
        input_shape=(cfg.model.img_size_h, cfg.model.img_size_w, 3),
        include_top=False) 
    base_model.trainable = False
    inputs = keras.Input(shape=(cfg.model.img_size_h, cfg.model.img_size_w, 3))
    x = base_model(inputs, training=False)
    last_stage = LastStage(72, 160, cfg["train"]["num_classes"], l2_reg=1e-5)
    x = last_stage(x)
    model = keras.Model(inputs, x)
    #datasets
    train_dataset = ImageSequence(cfg, "train")
    val_dataset = ImageSequence(cfg, "val")
    optimizer = get_optimizer(cfg)
    loss_object = get_loss(cfg)
    train_accuracy = BinaryFocalLossAccuracy('accuracy')

    model.compile(optimizer=optimizer,
              loss=[loss_object],
              metrics=[train_accuracy])
    model.summary()
    model.fit(train_dataset,
                    epochs=cfg.train.epochs,
                    validation_data=val_dataset)

    # #train
    # train_gender_loss = tf.keras.metrics.Mean('train_gender_loss', dtype=tf.float32)
    # train_age_loss = tf.keras.metrics.Mean('train_age_loss', dtype=tf.float32)
    # train_total_loss = tf.keras.metrics.Mean('train_total_loss', dtype=tf.float32)
    # train_gender_accuracy = tf.keras.metrics.CategoricalAccuracy("train_gender_accuracy")#BinaryFocalLossAccuracy('train_gender_accuracy')
    # train_age_accuracy = tf.keras.metrics.CategoricalAccuracy('train_age_accura+cy')
    # #val
    # val_gender_loss = tf.keras.metrics.Mean('val_gender_loss', dtype=tf.float32)
    # val_age_loss = tf.keras.metrics.Mean('val_age_loss', dtype=tf.float32)
    # val_total_loss = tf.keras.metrics.Mean('val_total_loss', dtype=tf.float32)
    # val_gender_accuracy = tf.keras.metrics.CategoricalAccuracy("val_gender_accuracy")#BinaryFocalLossAccuracy('val_gender_accuracy')
    # val_age_accuracy = tf.keras.metrics.CategoricalAccuracy('val_age_accuracy')
    # #summary
    # train_log_dir = 'logs/gradient_tape/train'
    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # val_log_dir = 'logs/gradient_tape/val'
    # val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    # #cycle
    # @tf.function
    # def train_step(x, y):
    #     with tf.GradientTape() as tape:
    #         logits = model(x, training = True)
    #         # tf.print(logits[0], y[0], y[2])
    #         gender_loss = gender_loss_object(y[0], logits[0])
    #         age_loss = age_loss_object(y[1], logits[1])
    #         loss = gender_loss + age_loss
    #     grads = tape.gradient(loss, model.trainable_weights)
    #     optimizer.apply_gradients(zip(grads, model.trainable_weights))
    #     #recoder
    #     train_gender_loss(gender_loss)
    #     train_age_loss(age_loss)
    #     train_total_loss(loss)
    #     train_gender_accuracy(y[0], logits[0])
    #     train_age_accuracy(y[1], logits[1])
    
    # @tf.function
    # def val_step(x, y):
    #     logits = model(x, training = False)
    #     # tf.print(logits[0], logits[1], y[0], y[1], image_path)
    #     gender_loss = gender_loss_object(y[0], logits[0])
    #     age_loss = age_loss_object(y[1], logits[1])
    #     loss = gender_loss + age_loss
    #     val_gender_loss(gender_loss)
    #     val_age_loss(age_loss)
    #     val_total_loss(loss)
    #     val_gender_accuracy(y[0], logits[0])
    #     val_age_accuracy(y[1], logits[1])

    # for epoch in range(cfg.train.epochs):
    #     train_dataset.on_epoch_end()
    #     for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
    #         train_step(x_batch_train, y_batch_train)
    #     template = 'Epoch {}, gender_loss: {}, age_loss: {}, total Loss: {}, gender_accuracy: {}, age_accuracy: {}'
    #     print(template.format(epoch + 1,
    #                         train_gender_loss.result(),
    #                         train_age_loss.result(),
    #                         train_total_loss.result(),
    #                         train_gender_accuracy.result() * 100,
    #                         train_age_accuracy.result() * 100
    #                         ))
    #     with train_summary_writer.as_default():
    #         tf.summary.scalar('train_age_loss', train_age_loss.result(), step=epoch)
    #         tf.summary.scalar('train_gender_loss', train_gender_loss.result(), step=epoch)
    #         tf.summary.scalar('train_total_loss', train_total_loss.result(), step=epoch)
    #         tf.summary.scalar('train_gender_accuracy', train_gender_accuracy.result(), step=epoch)
    #         tf.summary.scalar('train_age_accuracy', train_age_accuracy.result(), step=epoch)
        
    #     if epoch%1==0 and epoch>0:
    #         val_dataset.on_epoch_end()
    #         for step, (x_batch_tval, y_batch_val) in enumerate(val_dataset):
    #             val_step(x_batch_tval, y_batch_val)
    #         with val_summary_writer.as_default():
    #             tf.summary.scalar('val_gender_loss', val_age_loss.result(), step=epoch)
    #             tf.summary.scalar('val_gender_loss', val_gender_loss.result(), step=epoch)
    #             tf.summary.scalar('val_total_loss', val_total_loss.result(), step=epoch)
    #             tf.summary.scalar('val_gender_accuracy', val_gender_accuracy.result(), step=epoch)
    #             tf.summary.scalar('val_age_accuracy', val_age_accuracy.result(), step=epoch)
    #         val_template = 'val dataset: gender_loss: {}, age_loss: {}, total Loss: {}, gender_accuracy: {}, age_accuracy: {}'
    #         print(val_template.format(val_gender_loss.result(),
    #                         val_age_loss.result(),
    #                         val_total_loss.result(),
    #                         val_gender_accuracy.result() * 100,
    #                         val_age_accuracy.result() * 100
    #                         ))
    #         val_gender_loss.reset_states()
    #         val_age_loss.reset_states()
    #         val_total_loss.reset_states()
    #         val_gender_accuracy.reset_states()
    #         val_age_accuracy.reset_states()
    #     if epoch % 50 == 0 and epoch>0:
    #         model.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch), save_format='tf')


    #     train_gender_loss.reset_states()
    #     train_age_loss.reset_states()
    #     train_total_loss.reset_states()
    #     train_gender_accuracy.reset_states()
    #     train_age_accuracy.reset_states()
    # # save weights
    # model.save_weights(filepath=save_model_dir+"model", save_format='tf') #保存训练的权值



if __name__ == '__main__':
    main()
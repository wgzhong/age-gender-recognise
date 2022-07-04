import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from model.mobilenetv3_factory import build_mobilenetv3
from model.lenet5 import Lenet5
from model.loss import *
from config.config import *
from data.pa100k import *
from metric import *
from tqdm import tqdm
import hydra
import os
def model_save(save_model_dir, model):
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)  
    model.save(save_model_dir)

@hydra.main(config_path="./config/", config_name="config.yaml")
def main(cfg):
    save_model_dir="./save_model/"
    # model = build_mobilenetv3(
    #     "small",
    #     input_shape=(cfg.model.img_size_h, cfg.model.img_size_w, 3),
    #     num_classes=4,
    #     width_multiplier=1.0,
    #     )
    
    model = tf.keras.models.load_model("/home/wgzhong/pywork/age-gender-recognise/outputs/2022-07-01/21-44-35/save_model/")
    
    # checkpoint = tf.train.Checkpoint(model) # myAwesomeModel，这是你原来保存的checkpoint时的model名字
    # checkpoint.restore(tf.train.latest_checkpoint('/home/wgzhong/pywork/age-gender-recognise/outputs/2022-06-30/07-48-28')) # 恢复最新的checkpoint


    model.summary()
    optimizer = get_optimizer(cfg)
    loss_object = get_loss(cfg)
    #train
    train_total_loss = tf.keras.metrics.Mean('train_total_loss', dtype=tf.float32)
    train_accuracy = BinaryFocalLossAccuracy('train_gender_accuracy')
    #val
    val_total_loss = tf.keras.metrics.Mean('val_total_loss', dtype=tf.float32)
    val_accuracy = BinaryFocalLossAccuracy('val_gender_accuracy')
    #summary
    train_log_dir = 'logs/gradient_tape/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_log_dir = 'logs/gradient_tape/val'
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    #datasets
    train_dataset = ImageSequence(cfg, "train")
    val_dataset = ImageSequence(cfg, "val")
    #cycle
    @tf.function
    def train_step(x, y, image_path):
        with tf.GradientTape() as tape:
            logits = model(x, training = True)
            # tf.print(logits, y, image_path)
            loss = loss_object(y, logits)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        #recoder
        train_total_loss(loss)
        train_accuracy(y, logits)
    
    @tf.function
    def val_step(x, y, image_path):
        logits = model(x, training = False)
        # tf.print(logits[0], logits[1], y[0], y[1], image_path)
        loss = loss_object(y, logits)
        val_total_loss(loss)
        val_accuracy(y, logits)
        
    for epoch in range(cfg.train.epochs):
        train_dataset.on_epoch_end()
        loop = tqdm(enumerate(train_dataset), total =len(train_dataset))
        for step, (x_batch_train, y_batch_train, image_path) in loop:
            train_step(x_batch_train, y_batch_train, image_path)
            loop.set_description(f'Train Epoch [{epoch}/{cfg.train.epochs}]')
            loop.set_postfix(acc = train_accuracy.result().numpy() * 100, loss = train_total_loss.result().numpy())
        with train_summary_writer.as_default():
            tf.summary.scalar('train_total_loss', train_total_loss.result(), step=epoch)
            tf.summary.scalar('train_accuracy', train_accuracy.result(), step=epoch)
        train_total_loss.reset_states()
        train_accuracy.reset_states()
        ########## val
        val_dataset.on_epoch_end()
        loop = tqdm(enumerate(val_dataset), total =len(val_dataset))
        for step, (x_batch_tval, y_batch_val, image_path) in loop:
            val_step(x_batch_tval, y_batch_val, image_path)
            loop.set_description(f'Val Epoch [{epoch}/{cfg.train.epochs}]')
            loop.set_postfix(acc = val_accuracy.result().numpy() * 100, loss = val_total_loss.result().numpy())
        with val_summary_writer.as_default():
            tf.summary.scalar('val_total_loss', val_total_loss.result(), step=epoch)
            tf.summary.scalar('val_accuracy', val_accuracy.result(), step=epoch)
        val_total_loss.reset_states()
        val_accuracy.reset_states()
        if epoch % 1 == 0 and epoch>0:
            model_save(save_model_dir, model)
    model_save(save_model_dir, model) #保存训练的权值
    



if __name__ == '__main__':
    main()
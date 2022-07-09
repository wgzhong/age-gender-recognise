import tensorflow as tf
from data.pa100k import *
from utils import *
from tqdm import tqdm
import hydra
import numpy as np
import time

@hydra.main(config_path="./config/", config_name="config.yaml")
def test_datasest(cfg):
    model = tf.keras.models.load_model("./outputs/2022-07-02/01-59-14/save_model")
    model.summary()
    test_dataset = ImageSequence(cfg, "test")

    loop = tqdm(enumerate(test_dataset), total =len(test_dataset))
    total_acc=0.0
    for step, (x_batch_test, y_batch_test, image_path) in loop:
        logits = model(x_batch_test, training = False)
        gender = tf.where(logits[:,0]>0.5, 1, 0)
        p_age = tf.argmax(y_batch_test[0:,1:], axis=-1, output_type=tf.int32)
        age = tf.argmax(logits[0:,1:], axis=-1, output_type=tf.int32)
        a1 = tf.equal(gender, y_batch_test[:,0])
        a2 = tf.equal(age, p_age)
        a3 = np.mean(tf.cast(a1&a2, dtype="int32"))
        total_acc = total_acc + a3
        loop.set_description(f'test step [{step}/{(len(test_dataset)/cfg.train.batch_size)}]')
        loop.set_postfix(acc = a3 * 100)
    print("test avg acc is ", total_acc/len(test_dataset)*100)

def test_single_pic(pic_path):
    model = tf.keras.models.load_model("./outputs/2022-07-02/01-59-14/save_model")
    model.summary()
    start = time.time()
    img = cv2.imread(pic_path)
    img = data_resize(img)
    img = np.expand_dims(img, 0)
    logits = model(img, training = False)
    end = time.time()
    print("time is ", end-start)
    print(logits)
    cirte(logits)


if __name__ == '__main__':
    # test_datasest()
    test_single_pic("/home/wgzhong/pywork/datasets/pa100k/images/000464.jpg")



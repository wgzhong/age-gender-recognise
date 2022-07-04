from utils import *
import cv2
import re
import numpy as np
import time
import tensorflow as tf
from tqdm import tqdm

def run_datasets(root_path, model_path, images, labels):
    assert(len(labels)==len(images))
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    total_acc=0.0
    loop = tqdm(enumerate(zip(images, labels)), total =len(images))

    for step, (img_name, label) in loop:
        path= root_path+re.sub('\[|\]|\'','',img_name)
        img = cv2.imread(path)
        img = data_resize(img)
        img = np.expand_dims(img, 0).astype('float32')
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        logits = interpreter.get_tensor(output_details[0]['index'])
        gender = np.where(logits[:,0]>0.5, 1, 0)
        age = np.argmax(logits[0:,1:], axis=-1)
        p_age = np.argmax([int(label[2]), int(label[4]), int(label[6])], axis=-1)
        p_gender = int(label[0])
        a1 = np.equal(gender, p_gender)
        a2 = np.equal(age, p_age)
        if a1&a2:
            a3=1
        else:
            a3=0
        total_acc = total_acc + a3
        loop.set_description(f'test step [{step}/{(len(images))}]')
        loop.set_postfix(acc = a3 * 100)
    print("test avg acc is ", total_acc/len(images)*100)


def run_single_pic(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    start = time.time()
    pic_path="/home/wgzhong/pywork/datasets/pa100k/images/000464.jpg"
    img = cv2.imread(pic_path)
    img = data_resize(img)
    img = np.expand_dims(img, 0).astype('float32')
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()
    print("time is ", end-start)
    print(output_data)
    cirte(output_data)

if __name__ == '__main__':
    # run_single_pic("./mobilenetv2s_fp16.tflite")
    images = read_txt("/home/wgzhong/pywork/age-gender-recognise/data/pa100k/test_images_name.txt")
    labels = read_txt("/home/wgzhong/pywork/age-gender-recognise/data/pa100k/test_label.txt")
    run_datasets("/home/wgzhong/pywork/datasets/pa100k/images/", "./mobilenetv2s.tflite", images, labels)


from utils import *
import cv2
import numpy as np
import time
import tensorflow as tf

def run_tflite_fp32(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    start = time.time()
    pic_path="/home/vastai/zwg/pa100k/images/073388.jpg"
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
    run_tflite_fp32("./mobilenetv2s.tflite")

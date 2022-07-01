import tensorflow as tf

saved_model_dir = "/home/vastai/zwg/age-gender-recognise/outputs/2022-07-01/16-49-59/save_model"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
open("mobilenetv2s_fp16.tflite", "wb").write(tflite_quant_model)

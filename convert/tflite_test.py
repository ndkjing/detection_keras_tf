import numpy as np
import tensorflow as tf

"""
测试使用yolov4进行预测
"""
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter('yolov4.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the TensorFlow Lite model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
print('input_details,output_details',input_details,output_details)
tflite_results = interpreter.get_tensor(output_details[0]['index'])
print('tflite_results',tflite_results)
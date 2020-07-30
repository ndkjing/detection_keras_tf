import tensorflow as tf
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
import cv2
# Load TFLite model and allocate tensors.model_dynamic.tflite
interpreter = tf.lite.Interpreter(r'C:\PythonProject\model_dynamic.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the TensorFlow Lite model on random input data.
input_shape = input_details[0]['shape']

image_path = r'C:\PythonProject\aa86220c12d32594cae498ab57d21b57_00063.jpg'
image = Image.open(image_path)
image_pred = image.resize((input_shape[2],input_shape[2]), Image.ANTIALIAS)
image_pred = np.asarray(image_pred).astype('float32')
image_pred = np.expand_dims(image_pred,axis=0)
print(image_pred.shape)
start_time = time.time()
for i in range(100):
    interpreter.set_tensor(input_details[0]['index'], image_pred)
    interpreter.invoke()
    # print('input_details,output_details',input_details,output_details)
    tflite_results = interpreter.get_tensor(output_details[0]['index'])
print('检测花费时间',(time.time()-start_time)/100)
# print('tflite_results',tflite_results)
output_details = interpreter.get_output_details()[0]
out_tensor = np.squeeze(interpreter.get_tensor(output_details['index']))

boxes = out_tensor[:,:4]
classes = out_tensor[:,4]
scores = out_tensor[:,5]

results = []
for i in range(len(scores)):
    if scores[i] >= 0.5:
        result = {
            'bounding_box': boxes[i],
            'class_id': classes[i],
            'score': scores[i]
        }
        results.append(result)



labels = ['lighter', 'car', 'arm_end']
result_size = len(results)
show_image = cv2.imread(image_path)
for idx, obj in enumerate(results):
    print(obj)
    # Prepare image for drawing
    draw = ImageDraw.Draw(image)
    size = np.asarray(image).shape
    # Prepare boundary box
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * size[1])
    xmax = int(xmax * size[1])
    ymin = int(ymin * size[0])
    ymax = int(ymax * size[0])

    cv2.rectangle(show_image, (xmin, ymin), (xmax, ymax), (25, 0, 255), 4)

    # Annotate image with label and confidence score
    display_str = labels[int(obj['class_id'])] + ": " + str(round(obj['score'] * 100, 2)) + "%"


cv2.imshow('image',show_image)
cv2.waitKey(0)






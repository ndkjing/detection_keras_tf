# refer:https://github.com/hunglc007/tensorflow-yolov4-tflite

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from absl import app

import utils.yolov4.utils as utils
from models.yolo.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, decode
# from core.config import cfg
from configs.yolo.yolov4 import cfg

framework='tf'# '(tf, tflite')
weights_path='/Data/jing/weights/detection/keras_tf/yolo/pretrain/yolov4.weights'
                    #'path to weights file')
size=608                 #, 'resize images to')
tiny= False                 #, 'yolo or yolo-tiny')
model_name='yolov4'               #, 'yolov3 or yolov4')
image_path='../images_test/img.png'        #, 'path to input image')
image_write_path = os.path.join('images_out/', 'yolov4' + '.png')

def main(_argv):
    if tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, tiny)
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if model_name == 'yolov4':
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, tiny)
        else:
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_V3, tiny)
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    XYSCALE = cfg.YOLO.XYSCALE
    input_size = size

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    if framework == 'tf':
        input_layer = tf.keras.layers.Input([input_size, input_size, 3])
        if tiny:
            feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode(fm, NUM_CLASS, i)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)
            utils.load_weights_tiny(model, weights_path)
        else:
            if model_name == 'yolov3':
                feature_maps = YOLOv3(input_layer, NUM_CLASS)
                bbox_tensors = []
                for i, fm in enumerate(feature_maps):
                    bbox_tensor = decode(fm, NUM_CLASS, i)
                    bbox_tensors.append(bbox_tensor)
                model = tf.keras.Model(input_layer, bbox_tensors)
                utils.load_weights_v3(model, weights_path)
            elif model_name == 'yolov4':
                feature_maps = YOLOv4(input_layer, NUM_CLASS)
                bbox_tensors = []
                for i, fm in enumerate(feature_maps):
                    bbox_tensor = decode(fm, NUM_CLASS, i)
                    bbox_tensors.append(bbox_tensor)
                model = tf.keras.Model(input_layer, bbox_tensors)

                if weights_path.endswith("weights") :
                    utils.load_weights(model, weights_path)
                else:
                    model.load_weights(weights_path).expect_partial()

        model.summary()
        pred_bbox = model.predict(image_data)
    else:
        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=weights_path)
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
        pred_bbox = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

    if model == 'yolov4':
        pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
    else:
        pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
    bboxes = utils.nms(bboxes, 0.213, method='nms')

    image = utils.draw_bbox(original_image, bboxes)
    image = Image.fromarray(image)
    #image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_write_path, image)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

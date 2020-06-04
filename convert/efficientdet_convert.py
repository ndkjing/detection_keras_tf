# refer:https://github.com/xuannianz/EfficientDet.git
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))

import cv2
import json
import numpy as np
import os
import time
import glob

import tensorflow as tf

from models.efficientdet.build_efficientdet import efficientdet
from utils.efficientdet_utils import preprocess_image, postprocess_boxes
from utils.efficientdet_utils.draw_boxes import draw_boxes


# def read_images_file(file_path):
#     for image_path in glob.glob('datasets/VOC2007/JPEGImages/*.jpg'):
#         image = cv2.imread(image_path)
#         src_image = image.copy()
#         # BGR -> RGB
#         image = image[:, :, ::-1]
#         h, w = image.shape[:2]
#
#         image, scale = preprocess_image(image, image_size=image_size)
#         # run network
#         start = time.time()
#         boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
#         boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
#         print(time.time() - start)
#         boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)
#
#         # select indices which have a score above the threshold
#         indices = np.where(scores[:] > score_threshold)[0]
#
#         # select those detections
#         boxes = boxes[indices]
#         labels = labels[indices]
#
#         draw_boxes(src_image, boxes, scores, labels, colors, classes)
#
#         cv2.imwrite('./images_out/efficinetdet_img.png', src_image)
#         # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#         # cv2.imshow('image', src_image)
#         # cv2.waitKey(0)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    phi = 2
    weighted_bifpn = True
    model_path = '/Data/jing/weights/detection/keras_tf/efficientdet/pretrain/efficientdet-d%d.h5' % (phi)
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    # coco classes
    classes = {value['id'] - 1: value['name'] for value in json.load(open('../configs/coco_90.json', 'r')).values()}
    num_classes = 90
    score_threshold = 0.3
    colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
    _, model = efficientdet(phi=phi,
                            weighted_bifpn=weighted_bifpn,
                            num_classes=num_classes,
                            score_threshold=score_threshold)
    model.load_weights(model_path, by_name=True)
    model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TF Lite model.
    with tf.gfile.GFile('model.tflite', 'wb') as f:
        f.write(tflite_model)
    # image_path = '../images_test/img.png'
    # image = cv2.imread(image_path)
    # src_image = image.copy()
    # # BGR -> RGB
    # image = image[:, :, ::-1]
    # h, w = image.shape[:2]
    #
    # image, scale = preprocess_image(image, image_size=image_size)
    # # run network
    # start = time.time()
    # boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
    # boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
    # print(time.time() - start)
    # boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)
    #
    # # select indices which have a score above the threshold
    # indices = np.where(scores[:] > score_threshold)[0]
    #
    # # select those detections
    # boxes = boxes[indices]
    # labels = labels[indices]
    #
    # draw_boxes(src_image, boxes, scores, labels, colors, classes)
    # if not os.path.exists('images_out'):
    #     os.mkdir('images_out')
    # image_write_path = os.path.join('images_out/', 'efficientdet' + str(phi) + '.png')
    # cv2.imwrite(image_write_path, src_image)


if __name__ == '__main__':
    main()

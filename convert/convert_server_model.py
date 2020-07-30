# refer:https://github.com/xuannianz/EfficientDet.git
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf


def conver2():
    saved_model_obj = tf.saved_model.load(export_dir='/home/create/jing/tfmodels/tf_models2/research/object_detection/weights/arm/inference_graph/saved_model')
    print(saved_model_obj.signatures.keys())

    concrete_func = saved_model_obj.signatures['serving_default']
    concrete_func.inputs[0].set_shape([1, 512, 512, 3])

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()

    open("/home/create/jing/tfmodels/tf_models2/research/object_detection/weights/arm/inference_graph/efficientdetd0.tflite", "wb").write(tflite_model)

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
    conver2()

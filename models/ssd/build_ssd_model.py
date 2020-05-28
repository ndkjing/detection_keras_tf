'''
搭建ssd检测模型
'''

from __future__ import division
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, \
    Activation, Flatten, SeparableConv2D, AlphaDropout, GaussianDropout, Dropout, concatenate,ZeroPadding2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

from keras_layers.keras_layer_ssd_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_ssd_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_ssd_DecodeDetectionsFast import DecodeDetectionsFast
from backbone.ssd import build_base_ssd_7, build_base_vgg_ssd300, build_base_mb1_ssd300, build_base_mb2_ssd300




def build_ssd_7_model(image_size,
                      n_classes,
                      mode='training',
                      l2_regularization=0.0,
                      min_scale=0.1,
                      max_scale=0.9,
                      scales=None,
                      aspect_ratios_global=[0.5, 1.0, 2.0],
                      aspect_ratios_per_layer=None,
                      two_boxes_for_ar1=True,
                      steps=None,
                      offsets=None,
                      clip_boxes=False,
                      variances=[1.0, 1.0, 1.0, 1.0],
                      coords='centroids',
                      normalize_coords=False,
                      subtract_mean=[123, 117, 104],
                      divide_by_stddev=None,
                      swap_channels=False,
                      confidence_thresh=0.01,
                      iou_threshold=0.45,
                      top_k=200,
                      nms_max_output_size=400,
                      return_predictor_sizes=False):
    n_predictor_layers = 4  # The number of predictor conv layers in the network
    n_classes += 1  # Account for the background class.
    l2_reg = l2_regularization  # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                n_predictor_layers + 1, len(scales)))
    else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

    if len(variances) != 4:  # We need one variance value for each of the four box coordinates
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else:  # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack(
                [tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]],
                            tensor[..., swap_channels[3]]], axis=-1)

    ############################################################################
    # 构造骨干网络
    ############################################################################
    base_model = build_base_ssd_7(image_size,
                                  l2_regularization=l2_regularization,
                                  subtract_mean=None,
                                  divide_by_stddev=None,
                                  swap_channels=False, )

    # 搭建辅助卷积层
    # Build the convolutional predictor layers on top of conv layers 4, 5, 6, and 7.
    # We build two predictor layers on top of each of these layers: One for class prediction (classification), one for box coordinate prediction (localization)
    # We precidt `n_classes` confidence values for each box, hence the `classes` predictors have depth `n_boxes * n_classes`
    # We predict 4 box coordinates for each box, hence the `boxes` predictors have depth `n_boxes * 4`
    # Output shape of `classes`: `(batch, height, width, n_boxes * n_classes)`
    classes4 = Conv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes4')(base_model.get_layer("conv4").output)
    classes5 = Conv2D(n_boxes[1] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes5')(base_model.get_layer("conv5").output)
    classes6 = Conv2D(n_boxes[2] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes6')(base_model.get_layer("conv6").output)
    classes7 = Conv2D(n_boxes[3] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes7')(base_model.get_layer("conv7").output)
    # Output shape of `boxes`: `(batch, height, width, n_boxes * 4)`
    boxes4 = Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes4')(base_model.get_layer("conv4").output)
    boxes5 = Conv2D(n_boxes[1] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes5')(base_model.get_layer("conv5").output)
    boxes6 = Conv2D(n_boxes[2] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes6')(base_model.get_layer("conv6").output)
    boxes7 = Conv2D(n_boxes[3] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes7')(base_model.get_layer("conv7").output)

    # Generate the anchor boxes
    # Output shape of `anchors`: `(batch, height, width, n_boxes, 8)`
    anchors4 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                           aspect_ratios=aspect_ratios[0],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords,
                           name='anchors4')(boxes4)
    anchors5 = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2],
                           aspect_ratios=aspect_ratios[1],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords,
                           name='anchors5')(boxes5)
    anchors6 = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3],
                           aspect_ratios=aspect_ratios[2],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords,
                           name='anchors6')(boxes6)
    anchors7 = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4],
                           aspect_ratios=aspect_ratios[3],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords,
                           name='anchors7')(boxes7)

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    classes4_reshaped = Reshape((-1, n_classes), name='classes4_reshape')(classes4)
    classes5_reshaped = Reshape((-1, n_classes), name='classes5_reshape')(classes5)
    classes6_reshaped = Reshape((-1, n_classes), name='classes6_reshape')(classes6)
    classes7_reshaped = Reshape((-1, n_classes), name='classes7_reshape')(classes7)
    # Reshape the box coordinate predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    boxes4_reshaped = Reshape((-1, 4), name='boxes4_reshape')(boxes4)
    boxes5_reshaped = Reshape((-1, 4), name='boxes5_reshape')(boxes5)
    boxes6_reshaped = Reshape((-1, 4), name='boxes6_reshape')(boxes6)
    boxes7_reshaped = Reshape((-1, 4), name='boxes7_reshape')(boxes7)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    anchors4_reshaped = Reshape((-1, 8), name='anchors4_reshape')(anchors4)
    anchors5_reshaped = Reshape((-1, 8), name='anchors5_reshape')(anchors5)
    anchors6_reshaped = Reshape((-1, 8), name='anchors6_reshape')(anchors6)
    anchors7_reshaped = Reshape((-1, 8), name='anchors7_reshape')(anchors7)

    # Concatenate the predictions from the different layers and the assosciated anchor box tensors
    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1
    # Output shape of `classes_concat`: (batch, n_boxes_total, n_classes)
    classes_concat = Concatenate(axis=1, name='classes_concat')([classes4_reshaped,
                                                                 classes5_reshaped,
                                                                 classes6_reshaped,
                                                                 classes7_reshaped])

    # Output shape of `boxes_concat`: (batch, n_boxes_total, 4)
    boxes_concat = Concatenate(axis=1, name='boxes_concat')([boxes4_reshaped,
                                                             boxes5_reshaped,
                                                             boxes6_reshaped,
                                                             boxes7_reshaped])

    # Output shape of `anchors_concat`: (batch, n_boxes_total, 8)
    anchors_concat = Concatenate(axis=1, name='anchors_concat')([anchors4_reshaped,
                                                                 anchors5_reshaped,
                                                                 anchors6_reshaped,
                                                                 anchors7_reshaped])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)

    # Concatenate the class and box coordinate predictions and the anchors to one large predictions tensor
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([classes_softmax, boxes_concat, anchors_concat])

    if mode == 'training':
        model = Model(inputs=base_model.input, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=base_model.input, outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=base_model.input, outputs=decoded_predictions)
    else:
        raise ValueError(
            "`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        # The spatial dimensions are the same for the `classes` and `boxes` predictor layers.
        predictor_sizes = np.array([classes4._keras_shape[1:3],
                                    classes5._keras_shape[1:3],
                                    classes6._keras_shape[1:3],
                                    classes7._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model


def build_vgg_ssd300_model(image_size,
                           n_classes,
                           mode='training',
                           l2_regularization=0.0005,
                           min_scale=None,
                           max_scale=None,
                           scales=None,
                           aspect_ratios_global=None,
                           aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                    [1.0, 2.0, 0.5],
                                                    [1.0, 2.0, 0.5]],
                           two_boxes_for_ar1=True,
                           steps=[8, 16, 32, 64, 100, 300],
                           offsets=None,
                           clip_boxes=False,
                           variances=[0.1, 0.1, 0.2, 0.2],
                           coords='centroids',
                           normalize_coords=True,
                           subtract_mean=[123, 117, 104],
                           divide_by_stddev=None,
                           swap_channels=[2, 1, 0],
                           confidence_thresh=0.01,
                           iou_threshold=0.45,
                           top_k=200,
                           nms_max_output_size=400,
                           return_predictor_sizes=False):
    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack(
                [tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]],
                            tensor[..., swap_channels[3]]], axis=-1)

    # 用于预测的多特征层数
    n_predictor_layers = 6  # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1  # Account for the background class.
    l2_reg = l2_regularization  # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################
    # 先验框比例
    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    # default box相对于图片大小尺寸
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                n_predictor_layers + 1, len(scales)))
    else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)
    # max default box权重
    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []  # 各特征图prior box个数
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:  # 是否添加大于当前特征图的1:1候选框
                n_boxes.append(len(ar) + 1)  # 两个比例为1 的default box
            else:
                n_boxes.append(len(ar))
    else:  # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    # 构建基础网络
    x = Input(shape=image_size)

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(
            x1)

    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv1_1',trainable=True)(x1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=l2(l2_reg), name='fc6')(pool5)

    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=l2(l2_reg), name='fc7')(fc6)

    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv6_1')(fc7)
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv6_2')(conv6_1)

    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv7_1')(conv6_2)
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv7_2')(conv7_1)

    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv8_2')(conv8_1)

    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv9_2')(conv9_1)
    # base_mdoel = build_base_vgg_ssd300(image_size=image_size, input_x=input_x, l2_regularization=l2_regularization)

    # # 构建辅助卷积层和输出层
    # conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
    #                                 kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_conf')(
    #     base_mdoel.get_layer("conv4_3").output)
    # fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
    #                        kernel_regularizer=l2(l2_reg), name='fc7_mbox_conf')(base_mdoel.get_layer("fc7").output)
    # conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
    #                            kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_conf')(
    #     base_mdoel.get_layer("conv6_2").output)
    # conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
    #                            kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_conf')(
    #     base_mdoel.get_layer("conv7_2").output)
    # conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
    #                            kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf')(
    #     base_mdoel.get_layer("conv8_2").output)
    # conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
    #                            kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf')(
    #     base_mdoel.get_layer("conv9_2").output)
    # # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    # conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
    #                                kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_loc')(
    #     base_mdoel.get_layer("conv4_3").output)
    # fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
    #                       kernel_regularizer=l2(l2_reg), name='fc7_mbox_loc')(base_mdoel.get_layer("fc7").output)
    # conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
    #                           kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_loc')(
    #     base_mdoel.get_layer("conv6_2").output)
    # conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
    #                           kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_loc')(
    #     base_mdoel.get_layer("conv7_2").output)
    # conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
    #                           kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_loc')(
    #     base_mdoel.get_layer("conv8_2").output)
    # conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
    #                           kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_loc')(
    #     base_mdoel.get_layer("conv9_2").output)

    # 构建辅助卷积层和输出层
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_conf')(conv4_3)
    fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=l2(l2_reg), name='fc7_mbox_conf')(fc7)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf')(conv9_2)
    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_loc')(conv4_3)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(l2_reg), name='fc7_mbox_loc')(fc7)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_loc')(conv9_2)



    ### Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)

    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                                             aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0],
                                             this_offsets=offsets[0], clip_boxes=clip_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords,
                                             name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)
    fc7_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2],
                                    aspect_ratios=aspect_ratios[1],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                                    clip_boxes=clip_boxes,
                                    variances=variances, coords=coords, normalize_coords=normalize_coords,
                                    name='fc7_mbox_priorbox')(fc7_mbox_loc)
    conv6_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3],
                                        aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2],
                                        this_offsets=offsets[2], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
    conv7_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4],
                                        aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3],
                                        this_offsets=offsets[3], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
    conv8_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5],
                                        aspect_ratios=aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4],
                                        this_offsets=offsets[4], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
    conv9_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6],
                                        aspect_ratios=aspect_ratios[5],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5],
                                        this_offsets=offsets[5], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)

    ### Reshape

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='conv4_3_norm_mbox_conf_reshape')(
        conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(
        conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                               fc7_mbox_priorbox_reshape,
                                                               conv6_2_mbox_priorbox_reshape,
                                                               conv7_2_mbox_priorbox_reshape,
                                                               conv8_2_mbox_priorbox_reshape,
                                                               conv9_2_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    if mode == 'training':
        model = Model(inputs=x, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        raise ValueError(
            "`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        predictor_sizes = np.array([conv4_3_norm_mbox_conf._keras_shape[1:3],
                                    fc7_mbox_conf._keras_shape[1:3],
                                    conv6_2_mbox_conf._keras_shape[1:3],
                                    conv7_2_mbox_conf._keras_shape[1:3],
                                    conv8_2_mbox_conf._keras_shape[1:3],
                                    conv9_2_mbox_conf._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model


def build_mb1_ssd300_model(image_size,
                           n_classes,
                           mode='training',
                           l2_regularization=0.0005,
                           min_scale=None,
                           max_scale=None,
                           scales=None,
                           aspect_ratios_global=None,
                           aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                    [1.0, 2.0, 0.5],
                                                    [1.0, 2.0, 0.5]],
                           two_boxes_for_ar1=True,
                           steps=[8, 16, 32, 64, 100, 300],
                           offsets=None,
                           clip_boxes=False,
                           variances=[0.1, 0.1, 0.2, 0.2],
                           coords='centroids',
                           normalize_coords=True,
                           subtract_mean=[123, 117, 104],
                           divide_by_stddev=None,
                           swap_channels=[2, 1, 0],
                           confidence_thresh=0.01,
                           iou_threshold=0.45,
                           top_k=200,
                           nms_max_output_size=400,
                           return_predictor_sizes=False):
    # 用于预测的多特征层数
    n_predictor_layers = 6  # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1  # Account for the background class.
    l2_reg = l2_regularization  # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################
    # 先验框比例
    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    # default box相对于图片大小尺寸
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                n_predictor_layers + 1, len(scales)))
    else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)
    # max default box权重
    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []  # 各特征图prior box个数
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:  # 是否添加大于当前特征图的1:1候选框
                n_boxes.append(len(ar) + 1)  # 两个比例为1 的default box
            else:
                n_boxes.append(len(ar))
    else:  # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    # 构建基础网络
    mb1_basenet = build_base_mb1_ssd300(image_size=image_size, l2_regularization=l2_regularization)

    base_model = Model(inputs=mb1_basenet.input, outputs=mb1_basenet.get_layer('conv_dw_11_relu').output)
    base_model.summary()
    # 构建辅助卷积层和输出层
    net = {}
    net['input'] = Input(shape=image_size)
    net['mobilenet_conv_dw_11_relu'] = base_model(net['input'])
    net['conv11'] = Conv2D(512, (1, 1), padding='same', name='conv11',trainable=True)(net['mobilenet_conv_dw_11_relu'])
    net['conv11'] = BatchNormalization(momentum=0.99, name='bn11')(net['conv11'])
    net['conv11'] = Activation('relu')(net['conv11'])
    # Block
    # (19,19)
    net['conv12dw'] = SeparableConv2D(512, (3, 3), strides=(2, 2), padding='same', name='conv12dw')(net['conv11'])
    net['conv12dw'] = BatchNormalization(momentum=0.99, name='bn12dw')(net['conv12dw'])
    net['conv12dw'] = Activation('relu')(net['conv12dw'])
    net['conv12'] = Conv2D(1024, (1, 1), padding='same', name='conv12')(net['conv12dw'])
    net['conv12'] = BatchNormalization(momentum=0.99, name='bn12')(net['conv12'])
    net['conv12'] = Activation('relu')(net['conv12'])
    net['conv13dw'] = SeparableConv2D(1024, (3, 3), padding='same', name='conv13dw')(net['conv12'])
    net['conv13dw'] = BatchNormalization(momentum=0.99, name='bn13dw')(net['conv13dw'])
    net['conv13dw'] = Activation('relu')(net['conv13dw'])
    net['conv13'] = Conv2D(1024, (1, 1), padding='same', name='conv13')(net['conv13dw'])
    net['conv13'] = BatchNormalization(momentum=0.99, name='bn13')(net['conv13'])
    net['conv13'] = Activation('relu')(net['conv13'])
    net['conv14_1'] = Conv2D(256, (1, 1), padding='same', name='conv14_1')(net['conv13'])
    net['conv14_1'] = BatchNormalization(momentum=0.99, name='bn14_1')(net['conv14_1'])
    net['conv14_1'] = Activation('relu')(net['conv14_1'])
    net['conv14_2'] = Conv2D(512, (3, 3), strides=(2, 2), padding='same', name='conv14_2')(net['conv14_1'])
    net['conv14_2'] = BatchNormalization(momentum=0.99, name='bn14_2')(net['conv14_2'])
    net['conv14_2'] = Activation('relu')(net['conv14_2'])
    net['conv15_1'] = Conv2D(128, (1, 1), padding='same', name='conv15_1')(net['conv14_2'])
    net['conv15_1'] = BatchNormalization(momentum=0.99, name='bn15_1')(net['conv15_1'])
    net['conv15_1'] = Activation('relu')(net['conv15_1'])
    net['conv15_2'] = Conv2D(256, (3, 3), strides=(2, 2), padding='same', name='conv15_2')(net['conv15_1'])
    net['conv15_2'] = BatchNormalization(momentum=0.99, name='bn15_2')(net['conv15_2'])
    net['conv15_2'] = Activation('relu')(net['conv15_2'])
    net['conv16_1'] = Conv2D(128, (1, 1), padding='same', name='conv16_1')(net['conv15_2'])
    net['conv16_1'] = BatchNormalization(momentum=0.99, name='bn16_1')(net['conv16_1'])
    net['conv16_1'] = Activation('relu')(net['conv16_1'])
    net['conv16_2'] = Conv2D(256, (3, 3), strides=(2, 2), padding='same', name='conv16_2')(net['conv16_1'])
    net['conv16_2'] = BatchNormalization(momentum=0.99, name='bn16_2')(net['conv16_2'])
    net['conv16_2'] = Activation('relu')(net['conv16_2'])
    net['conv17_1'] = Conv2D(64, (1, 1), padding='same', name='conv17_1')(net['conv16_2'])
    net['conv17_1'] = BatchNormalization(momentum=0.99, name='bn17_1')(net['conv17_1'])
    net['conv17_1'] = Activation('relu')(net['conv17_1'])
    net['conv17_2'] = Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='conv17_2')(net['conv17_1'])
    net['conv17_2'] = BatchNormalization(momentum=0.99, name='bn17_2')(net['conv17_2'])
    net['conv17_2'] = Activation('relu')(net['conv17_2'])

    # Prediction from conv11
    net['conv11_mbox_loc'] = Conv2D(n_boxes[0] * 4, (1, 1), padding='same', name='conv11_mbox_loc')(net['conv11'])
    net['conv11_mbox_conf'] = Conv2D(n_boxes[0] * n_classes, (1, 1), padding='same', name='conv11_mbox_conf')(
        net['conv11'])

    # Prediction from conv13
    net['conv13_mbox_loc'] = Conv2D(n_boxes[1] * 4, (1, 1), padding='same', name='conv13_mbox_loc')(net['conv13'])
    net['conv13_mbox_conf'] = Conv2D(n_boxes[1] * n_classes, (1, 1), padding='same', name='conv13_mbox_conf')(
        net['conv13'])

    # Prediction from conv14_2
    net['conv14_2_mbox_loc'] = Conv2D(n_boxes[2] * 4, (1, 1), padding='same', name='conv14_2_mbox_loc')(net['conv14_2'])
    net['conv14_2_mbox_conf'] = Conv2D(n_boxes[2] * n_classes, (1, 1), padding='same', name='conv14_2_mbox_conf')(
        net['conv14_2'])

    # Prediction from conv15_2
    net['conv15_2_mbox_loc'] = Conv2D(n_boxes[3] * 4, (1, 1), padding='same', name='conv15_2_mbox_loc')(net['conv15_2'])
    net['conv15_2_mbox_conf'] = Conv2D(n_boxes[3] * n_classes, (1, 1), padding='same', name='conv15_2_mbox_conf')(
        net['conv15_2'])

    # Prediction from conv16_2
    net['conv16_2_mbox_loc'] = Conv2D(n_boxes[4] * 4, (1, 1), padding='same', name='conv16_2_mbox_loc')(net['conv16_2'])
    net['conv16_2_mbox_conf'] = Conv2D(n_boxes[4] * n_classes, (1, 1), padding='same', name='conv16_2_mbox_conf')(
        net['conv16_2'])

    # Prediction from conv17_2
    net['conv17_2_mbox_loc'] = Conv2D(n_boxes[5] * 4, (1, 1), padding='same', name='conv17_2_mbox_loc')(net['conv17_2'])
    net['conv17_2_mbox_conf'] = Conv2D(n_boxes[5] * n_classes, (1, 1), padding='same', name='conv17_2_mbox_conf')(
        net['conv17_2'])

    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    net["conv11_norm_mbox_priorbox"] = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                                                   aspect_ratios=aspect_ratios[0],
                                                   two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0],
                                                   this_offsets=offsets[0], clip_boxes=clip_boxes,
                                                   variances=variances, coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   name='conv4_3_norm_mbox_priorbox')(net['conv11_mbox_loc'])
    net["conv13_norm_mbox_priorbox"] = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2],
                                                   aspect_ratios=aspect_ratios[1],
                                                   two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1],
                                                   this_offsets=offsets[1],
                                                   clip_boxes=clip_boxes,
                                                   variances=variances, coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   name='fc7_mbox_priorbox')(net['conv13_mbox_loc'])
    net["conv14_2_norm_mbox_priorbox"] = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3],
                                                     aspect_ratios=aspect_ratios[2],
                                                     two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2],
                                                     this_offsets=offsets[2], clip_boxes=clip_boxes,
                                                     variances=variances, coords=coords,
                                                     normalize_coords=normalize_coords,
                                                     name='conv6_2_mbox_priorbox')(net['conv14_2_mbox_loc'])
    net["conv15_2_norm_mbox_priorbox"] = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4],
                                                     aspect_ratios=aspect_ratios[3],
                                                     two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3],
                                                     this_offsets=offsets[3], clip_boxes=clip_boxes,
                                                     variances=variances, coords=coords,
                                                     normalize_coords=normalize_coords,
                                                     name='conv15_2_mbox_priorbox')(net['conv15_2_mbox_loc'])
    net["conv16_2_norm_mbox_priorbox"] = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5],
                                                     aspect_ratios=aspect_ratios[4],
                                                     two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4],
                                                     this_offsets=offsets[4], clip_boxes=clip_boxes,
                                                     variances=variances, coords=coords,
                                                     normalize_coords=normalize_coords,
                                                     name='conv16_2_mbox_priorbox')(net['conv16_2_mbox_loc'])
    net["conv17_2_norm_mbox_priorbox"] = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6],
                                                     aspect_ratios=aspect_ratios[5],
                                                     two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5],
                                                     this_offsets=offsets[5], clip_boxes=clip_boxes,
                                                     variances=variances, coords=coords,
                                                     normalize_coords=normalize_coords,
                                                     name='conv17_2_mbox_priorbox')(net['conv17_2_mbox_loc'])

    ### Reshape
    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    net['conv11_mbox_conf_reshape'] = Reshape((-1, n_classes), name='conv11_norm_mbox_conf_reshape')(
        net['conv11_mbox_conf'])
    net['conv13_mbox_conf_reshape'] = Reshape((-1, n_classes), name='conv13_mbox_conf_reshape')(net['conv13_mbox_conf'])
    net['conv14_2_mbox_conf_reshape'] = Reshape((-1, n_classes), name='conv14_2_mbox_conf_reshape')(
        net['conv14_2_mbox_conf'])
    net['conv15_2_mbox_conf_reshape'] = Reshape((-1, n_classes), name='conv15_2_mbox_conf_reshape')(
        net['conv15_2_mbox_conf'])
    net['conv16_2_mbox_conf_reshape'] = Reshape((-1, n_classes), name='conv16_2_mbox_conf_reshape')(
        net['conv16_2_mbox_conf'])
    net['conv17_2_mbox_conf_reshape'] = Reshape((-1, n_classes), name='conv17_2_mbox_conf_reshape')(
        net['conv17_2_mbox_conf'])

    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    net['conv11_mbox_loc_reshape'] = Reshape((-1, 4), name='conv11_norm_mbox_loc_reshape')(net['conv11_mbox_loc'])
    net['conv13_mbox_loc_reshape'] = Reshape((-1, 4), name='conv13_mbox_loc_reshape')(net['conv13_mbox_loc'])
    net['conv14_2_mbox_loc_reshape'] = Reshape((-1, 4), name='conv14_2_mbox_loc_reshape')(net['conv14_2_mbox_loc'])
    net['conv15_2_mbox_loc_reshape'] = Reshape((-1, 4), name='conv15_2_mbox_loc_reshape')(net['conv15_2_mbox_loc'])
    net['conv16_2_mbox_loc_reshape'] = Reshape((-1, 4), name='conv16_2_mbox_loc_reshape')(net['conv16_2_mbox_loc'])
    net['conv17_2_mbox_loc_reshape'] = Reshape((-1, 4), name='conv17_2_mbox_loc_reshape')(net['conv17_2_mbox_loc'])

    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    net["conv11_norm_mbox_priorbox_reshape"] = Reshape((-1, 8), name='conv11_norm_mbox_priorbox_reshape')(
        net["conv11_norm_mbox_priorbox"])
    net["conv13_norm_mbox_priorbox_reshape"] = Reshape((-1, 8), name='conv13_norm_mbox_priorbox_reshape')(
        net["conv13_norm_mbox_priorbox"])
    net["conv14_2_norm_mbox_priorbox_reshape"] = Reshape((-1, 8), name='conv14_2_norm_mbox_priorbox_reshape')(
        net["conv14_2_norm_mbox_priorbox"])
    net["conv15_2_norm_mbox_priorbox_reshape"] = Reshape((-1, 8), name='conv15_2_norm_mbox_priorbox_reshape')(
        net["conv15_2_norm_mbox_priorbox"])
    net["conv16_2_norm_mbox_priorbox_reshape"] = Reshape((-1, 8), name='conv16_2_norm_mbox_priorbox_reshape')(
        net["conv16_2_norm_mbox_priorbox"])
    net["conv17_2_norm_mbox_priorbox_reshape"] = Reshape((-1, 8), name='conv17_2_norm_mbox_priorbox_reshape')(
        net["conv17_2_norm_mbox_priorbox"])
    ### Concatenate the predictions from the different layers
    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([net['conv11_mbox_conf_reshape'],
                                                       net['conv13_mbox_conf_reshape'],
                                                       net['conv14_2_mbox_conf_reshape'],
                                                       net['conv15_2_mbox_conf_reshape'],
                                                       net['conv16_2_mbox_conf_reshape'],
                                                       net['conv17_2_mbox_conf_reshape']])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([net['conv11_mbox_loc_reshape'],
                                                     net['conv13_mbox_loc_reshape'],
                                                     net['conv14_2_mbox_loc_reshape'],
                                                     net['conv15_2_mbox_loc_reshape'],
                                                     net['conv16_2_mbox_loc_reshape'],
                                                     net['conv17_2_mbox_loc_reshape']])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([net["conv11_norm_mbox_priorbox_reshape"],
                                                               net["conv13_norm_mbox_priorbox_reshape"],
                                                               net["conv14_2_norm_mbox_priorbox_reshape"],
                                                               net["conv15_2_norm_mbox_priorbox_reshape"],
                                                               net["conv16_2_norm_mbox_priorbox_reshape"],
                                                               net["conv17_2_norm_mbox_priorbox_reshape"]])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    if mode == 'training':
        model = Model(inputs=net['input'], outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=net['input'], outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=base_model.input, outputs=decoded_predictions)
    else:
        raise ValueError(
            "`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        predictor_sizes = np.array([net['conv11_mbox_conf']._keras_shape[1:3],
                                    net['conv13_mbox_conf']._keras_shape[1:3],
                                    net['conv14_2_mbox_conf']._keras_shape[1:3],
                                    net['conv15_2_mbox_conf']._keras_shape[1:3],
                                    net['conv16_2_mbox_conf']._keras_shape[1:3],
                                    net['conv17_2_mbox_conf']._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model


def build_mb2_ssd300_model(image_size,
                           n_classes,
                           mode='training',
                           l2_regularization=0.0005,
                           min_scale=None,
                           max_scale=None,
                           scales=None,
                           aspect_ratios_global=None,
                           aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                    [1.0, 2.0, 0.5],
                                                    [1.0, 2.0, 0.5]],
                           two_boxes_for_ar1=True,
                           steps=[8, 16, 32, 64, 100, 300],
                           offsets=None,
                           clip_boxes=False,
                           variances=[0.1, 0.1, 0.2, 0.2],
                           coords='centroids',
                           normalize_coords=True,
                           subtract_mean=[123, 117, 104],
                           divide_by_stddev=None,
                           swap_channels=[2, 1, 0],
                           confidence_thresh=0.01,
                           iou_threshold=0.45,
                           top_k=200,
                           nms_max_output_size=400,
                           return_predictor_sizes=False):
    # 用于预测的多特征层数
    n_predictor_layers = 6  # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1  # Account for the background class.
    l2_reg = l2_regularization  # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################
    # 先验框比例
    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    # default box相对于图片大小尺寸
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                n_predictor_layers + 1, len(scales)))
    else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)
    # max default box权重
    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []  # 各特征图prior box个数
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:  # 是否添加大于当前特征图的1:1候选框
                n_boxes.append(len(ar) + 1)  # 两个比例为1 的default box
            else:
                n_boxes.append(len(ar))
    else:  # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    # 构建基础网络
    # mb2_basenet = build_base_mb2_ssd300(image_size=image_size, l2_regularization=l2_regularization)
    # mb2_basenet.summary()
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]
    l2_reg = l2_regularization
    mobilenet_input_shape = (224, 224, 3)
    net = {}
    base_mb2 = MobileNetV2(input_shape=mobilenet_input_shape, include_top=False, weights='imagenet')
    base_model = Model(inputs=base_mb2.input, outputs=base_mb2.get_layer('block_13_expand_relu').output)
    base_model.summary()
    # 构建辅助卷积层和输出层
    net = {}
    input_x = Input(shape=image_size)
    mobilenet_conv_dw_11_relu = base_model(input_x)
    conv11 = Conv2D(512, (1, 1), padding='same', name='conv11')(mobilenet_conv_dw_11_relu)
    conv11_bn = BatchNormalization(momentum=0.99, name='bn11')(conv11)
    conv11_re = Activation('relu')(conv11_bn)
    # Block
    # (19,19)
    conv12dw = SeparableConv2D(512, (3, 3), strides=(2, 2), padding='same', name='conv12dw')(conv11_re)
    conv12dw = BatchNormalization(momentum=0.99, name='bn12dw')(conv12dw)
    conv12dw = Activation('relu')(conv12dw)
    conv12 = Conv2D(1024, (1, 1), padding='same', name='conv12')(conv12dw)
    conv12 = BatchNormalization(momentum=0.99, name='bn12')(conv12)
    conv12 = Activation('relu')(conv12)
    conv13dw = SeparableConv2D(1024, (3, 3), padding='same', name='conv13dw')(conv12)
    conv13dw = BatchNormalization(momentum=0.99, name='bn13dw')(conv13dw)
    conv13dw = Activation('relu')(conv13dw)
    conv13 = Conv2D(1024, (1, 1), padding='same', name='conv13')(conv13dw)
    conv13 = BatchNormalization(momentum=0.99, name='bn13')(conv13)
    conv13 = Activation('relu')(conv13)
    conv14_1 = Conv2D(256, (1, 1), padding='same', name='conv14_1')(conv13)
    conv14_1 = BatchNormalization(momentum=0.99, name='bn14_1')(conv14_1)
    conv14_1 = Activation('relu')(conv14_1)
    conv14_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', name='conv14_2')(conv14_1)
    conv14_2 = BatchNormalization(momentum=0.99, name='bn14_2')(conv14_2)
    conv14_2 = Activation('relu')(conv14_2)
    conv15_1 = Conv2D(128, (1, 1), padding='same', name='conv15_1')(conv14_2)
    conv15_1 = BatchNormalization(momentum=0.99, name='bn15_1')(conv15_1)
    conv15_1 = Activation('relu')(conv15_1)
    conv15_2 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', name='conv15_2')(conv15_1)
    conv15_2 = BatchNormalization(momentum=0.99, name='bn15_2')(conv15_2)
    conv15_2 = Activation('relu')(conv15_2)
    conv16_1 = Conv2D(128, (1, 1), padding='same', name='conv16_1')(conv15_2)
    conv16_1 = BatchNormalization(momentum=0.99, name='bn16_1')(conv16_1)
    conv16_1 = Activation('relu')(conv16_1)
    conv16_2 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', name='conv16_2')(conv16_1)
    nconv16_2 = BatchNormalization(momentum=0.99, name='bn16_2')(conv16_2)
    conv16_2 = Activation('relu')(conv16_2)
    conv17_1 = Conv2D(64, (1, 1), padding='same', name='conv17_1')(conv16_2)
    conv17_1 = BatchNormalization(momentum=0.99, name='bn17_1')(conv17_1)
    conv17_1 = Activation('relu')(conv17_1)
    conv17_2 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='conv17_2')(conv17_1)
    conv17_2 = BatchNormalization(momentum=0.99, name='bn17_2')(conv17_2)
    conv17_2 = Activation('relu')(conv17_2)

    # Prediction from conv11
    conv11_mbox_loc = Conv2D(n_boxes[0] * 4, (1, 1), padding='same', name='conv11_mbox_loc')(conv11)
    conv13_mbox_loc = Conv2D(n_boxes[1] * 4, (1, 1), padding='same', name='conv13_mbox_loc')(conv13)
    conv14_2_mbox_loc = Conv2D(n_boxes[2] * 4, (1, 1), padding='same', name='conv14_2_mbox_loc')(conv14_2)
    conv15_2_mbox_loc = Conv2D(n_boxes[3] * 4, (1, 1), padding='same', name='conv15_2_mbox_loc')(conv15_2)
    conv16_2_mbox_loc = Conv2D(n_boxes[4] * 4, (1, 1), padding='same', name='conv16_2_mbox_loc')(conv16_2)
    conv17_2_mbox_loc = Conv2D(n_boxes[5] * 4, (1, 1), padding='same', name='conv17_2_mbox_loc')(conv17_2)


    conv11_mbox_conf = Conv2D(n_boxes[0] * n_classes, (1, 1), padding='same', name='conv11_mbox_conf')(conv11)
    conv13_mbox_conf = Conv2D(n_boxes[1] * n_classes, (1, 1), padding='same', name='conv13_mbox_conf')(conv13)
    conv14_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (1, 1), padding='same', name='conv14_2_mbox_conf')(conv14_2)
    conv15_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (1, 1), padding='same', name='conv15_2_mbox_conf')(conv15_2)
    conv16_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (1, 1), padding='same', name='conv16_2_mbox_conf')(conv16_2)
    conv17_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (1, 1), padding='same', name='conv17_2_mbox_conf')(conv17_2)


    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    conv11_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                                            aspect_ratios=aspect_ratios[0],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0],
                                            this_offsets=offsets[0], clip_boxes=clip_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords,
                                            name='conv4_3_norm_mbox_priorbox')(conv11_mbox_loc)
    conv13_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2],
                                            aspect_ratios=aspect_ratios[1],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1],
                                            this_offsets=offsets[1],
                                            clip_boxes=clip_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords,
                                            name='fc7_mbox_priorbox')(conv13_mbox_loc)
    conv14_2_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3],
                                              aspect_ratios=aspect_ratios[2],
                                              two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2],
                                              this_offsets=offsets[2], clip_boxes=clip_boxes,
                                              variances=variances, coords=coords, normalize_coords=normalize_coords,
                                              name='conv6_2_mbox_priorbox')(conv14_2_mbox_loc)
    conv15_2_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4],
                                              aspect_ratios=aspect_ratios[3],
                                              two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3],
                                              this_offsets=offsets[3], clip_boxes=clip_boxes,
                                              variances=variances, coords=coords, normalize_coords=normalize_coords,
                                              name='conv15_2_mbox_priorbox')(conv15_2_mbox_loc)
    conv16_2_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5],
                                              aspect_ratios=aspect_ratios[4],
                                              two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4],
                                              this_offsets=offsets[4], clip_boxes=clip_boxes,
                                              variances=variances, coords=coords, normalize_coords=normalize_coords,
                                              name='conv16_2_mbox_priorbox')(conv16_2_mbox_loc)
    conv17_2_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6],
                                              aspect_ratios=aspect_ratios[5],
                                              two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5],
                                              this_offsets=offsets[5], clip_boxes=clip_boxes,
                                              variances=variances, coords=coords, normalize_coords=normalize_coords,
                                              name='conv17_2_mbox_priorbox')(conv17_2_mbox_loc)

    ### Reshape
    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv11_mbox_conf_reshape = Reshape((-1, n_classes), name='conv11_norm_mbox_conf_reshape')(conv11_mbox_conf)
    conv13_mbox_conf_reshape = Reshape((-1, n_classes), name='conv13_mbox_conf_reshape')(conv13_mbox_conf)
    conv14_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv14_2_mbox_conf_reshape')(conv14_2_mbox_conf)
    conv15_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv15_2_mbox_conf_reshape')(conv15_2_mbox_conf)
    conv16_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv16_2_mbox_conf_reshape')(conv16_2_mbox_conf)
    conv17_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv17_2_mbox_conf_reshape')(conv17_2_mbox_conf)

    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv11_mbox_loc_reshape = Reshape((-1, 4), name='conv11_norm_mbox_loc_reshape')(conv11_mbox_loc)
    conv13_mbox_loc_reshape = Reshape((-1, 4), name='conv13_mbox_loc_reshape')(conv13_mbox_loc)
    conv14_2_mbox_loc_reshape = Reshape((-1, 4), name='conv14_2_mbox_loc_reshape')(conv14_2_mbox_loc)
    conv15_2_mbox_loc_reshape = Reshape((-1, 4), name='conv15_2_mbox_loc_reshape')(conv15_2_mbox_loc)
    conv16_2_mbox_loc_reshape = Reshape((-1, 4), name='conv16_2_mbox_loc_reshape')(conv16_2_mbox_loc)
    conv17_2_mbox_loc_reshape = Reshape((-1, 4), name='conv17_2_mbox_loc_reshape')(conv17_2_mbox_loc)

    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv11_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv11_norm_mbox_priorbox_reshape')(
        conv11_norm_mbox_priorbox)
    conv13_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv13_norm_mbox_priorbox_reshape')(
        conv13_norm_mbox_priorbox)
    conv14_2_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv14_2_norm_mbox_priorbox_reshape')(
        conv14_2_norm_mbox_priorbox)
    conv15_2_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv15_2_norm_mbox_priorbox_reshape')(
        conv15_2_norm_mbox_priorbox)
    conv16_2_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv16_2_norm_mbox_priorbox_reshape')(
        conv16_2_norm_mbox_priorbox)
    conv17_2_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv17_2_norm_mbox_priorbox_reshape')(
        conv17_2_norm_mbox_priorbox)
    ### Concatenate the predictions from the different layers
    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv11_mbox_conf_reshape,
                                                       conv13_mbox_conf_reshape,
                                                       conv14_2_mbox_conf_reshape,
                                                       conv15_2_mbox_conf_reshape,
                                                       conv16_2_mbox_conf_reshape,
                                                       conv17_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv11_mbox_loc_reshape,
                                                     conv13_mbox_loc_reshape,
                                                     conv14_2_mbox_loc_reshape,
                                                     conv15_2_mbox_loc_reshape,
                                                     conv16_2_mbox_loc_reshape,
                                                     conv17_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv11_norm_mbox_priorbox_reshape,
                                                               conv13_norm_mbox_priorbox_reshape,
                                                               conv14_2_norm_mbox_priorbox_reshape,
                                                               conv15_2_norm_mbox_priorbox_reshape,
                                                               conv16_2_norm_mbox_priorbox_reshape,
                                                               conv17_2_norm_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    if mode == 'training':
        model = Model(inputs=input_x, outputs = predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
        iou_threshold = iou_threshold,
        top_k = top_k,
        nms_max_output_size = nms_max_output_size,
        coords = coords,
        normalize_coords = normalize_coords,
        img_height = img_height,
        img_width = img_width,
        name = 'decoded_predictions')(predictions)
        model = Model(inputs=net['input'], outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
        iou_threshold = iou_threshold,
        top_k = top_k,
        nms_max_output_size = nms_max_output_size,
        coords = coords,
        normalize_coords = normalize_coords,
        img_height = img_height,
        img_width = img_width,
        name = 'decoded_predictions')(predictions)
        model = Model(inputs=base_model.input, outputs=decoded_predictions)
    else:
        raise ValueError(
            "`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        predictor_sizes = np.array([net['conv11_mbox_conf']._keras_shape[1:3],
                                    net['conv13_mbox_conf']._keras_shape[1:3],
                                    net['conv14_2_mbox_conf']._keras_shape[1:3],
                                    net['conv15_2_mbox_conf']._keras_shape[1:3],
                                    net['conv16_2_mbox_conf']._keras_shape[1:3],
                                    net['conv17_2_mbox_conf']._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model

model_mapping = {
    "ssd_7": build_ssd_7_model,
    "vgg_ssd": build_vgg_ssd300_model,
    "mb1_ssd": build_mb1_ssd300_model,
    "mb2_ssd": None,
    "mb3_ssd": None,
}

if __name__ == "__main__":
    ### vgg_ssd300 配置参数
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    img_height = 300  # Height of the model input images
    img_width = 300  # Width of the model input images
    img_channels = 3  # Number of color channels of the model input images
    mean_color = [123, 117,
                  104]  # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
    swap_channels = [2, 1,
                     0]  # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
    n_classes = 20  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
    scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
                     1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
    scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87,
                   1.05]  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
    scales = scales_pascal
    aspect_ratios = [[1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
    two_boxes_for_ar1 = True
    steps = [8, 16, 32, 64, 100,
             300]  # The space between two adjacent anchor box center points for each predictor layer.
    offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
               0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
    clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
    variances = [0.1, 0.1, 0.2,
                 0.2]  # The variances by which the encoded target coordinates are divided as in the original implementation
    normalize_coords = True
    K.clear_session()  # Clear previous models from memory.

    # model = build_vgg_ssd300_model(image_size=(img_height, img_width, img_channels),
    #                             n_classes=n_classes,
    #                             mode='training',
    #                             l2_regularization=0.0005,
    #                             scales=scales,
    #                             aspect_ratios_per_layer=aspect_ratios,
    #                             two_boxes_for_ar1=two_boxes_for_ar1,
    #                             steps=steps,
    #                             offsets=offsets,
    #                             clip_boxes=clip_boxes,
    #                             variances=variances,
    #                             normalize_coords=normalize_coords,
    #                             subtract_mean=mean_color,
    #                             swap_channels=swap_channels)
    model = build_mb2_ssd300_model(image_size=(img_height, img_width, img_channels),
                                   n_classes=n_classes,
                                   mode='training',
                                   l2_regularization=0.0005,
                                   scales=scales,
                                   aspect_ratios_per_layer=aspect_ratios,
                                   two_boxes_for_ar1=two_boxes_for_ar1,
                                   steps=steps,
                                   offsets=offsets,
                                   clip_boxes=clip_boxes,
                                   variances=variances,
                                   normalize_coords=normalize_coords,
                                   subtract_mean=mean_color,
                                   swap_channels=swap_channels)
    model.summary()

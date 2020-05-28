"""
refer:keras ssd  https://github.com/pierluigiferrari/ssd_keras.git

"""
############### 目前VGG_ssd模型前面没有梯度，mb_ssd 完全没有梯度
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))
from math import ceil
import numpy as np
from matplotlib import pyplot as plt


import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from tensorflow.keras.models import load_model

from models.ssd.build_ssd_model import build_vgg_ssd300_model,build_mb1_ssd300_model,build_mb2_ssd300_model

from keras_loss_function.ssd_loss import SSDLoss

from encoder_decoder.ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder

from datasets.ssd_data_generate.object_detection_2d_data_generator import DataGenerator
from datasets.ssd_data_generate.object_detection_2d_geometric_ops import Resize
from datasets.ssd_data_generate.object_detection_2d_photometric_ops import ConvertTo3Channels
from datasets.ssd_data_generate.data_augmentation_chain_original_ssd import SSDDataAugmentation
from datasets.ssd_data_generate.object_detection_2d_misc_utils import apply_inverse_transforms

from configs import vgg_ssd300

# 1 构建模型
print("config info:",
      vgg_ssd300.image_size,
      type(vgg_ssd300.image_size),
      vgg_ssd300.n_classes,
      type(vgg_ssd300 .n_classes),
      vgg_ssd300.mode,
      type(vgg_ssd300 .mode),
      vgg_ssd300.l2_regularization,
      type(vgg_ssd300.l2_regularization)
      )
model_name = 'mb1_ssd300'

model_mapping={
    'vgg_ssd300':build_vgg_ssd300_model,
    'mb1_ssd300':build_mb1_ssd300_model,
    'mb2_ssd300':build_mb2_ssd300_model
}

model = model_mapping[model_name](vgg_ssd300.image_size,
                               n_classes=vgg_ssd300.n_classes,
                               mode=vgg_ssd300.mode,
                               l2_regularization=vgg_ssd300.l2_regularization,
                               scales=vgg_ssd300.scales,
                               aspect_ratios_per_layer=vgg_ssd300.aspect_ratios,
                               two_boxes_for_ar1=vgg_ssd300.two_boxes_for_ar1,
                               steps=vgg_ssd300.steps,
                               offsets=vgg_ssd300.offsets,
                               clip_boxes=vgg_ssd300.clip_boxes,
                               variances=vgg_ssd300.variances,
                               coords=vgg_ssd300.coords,
                              )

print(model.name)
for layer in model.layers:
    layer.trainable = True
model.summary()

# print(model.trainable_variables)
# 2 加载权重
weights_path = './weights/VGG_ILSVRC_16_layers_fc_reduced.h5'
try:
    model.load_weights(weights_path, by_name=True)
except:
    print('weight path is not right or weights')

# 3 优化器
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

# 4 损失函数
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

#  编译模型
model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)

tf.keras.utils.plot_model(model,model_name+'.png',show_shapes=True,show_layer_names=True)

# 5: 设置数据集路径.

# The directories that contain the images.
VOC_2007_images_dir = '/Data/jing/VOCdevkit/VOC2007/JPEGImages/'
VOC_2012_images_dir = '/Data/jing/VOCdevkit/VOC2012/JPEGImages/'

# The directories that contain the annotations.
VOC_2007_annotations_dir = '/Data/jing/VOCdevkit/VOC2007/Annotations/'
VOC_2012_annotations_dir = '/Data/jing/VOCdevkit/VOC2012/Annotations/'

# The paths to the image sets.
VOC_2007_train_image_set_filename = '/Data/jing/VOCdevkit/VOC2007/ImageSets/Main/train.txt'
VOC_2012_train_image_set_filename = '/Data/jing/VOCdevkit/VOC2012/ImageSets/Main/train.txt'
VOC_2007_val_image_set_filename = '/Data/jing/VOCdevkit/VOC2007/ImageSets/Main/val.txt'
VOC_2012_val_image_set_filename = '/Data/jing/VOCdevkit/VOC2012/ImageSets/Main/val.txt'
VOC_2007_trainval_image_set_filename = '/Data/jing/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
VOC_2012_trainval_image_set_filename = '/Data/jing/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
VOC_2007_test_image_set_filename = '/Data/jing/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']



# Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
# speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
# option in the constructor, because in that cas the images are in memory already anyway. If you don't
# want to create HDF5 datasets, comment out the subsequent two function calls.
# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.
trainval_h5_path = '/Data/jing/VOCdevkit/dataset_pascal_voc_07+12_trainval.h5'
test_h5_path = '/Data/jing/VOCdevkit/dataset_pascal_voc_07_test.h5'
train_dataset = DataGenerator(hdf5_dataset_path=trainval_h5_path,load_images_into_memory=False)
val_dataset = DataGenerator(hdf5_dataset_path=test_h5_path,load_images_into_memory=False)

if not os.path.exists(trainval_h5_path) or not os.path.exists(test_h5_path):
    train_dataset.parse_xml(images_dirs=[VOC_2007_images_dir,
                                         VOC_2012_images_dir],
                            image_set_filenames=[VOC_2007_trainval_image_set_filename,
                                                 VOC_2012_trainval_image_set_filename],
                            annotations_dirs=[VOC_2007_annotations_dir,
                                              VOC_2012_annotations_dir],
                            classes=classes,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)

    val_dataset.parse_xml(images_dirs=[VOC_2007_images_dir],
                          image_set_filenames=[VOC_2007_test_image_set_filename],
                          annotations_dirs=[VOC_2007_annotations_dir],
                          classes=classes,
                          include_classes='all',
                          exclude_truncated=False,
                          exclude_difficult=True,
                          ret=False)

    train_dataset.create_hdf5_dataset(file_path='/Data/jing/VOCdevkit/dataset_pascal_voc_07+12_trainval.h5',
                                      resize=False,
                                      variable_image_size=True,
                                      verbose=True)

    val_dataset.create_hdf5_dataset(file_path='/Data/jing/VOCdevkit/dataset_pascal_voc_07_test.h5',
                                    resize=False,
                                    variable_image_size=True,
                                    verbose=True)
else:
    print('load h5 file')
    train_dataset.load_hdf5_dataset(trainval_h5_path)
    val_dataset.load_hdf5_dataset(test_h5_path)
# 3: Set the batch size.

batch_size = 32  # Change the batch size if you like, or if you run into GPU memory issues.

# 4: Set the image transformations for pre-processing and data augmentation options.

# For the training generator:
ssd_data_augmentation = SSDDataAugmentation(img_height=vgg_ssd300.image_size[0],
                                            img_width=vgg_ssd300.image_size[1],
                                            background=vgg_ssd300.mean_color)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=vgg_ssd300.image_size[0], width=vgg_ssd300.image_size[1])

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.

if model_name=='vgg_ssd300':
    predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                       model.get_layer('fc7_mbox_conf').output_shape[1:3],
                       model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]
elif model_name == 'mb1_ssd300':
    predictor_sizes = np.array([model.get_layer('conv11_mbox_conf').output_shape[1:3],
                                model.get_layer('conv13_mbox_conf').output_shape[1:3],
                                model.get_layer('conv14_2_mbox_conf').output_shape[1:3],
                                model.get_layer('conv15_2_mbox_conf').output_shape[1:3],
                                model.get_layer('conv16_2_mbox_conf').output_shape[1:3],
                                model.get_layer('conv17_2_mbox_conf').output_shape[1:3]])

elif model_name == 'mb2_ssd300':
    predictor_sizes = np.array([model.get_layer('conv11_mbox_conf').output_shape[1:3],
                                model.get_layer('conv13_mbox_conf').output_shape[1:3],
                                model.get_layer('conv14_2_mbox_conf').output_shape[1:3],
                                model.get_layer('conv15_2_mbox_conf').output_shape[1:3],
                                model.get_layer('conv16_2_mbox_conf').output_shape[1:3],
                                model.get_layer('conv17_2_mbox_conf').output_shape[1:3]])

ssd_input_encoder = SSDInputEncoder(img_height=vgg_ssd300.image_size[0],
                                    img_width=vgg_ssd300.image_size[1],
                                    n_classes=vgg_ssd300.n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=vgg_ssd300.scales,
                                    aspect_ratios_per_layer=vgg_ssd300.aspect_ratios,
                                    two_boxes_for_ar1=vgg_ssd300.two_boxes_for_ar1,
                                    steps=vgg_ssd300.steps,
                                    offsets=vgg_ssd300.offsets,
                                    clip_boxes=vgg_ssd300.clip_boxes,
                                    variances=vgg_ssd300.variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=vgg_ssd300.normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# Define a learning rate schedule.

def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001


# Define model callbacks.

model_checkpoint = ModelCheckpoint(
    filepath='/Data/jing/weights/ssd300_pascal_07+12_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1)
# model_checkpoint.best =

csv_logger = CSVLogger(filename='ssd300_pascal_07+12_training_log.csv',
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan]

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch = 0
final_epoch = 120
steps_per_epoch = 1000

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size / batch_size),
                              initial_epoch=initial_epoch)

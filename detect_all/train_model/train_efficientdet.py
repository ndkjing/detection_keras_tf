"""
refer:https://github.com/xuannianz/EfficientDet.git
"""

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from augment.efficientdet.color import VisualEffect
from augment.efficientdet.misc import MiscEffect
from models.efficientdet.build_efficientdet import efficientdet
from keras_layers.efficientdet_layers.efficientdet_loss import smooth_l1, focal, smooth_l1_quad
from backbone.efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES
from configs import efficientdet as efficientdet_config

def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """
    Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_callbacks(training_model, prediction_model, validation_generator, efficientdet_config):
    """
    Creates the callbacks to use during training.

    Args
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    if efficientdet_config.tensorboard_dir:
        if tf.version.VERSION > '2.0.0':
            file_writer = tf.summary.create_file_writer(efficientdet_config.tensorboard_dir)
            file_writer.set_as_default()
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=efficientdet_config.tensorboard_dir,
            histogram_freq=0,
            batch_size=efficientdet_config.batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        callbacks.append(tensorboard_callback)

    if efficientdet_config.evaluation and validation_generator:
        if efficientdet_config.dataset_type == 'coco':
            from eval_infer.efficientdet.coco import Evaluate
            # use prediction model for evaluation
            evaluation = Evaluate(validation_generator, prediction_model, tensorboard=tensorboard_callback)
        else:
            from eval_infer.efficientdet.pascal import Evaluate
            evaluation = Evaluate(validation_generator, prediction_model, tensorboard=tensorboard_callback)
        callbacks.append(evaluation)

    # save the model
    if efficientdet_config.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(efficientdet_config.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                efficientdet_config.snapshot_path,
                f'{efficientdet_config.dataset_type}_{{epoch:02d}}_{{loss:.4f}}_{{val_loss:.4f}}.h5' if efficientdet_config.compute_val_loss
                else f'{efficientdet_config.dataset_type}_{{epoch:02d}}_{{loss:.4f}}.h5'
            ),
            verbose=1,
            save_weights_only=True,
            # save_best_only=True,
            # monitor="mAP",
            # mode='max'
        )
        callbacks.append(checkpoint)

    # callbacks.append(keras.callbacks.ReduceLROnPlateau(
    #     monitor='loss',
    #     factor=0.1,
    #     patience=2,
    #     verbose=1,
    #     mode='auto',
    #     min_delta=0.0001,
    #     cooldown=0,
    #     min_lr=0
    # ))

    return callbacks


def create_generators(efficientdet_config):
    """
    Create generators for training and validation.

    Args
        args: parseargs object containing configuration for generators.
        preprocess_image: Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': efficientdet_config.batch_size,
        'phi': efficientdet_config.phi,
        'detect_text': efficientdet_config.detect_text,
        'detect_quadrangle': efficientdet_config.detect_quadrangle
    }

    # create random transform generator for augmenting training data
    if efficientdet_config.random_transform:
        misc_effect = MiscEffect()
        visual_effect = VisualEffect()
    else:
        misc_effect = None
        visual_effect = None

    if efficientdet_config.dataset_type == 'pascal':
        from datasets.efficientdet_data_generate.pascal import PascalVocGenerator
        train_generator = PascalVocGenerator(
            efficientdet_config.pascal_path,
            'trainval',
            skip_difficult=True,
            misc_effect=misc_effect,
            visual_effect=visual_effect,
            **common_args
        )

        validation_generator = PascalVocGenerator(
            efficientdet_config.pascal_path,
            'val',
            skip_difficult=True,
            shuffle_groups=False,
            **common_args
        )
    elif efficientdet_config.dataset_type == 'csv':
        from datasets.efficientdet_data_generate.csv_ import CSVGenerator
        train_generator = CSVGenerator(
            efficientdet_config.annotations_path,
            efficientdet_config.classes_path,
            misc_effect=misc_effect,
            visual_effect=visual_effect,
            **common_args
        )

        if efficientdet_config.val_annotations_path:
            validation_generator = CSVGenerator(
                efficientdet_config.val_annotations_path,
                efficientdet_config.classes_path,
                shuffle_groups=False,
                **common_args
            )
        else:
            validation_generator = None
    elif efficientdet_config.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from datasets.efficientdet_data_generate.coco import CocoGenerator
        train_generator = CocoGenerator(
            efficientdet_config.coco_path,
            'train2017',
            misc_effect=misc_effect,
            visual_effect=visual_effect,
            group_method='random',
            **common_args
        )

        validation_generator = CocoGenerator(
            efficientdet_config.coco_path,
            'val2017',
            shuffle_groups=False,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(efficientdet_config.dataset_type))

    return train_generator, validation_generator


def main(args=None):

    # create the generators
    train_generator, validation_generator = create_generators(efficientdet_config)
    num_classes = train_generator.num_classes()
    num_anchors = train_generator.num_anchors

    # optionally choose specific GPU
    if efficientdet_config.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = efficientdet_config.gpu

    # K.set_session(get_session())

    model, prediction_model = efficientdet(efficientdet_config.phi,
                                           num_classes=num_classes,
                                           num_anchors=num_anchors,
                                           weighted_bifpn=efficientdet_config.weighted_bifpn,
                                           freeze_bn=efficientdet_config.freeze_bn,
                                           detect_quadrangle=efficientdet_config.detect_quadrangle
                                           )
    # load pretrained weights
    if efficientdet_config.snapshot:
        if efficientdet_config.snapshot == 'imagenet':
            model_name = 'efficientnet-b{}'.format(efficientdet_config.phi)
            file_name = '{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(model_name)
            file_hash = WEIGHTS_HASHES[model_name][1]
            weights_path = keras.utils.get_file(file_name,
                                                BASE_WEIGHTS_PATH + file_name,
                                                cache_subdir='models',
                                                file_hash=file_hash)
            model.load_weights(weights_path, by_name=True)
        else:
            print('Loading model, this may take a second...')
            model.load_weights(efficientdet_config.snapshot, by_name=True)

    # freeze backbone layers
    if efficientdet_config.freeze_backbone:
        # 227, 329, 329, 374, 464, 566, 656
        for i in range(1, [227, 329, 329, 374, 464, 566, 656][efficientdet_config.phi]):
            model.layers[i].trainable = False

    if efficientdet_config.gpu and len(efficientdet_config.gpu.split(',')) > 1:
        model = keras.utils.multi_gpu_model(model, gpus=list(map(int, efficientdet_config.gpu.split(','))))

    # compile model
    model.compile(optimizer=Adam(lr=1e-3), loss={
        'regression': smooth_l1_quad() if efficientdet_config.detect_quadrangle else smooth_l1(),
        'classification': focal()
    }, )

    # print(model.summary())

    # create the callbacks
    callbacks = create_callbacks(
        model,
        prediction_model,
        validation_generator,
        efficientdet_config,
    )

    if not efficientdet_config.compute_val_loss:
        validation_generator = None
    elif efficientdet_config.compute_val_loss and validation_generator is None:
        raise ValueError('When you have no validation data, you should not specify --compute-val-loss.')

    # start training
    return model.fit_generator(
        generator=train_generator,
        steps_per_epoch=efficientdet_config.steps,
        initial_epoch=0,
        epochs=efficientdet_config.epochs,
        verbose=1,
        callbacks=callbacks,
        workers=efficientdet_config.workers,
        use_multiprocessing=efficientdet_config.multiprocessing,
        max_queue_size=efficientdet_config.max_queue_size,
        validation_data=validation_generator
    )


if __name__ == '__main__':
    main()
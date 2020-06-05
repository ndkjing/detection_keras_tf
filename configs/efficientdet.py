dataset_type = 'pascal'  # "coco"
pascal_path = '/Data/jing/VOCdevkit/VOC2012'
detect_quadrangle = False
detect_text = False
snapshot = 'imagenet'  # Resume training from a snapshot.')
freeze_backbone = False
freeze_bn = False  # Freeze training of BatchNormalization layers
weighted_bifpn = False  # Use weighted BiFPN
batch_size = 1 # Size of the batches
phi = 1  # Hyper parameter phi', default=0, type=int, choices=(0, 1, 2, 3, 4, 5, 6))
assert phi in [0, 1, 2, 3, 4, 5, 6]
gpu = "2"  # Id of the GPU to use (as reported by nvidia-smi).')
epochs = 500
steps = 500
snapshot_path = '/Data/jing/weights/detection/keras_tf/efficientdet'  # 'Path to store snapshots of models during training',
tensorboard_dir = '/Data/jing/weights/detection/keras_tf/efficientdet'
no_snapshots = True  # ', help='Disable saving snapshots.', dest='snapshots', action='store_false')
snapshots=True
no_evaluation = True  # ', help='Disable per epoch evaluation.', dest='evaluation',
evaluation = True
random_transform = False  # ', help='Randomly transform image and annotations.', action='store_true')
compute_val_loss = False  # ', help='Compute validation loss during training', dest='compute_val_loss',

# Fit generator arguments
multiprocessing = False  # ', help='Use multiprocessing in fit_generator.', action='store_true')
workers = 1  # ', help='Number of generator workers.', type=int, default=1)
max_queue_size = 10  # ', help='Queue length for multiprocessing workers in fit_generator.', type=int,

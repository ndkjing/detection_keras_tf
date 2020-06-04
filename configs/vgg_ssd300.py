image_size= (300,300,3)  # Height  Width Channel
mean_color = [123, 117,104],  # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
mode = 'training'  # ['training ','inference ]
l2_regularization = 0.0005,
swap_channels = [2, 1,0],  # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 20  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87,1.05]  # anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_pascal
        # The anchor box aspect ratios used in the original SSD300; the order matters
aspect_ratios = [[1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                         [1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5]]
two_boxes_for_ar1 = True
        # The space between two adjacent anchor box center points for each predictor layer.
steps = [8, 16, 32, 64, 100,300]
        # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5,0.5]
        # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
clip_boxes = False
        # The variances by which the encoded target coordinates are divided as in the original implementation
variances = [0.1, 0.1, 0.2,0.2]
normalize_coords = True
coords = 'centroids',
subtract_mean = [123, 117, 104],
divide_by_stddev = None,
confidence_thresh = 0.01,
iou_threshold = 0.45,
top_k = 200,
nms_max_output_size = 400,
return_predictor_sizes = False


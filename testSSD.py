from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import tfrecord_voc_utils as voc_utils
import tensorflow as tf
import SSD300 as net
import os

lr = 0.001
batch_size = 16
buffer_size = 24
epochs = 160
reduce_lr_epoch = []
ckpt_path = os.path.join('.', 'vgg_16.ckpt')
config = {
    'mode': 'train',
    'data_format': 'channels_last',
    'num_classes': 20,
    'weight_decay': 1e-4,
    'keep_prob': 0.5,
    'batch_size': batch_size,
    'nms_score_threshold': 0.3,
    'nms_max_boxes': 20,
    'nms_iou_threshold': 0.5,
    'pretraining_weight': ckpt_path
}

image_preprocess_config = {
    'data_format': 'channels_last',
    'target_size': [448, 448],
    'shorter_side': 480,
    'is_random_crop': False,
    'random_horizontal_flip': 0.5,
    'random_vertical_flip': 0.,
    'pad_truth_to': 50
}

data = ['./test/test_00000-of-00005.tfrecord',
        './test/test_00001-of-00005.tfrecord',
        './test/test_00002-of-00005.tfrecord',
        './test/test_00003-of-00005.tfrecord',
        './test/test_00004-of-00005.tfrecord']

train_gen = voc_utils.get_generator(data,
                                    batch_size, buffer_size, image_preprocess_config)
trainset_provider = {
    'data_shape': [448, 448, 3],
    'num_train': 5011,
    'num_val': 0,
    'train_generator': train_gen,
    'val_generator': None
}
ssd300 = net.SSD300(config, trainset_provider)

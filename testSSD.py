from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import tfrecord_voc_utils as voc_utils
import tensorflow as tf
import SSD300 as net
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
lr = 0.001
batch_size = 1
buffer_size = 5
epochs = 160
reduce_lr_epoch = [50, 150]
ckpt_path = os.path.join('.', 'vgg_16.ckpt')
config = {
    'mode': 'train',
    'data_format': 'channels_last',
    'num_classes': 20,
    'weight_decay': 1e-4,
    'keep_prob': 0.5,
    'batch_size': batch_size,
    'nms_score_threshold': 0.5,
    'nms_max_boxes': 20,
    'nms_iou_threshold': 0.5,
    'pretraining_weight': ckpt_path
}

image_preprocess_config = {
    'data_format': 'channels_last',
    'target_size': [300, 300],
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
    'data_shape': [300, 300, 3],
    'num_train': 5011,
    'num_val': 0,
    'train_generator': train_gen,
    'val_generator': None
}
ssd300 = net.SSD300(config, trainset_provider)
ssd300.train_one_epoch(lr)
for i in range(epochs):
    print('-'*25, 'epoch', i, '-'*25)
    if i in reduce_lr_epoch:
        lr = lr/10.
        print('reduce lr, lr=', lr, 'now')
    mean_loss = ssd300.train_one_epoch(lr)
    print('>> mean loss', mean_loss)
    ssd300.save_weight('latest', './weight/test')

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import tfrecord_voc_utils as voc_utils
import tensorflow as tf
import numpy as np
import SSD300 as net
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, transform
from utils.voc_classname_encoder import  classname_to_ids
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
lr = 0.0001
batch_size = 32
buffer_size = 512
epochs = 160
reduce_lr_epoch = [50, 150]
ckpt_path = os.path.join('.', 'vgg_16.ckpt')
config = {
    'mode': 'test',
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
    'pad_truth_to': 60
}

data = ['./dat/test_00000-of-00010.tfrecord',
        './dat/test_00001-of-00010.tfrecord',
        './dat/test_00002-of-00010.tfrecord',
        './dat/test_00003-of-00010.tfrecord',
        './dat/test_00004-of-00010.tfrecord',
        './dat/test_00005-of-00010.tfrecord',
        './dat/test_00006-of-00010.tfrecord',
        './dat/test_00007-of-00010.tfrecord',
        './dat/test_00008-of-00010.tfrecord',
        './dat/test_00009-of-00010.tfrecord']

train_gen = voc_utils.get_generator(data,
                                    batch_size, buffer_size, image_preprocess_config)
trainset_provider = {
    'data_shape': [300, 300, 3],
    'num_train': 22136,
    'num_val': 0,
    'train_generator': train_gen,
    'val_generator': None
}
ssd300 = net.SSD300(config, trainset_provider)
# ssd300.load_weight('./ssd/test-64954')
for i in range(epochs):
    print('-'*25, 'epoch', i, '-'*25)
    if i in reduce_lr_epoch:
        lr = lr/10.
        print('reduce lr, lr=', lr, 'now')
    mean_loss = ssd300.train_one_epoch(lr)
    print('>> mean loss', mean_loss)
    ssd300.save_weight('latest', './ssd/test')
# img = io.imread('000026.jpg')
# img = transform.resize(img, [300,300])
# img = np.expand_dims(img, 0)
# result = ssd300.test_one_image(img)
# id_to_clasname = {k:v for (v,k) in classname_to_ids.items()}
# scores = result[0]
# bbox = result[1]
# class_id = result[2]
# print(scores, bbox, class_id)
# plt.figure(1)
# plt.imshow(np.squeeze(img))
# axis = plt.gca()
# for i in range(len(scores)):
#     rect = patches.Rectangle((bbox[i][1],bbox[i][0]), bbox[i][3]-bbox[i][1],bbox[i][2]-bbox[i][0],linewidth=2,edgecolor='b',facecolor='none')
#     axis.add_patch(rect)
#     plt.text(bbox[i][1],bbox[i][0], id_to_clasname[class_id[i]]+str(' ')+str(scores[i]), color='red', fontsize=12)
# plt.show()





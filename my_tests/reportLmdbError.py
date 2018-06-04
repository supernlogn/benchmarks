from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from six.moves import xrange  # pylint: disable=redefined-builtin
import lmdb
import PIL.Image
from StringIO import StringIO

# specify dataset path
path_prefix = '/mnt/terabyte/datasets/imagenet/caffe/ilsvrc12_'
path_postfix = '_lmdb'
supported_modes = ['train', 'val']
mode = supported_modes[0]
full_path = path_prefix + mode + path_postfix

# specify how many datums to read at once
batch_length = 11

# set numpy array print options 
np.set_printoptions(threshold=21)

reader = tf.LMDBReader(name='reader')
keys_queue = tf.FIFOQueue(
                    capacity=32,
                    dtypes=[dtypes.string],
                    shapes=())

# scenario 1 (buggy)
keys1, values1 = reader.read_up_to(keys_queue, batch_length)
jpg_buffer1 = tf.decode_raw(values1, out_type=tf.uint8)

# scenario 2 (good)
keys2, values2 = reader.read_up_to(keys_queue, 11)
jpg_buffer2 = tf.decode_raw(values2, out_type=tf.uint8)

with tf.Session() as sess:
    keys_queue.enqueue([full_path]).run()
    keys_queue.close().run()
    buffer2 = sess.run(jpg_buffer2)
    print(buffer2.shape) 
    print(buffer2[0:20])
    buffer1 = sess.run(jpg_buffer1)
    print(buffer1.shape)
    print(buffer1[:,0:20])
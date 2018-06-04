# Copyright 2017 Ioannis Athanasiadis(supernlogn). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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


path_prefix = '/mnt/terabyte/datasets/imagenet/caffe/ilsvrc12_'
path_postfix = '_lmdb'
supported_modes = ['train', 'val']
mode = supported_modes[0]

full_path = path_prefix + mode + path_postfix

# def read_lmdb(lmdb_file):
#     cursor = lmdb.open(lmdb_file, readonly=True).begin().cursor()
#     datum = caffe.proto.caffe_pb2.Datum()
#     for _, value in cursor:
#         datum.ParseFromString(value)
#         s = StringIO()
#         s.write(datum.data)
#         s.seek(0)

#         yield np.array(PIL.Image.open(s)), datum.label

# for im, label in read_lmdb(full_path):
#     print label, im

np.set_printoptions(threshold='nan')
# env = lmdb.open(full_path, readonly=True)
keys_list = []

i = 1
# with env.begin() as txn:
#     cursor = txn.cursor()
#     for key, value in cursor:
#         val = np.fromstring(value, dtype=np.uint8)
#         print("env print: ", val.shape)
#         label = val[12:17]
#         print([bin(x)[2:].zfill(8) for x in label])
#         keys_list.append(key)
#         i = i +1
#         if i >= 10:
#             break
# print(key)
# env.close()

np.set_printoptions(threshold=20)
# print(int(value[0]))
# I1 = value.index( '\xFF\xD8' )
# I2 = value.index( '',I1)
# print(I1)

# value = np.fromstring(value,dtype=np.uint8)
# # print(value[I1:])

# header_data = value[0:17]

# print(np.size(value) - 256*256*3)
# img = value[17:]
# img = np.reshape(img, [256, 256, 3])
# imgToShow = PIL.Image.fromarray(img, 'RGB')
# imgToShow.save('tensImg.png')

# path_tensor = tf.p.aconvert_to_tensor(len(full_path), dtype=tf.int32)
# tf.random_shuffle(keys_tensor)


# tf.train.add_queue_runner(tf.train.QueueRunner(keys_queue,[kq_enqueue_op] * 1))
print(len(full_path))


reader = tf.LMDBReader(name='reader')
keys_queue = tf.FIFOQueue(
    capacity=2,
    dtypes=[dtypes.string],
    shapes=())

# i = tf.Variable(initial_value=0, trainable=False, name="lmdb_iterator_var")

datum_size = 196625

vals = tf.zeros(shape=(1, datum_size), dtype=tf.uint8)


def in_body(in_iterator, vals):
  vals = tf.concat(axis=0,
                   values=[vals,
                           tf.expand_dims(axis=0,
                                          input=tf.decode_raw(reader.read(keys_queue)[1],
                                                              out_type=tf.uint8)[:])])
  return in_iterator + 1, vals
in_i = []
in_while_reader = []
for i in range(0, 3):
  in_i.append(tf.constant(0))
  in_while_reader.append(tf.while_loop(cond=lambda i, vals: tf.less(i, 10),
                                       body=in_body,
                                       loop_vars=[in_i[-1], vals],
                                       shape_invariants=[
                                           in_i[-1].get_shape(), tf.TensorShape((None, datum_size))],
                                       parallel_iterations=1))


def out_body(out_iterator, vals):
  out_case = []
  for i in range(0, 3):
    out_case.append((tf.equal(out_iterator, i),
                      lambda: in_while_reader[i]))
  r = tf.case(out_case, default=lambda: in_while_reader[0])
  vals = tf.concat(axis=0,
                   values=[vals, r[1]])
  return out_iterator + 1, vals

out_i = tf.constant(0)

out_while_reader = tf.while_loop(cond=lambda out_i, vals: tf.less(out_i, 2),
                                 body=out_body,
                                 loop_vars=[out_i, vals],
                                 shape_invariants=[
                                     out_i.get_shape(), tf.TensorShape((None, datum_size))],
                                 parallel_iterations=1)


keys, values = reader.read(keys_queue)
jpg_buffer = tf.decode_raw(values, out_type=tf.uint8)

enqueue_op = keys_queue.enqueue([values])

# jpg_label = jpg_buffer[:,-5:-1]
# jpg_img = jpg_buffer[:,12:-5]
# jpg_img = tf.reshape(jpg_img, [32, 3, 256, 256])
# jpg_img = tf.transpose(jpg_img, [0,2,3,1])

# rev = tf.constant([2], dtype=tf.int32)
# jpg_img = tf.reverse(jpg_img, rev)


with tf.Session() as sess:
  # keys_queue.enqueue([full_path]).run()
  # keys_queue.close().run()
  w = sess.run(enqueue_op)
  print(w.shape)
  # print(w)
# search if two rows are the same


for it1 in range(w.shape[0]):
  ans = False
  for it2 in range(it1 + 1, w.shape[0]):
    if (np.array_equal(w[it1, :], w[it2, :])):
      print("Found them: %d, %d" % (it1, it2))

#     # coord = tf.train.Coordinator()
#     # threads =  tf.train.start_queue_runners(coord=coord)

# imgToShow = PIL.Image.fromarray(img, 'RGB')
# imgToShow.save('tensImg2.jpg')
#     k,v = sess.run([keys, values])
#     # print(k, v)
#     print((len(v) - 2556*256*3))
#     b = np.array(v)
#     np.reshape(b,[256, 256, 3])
#     # coord.request_stop()
#     # coord.join(threads)

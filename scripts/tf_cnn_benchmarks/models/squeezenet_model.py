# Copyright 2017 Ioannis Athanasiadis(supernlogn) one of the wanna be TensorFlow Authors. All Rights Reserved.
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
"""SqueezeNet model configuration.

References:
  Iandola, Forrest N., et al. "SqueezeNet: AlexNet-level accuracy
  with 50x fewer parameters and< 0.5 MB model size."
  arXiv preprint arXiv:1602.07360 (2016).
"""


from six.moves import xrange  # pylint: disable=redefined-builtin
import model as model_lib
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.layers.python.layers import utils

def fire_module(cnn,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None,
                outputs_collections=None):
  def squeeze(cnn, num_outputs):
    cnn.conv(num_outputs, 1, 1, 1, 1)

  def expand(cnn, num_outputs):
    input_layer = cnn.top_layer
    cnn.conv(num_outputs, 1, 1, 1, 1, input_layer=input_layer)
    e1x1 = cnn.top_layer
    cnn.conv(num_outputs, 3, 3, 1, 1, input_layer=input_layer)
    e3x3 = cnn.top_layer
    cnn.concat_layers(list_of_layers=[e1x1, e3x3])                

  squeeze(cnn, squeeze_depth)
  expand(cnn, expand_depth)
  # return utils.collect_named_outputs(outputs_collections,
  #                                    sc.original_name_scope, outputs)
   
class SqueezenetModel(model_lib.Model):
  def __init__(self, model):
    image_size = 224
    batch_size = 32
    learning_rate = 10.0 ** (-4)
    self.num_classes = 1000
    super(SqueezenetModel, self).__init__(
        model, image_size, batch_size, learning_rate)

  def add_inference(self, cnn):
    """Original squeezenet architecture for 224x224 images."""
    cnn.conv(96, 7, 7, 2, 2)
    cnn.mpool(3, 3, 2, 2)
    fire_module(cnn, 16, 64)
    fire_module(cnn, 16, 64)
    fire_module(cnn, 32, 128)
    cnn.mpool(3, 3, 2, 2)
    fire_module(cnn, 32, 128)
    fire_module(cnn, 48, 192)
    fire_module(cnn, 48, 192)
    fire_module(cnn, 64, 256)
    cnn.mpool(3, 3, 2, 2)
    fire_module(cnn, 64, 256)
    cnn.conv(self.num_classes, 1, 1, 1, 1)
    cnn.apool(13, 13, 1, 1)
    cnn.spatial_mean()

class SqueezenetCifar10Model(model_lib.Model):

  def __init__(self, model):
    batch_norm_decay = 0.999
    self.num_classes = 10
    image_size = 32
    batch_size = 64
    learning_rate = 10.0 ** (-4)
    layer_counts = None
    super(SqueezenetCifar10Model, self).__init__(
        model, image_size, batch_size, learning_rate)

  def add_inference(self, cnn):
    """Modified version of squeezenet for CIFAR images"""
    cnn.conv(96, 2, 2, 1, 1, activation=None)
    cnn.mpool(2, 2, 1, 1)
    fire_module(cnn, 16, 64)
    fire_module(cnn, 16, 64)
    fire_module(cnn, 32, 128)
    cnn.mpool(2, 2, 1, 1)
    fire_module(cnn, 32, 128)
    fire_module(cnn, 48, 192)
    fire_module(cnn, 48, 192)
    fire_module(cnn, 64, 256)
    cnn.mpool(2, 2, 1, 1)
    fire_module(cnn, 64, 256)
    # Use global average pooling per 'Network in Network [1]'
    # net = slim.avg_pool2d(net, [4, 4], scope='avgpool10')
    cnn.apool(4, 4, 1, 1)
    # net = slim.conv2d(net, num_classes, [1, 1],
    #                 activation_fn=None,
    #                 normalizer_fn=None,
    #                 scope='conv10')
    cnn.conv(self.num_classes, 1, 1, 1, 1)
    cnn.spatial_mean()



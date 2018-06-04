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

"""runs all benchmarks from benchmark_configs.yml file,
   except vggXX, because of malfunction.
This script should only be run from opensource repository.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import logging
import os
from string import maketrans
import subprocess
import sys
import yaml
import tensorflow as tf
import numpy as np

import benchmark_cnn
import cnn_util
from cnn_util import log_fn




_OUTPUT_FILE_ENV_VAR = 'TF_DIST_BENCHMARK_RESULTS_FILE'
_TEST_NAME_ENV_VAR = 'TF_DIST_BENCHMARK_NAME'
_PORT = 5000



def _RunBenchmark(name, yaml_file):
    pass

benchmark_cnn.define_flags()


def main(_):
  params = benchmark_cnn.make_params_from_flags()
  
  models = ['alexnet', ]

if __name__ == '__main__':
  tf.app.run()
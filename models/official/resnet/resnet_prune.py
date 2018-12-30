# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Train a ResNet-50 model on ImageNet on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf

from common import tpu_profiler_hook
from official.resnet import imagenet_input
from official.resnet import lars_util
from official.resnet import resnet_model
from official.resnet import resnet_main
from tensorflow.contrib import summary
from tensorflow.contrib.tpu.python.tpu import async_checkpoint
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.estimator import estimator

FLAGS = flags.FLAGS

flags.DEFINE_string('prune_percs', default=None, help="")
flags.DEFINE_boolean('do_prune', default=False, help="")


def main(unused_argv):
  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu if (FLAGS.tpu or FLAGS.use_tpu) else '',
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)

  if FLAGS.use_async_checkpointing:
    save_checkpoints_steps = None
  else:
    save_checkpoints_steps = max(100, FLAGS.iterations_per_loop)
  config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=save_checkpoints_steps,
      log_step_count_steps=FLAGS.log_step_count_steps,
      session_config=tf.ConfigProto(
          graph_options=tf.GraphOptions(
              rewrite_options=rewriter_config_pb2.RewriterConfig(
                  disable_meta_optimizer=True))),
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_cores,
          per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.
          PER_HOST_V2))  # pylint: disable=line-too-long

  assert FLAGS.precision == 'bfloat16' or FLAGS.precision == 'float32', (
      'Invalid value for --precision flag; must be bfloat16 or float32.')
  tf.logging.info('Precision: %s', FLAGS.precision)
  use_bfloat16 = FLAGS.precision == 'bfloat16'

  tf.logging.info('Using dataset: %s', FLAGS.data_dir)
  imagenet_train, imagenet_eval = [
      imagenet_input.ImageNetInput(
          is_training=is_training,
          data_dir=FLAGS.data_dir,
          transpose_input=FLAGS.transpose_input,
          cache=FLAGS.use_cache and is_training,
          image_size=FLAGS.image_size,
          num_parallel_calls=FLAGS.num_parallel_calls,
          use_bfloat16=use_bfloat16) for is_training in [True, False]
  ]

  steps_per_epoch = FLAGS.num_train_images // FLAGS.train_batch_size
  eval_steps = FLAGS.num_eval_images // FLAGS.eval_batch_size

  prune_percents = [float(p) for p in FLAGS.prune_percs.split(',')]
  for p in prune_percents:
    FLAGS.do_prune = True
    FLAGS.drop_prob = 1.0
    FLAGS.targ_rate = p

    resnet_classifier = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=resnet_main.resnet_model_fn,
        config=config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        export_to_tpu=FLAGS.export_to_tpu)

    start_timestamp = time.time()  # This time will include compilation time
    eval_results = resnet_classifier.evaluate(
        input_fn=imagenet_eval.input_fn, steps=eval_steps)
    elapsed_time = int(time.time() - start_timestamp)
    tf.logging.info('%f%% Prune -- Eval results: %s. Elapsed seconds: %d',
                    p * 100, eval_results, elapsed_time)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)

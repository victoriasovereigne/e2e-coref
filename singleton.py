#!/usr/bin/env python

import os
import sys
sys.path.append(os.getcwd())
import json
import time
import random

import numpy as np
import tensorflow as tf
import coref_model as cm
import util

if __name__ == "__main__":
  start_time = time.time()
  if len(sys.argv) > 1:
    name = sys.argv[1]
  else:
    name = os.environ["EXP"]
  config = util.get_config("experiments.conf")[name]
  report_frequency = config["report_frequency"]

  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))
  util.print_config(config)

  # print "Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"])
  print "Setting CUDA_VISIBLE_DEVICES to: ", str(os.environ["CUDA_VISIBLE_DEVICES"])
  # if "GPU" in os.environ:
  #   util.set_gpus(int(os.environ["GPU"]))
  # else:
  #   util.set_gpus()

  model = cm.CorefModel(config)
  saver = tf.train.Saver()
  init_op = tf.global_variables_initializer()

  log_dir = config["log_dir"]
  writer = tf.summary.FileWriter(os.path.join(log_dir, "train"), flush_secs=20)

  stopping_criteria = config["stopping_criteria"]

  # Create a "supervisor", which oversees the training process.
  sv = tf.train.Supervisor(logdir=log_dir,
                           init_op=init_op,
                           saver=saver,
                           global_step=model.global_step,
                           save_model_secs=120)

  # The supervisor takes care of session initialization, restoring from
  # a checkpoint, and closing when done or an error occurs.
  '''
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
  # Creates a session with log_device_placement set to True.
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
  # Runs the op.
  print(sess.run(c))
  '''
  with sv.managed_session() as session:

    init_global_step = session.run(model.global_step)
    print "init_global_step", init_global_step

    model.start_enqueue_thread(session)
    accumulated_loss = 0.0
    initial_time = time.time()
    while not sv.should_stop():
      tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])
      # print "use_gpu", use_gpu
      accumulated_loss += tf_loss

      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency
        print "[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second)
        writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
        accumulated_loss = 0.0

        # 220001 > 200000 + 20000
      # if tf_global_step > init_global_step + stopping_criteria:
      if tf_global_step > stopping_criteria:
        sv.stop()
  # Ask for all the services to stop.
  sv.stop()
  end_time = time.time()

  running_time = end_time - start_time
  print "Running time:", running_time

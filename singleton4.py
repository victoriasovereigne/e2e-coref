#!/usr/bin/env python

import os
import sys
sys.path.append(os.getcwd())
import json
import time
import random

import numpy as np
import tensorflow as tf
import coref_model4 as cm
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
  with sv.managed_session() as session:

    init_global_step = session.run(model.global_step)
    print "init_global_step", init_global_step

    model.start_enqueue_thread(session)
    accumulated_loss = 0.0
    acc_total_loss = 0.0
    acc_domain_loss = 0.0
    acc_dmrm = 0.0

    initial_time = time.time()
    while not sv.should_stop():
      tf_loss, tf_global_step, _, pred, total_loss, domain_loss, values = session.run([model.loss, 
                              model.global_step, model.train_op, model.predictions, model.total_loss, 
                              model.domain_loss, model.values])

      domain_accuracy = values[0]
      domain_loss_reduce_mean = values[1]
      pairwise_reduced = values[2]
      N = values [3]
      neg_ll = values[4]
      d_logits2 = values[5]
      d_logits = values[6]

      print "----------------------"
      print "d_logits2"
      print d_logits2.shape
      print d_logits2[0]
      print "----------------------"
      print "d_logits"
      print d_logits.shape
      print d_logits[0]
      print "----------------------"
      break

      # is infinity
      # print type(neg_ll)
      inf_exist = np.isinf(neg_ll)
      nan_exist = np.isnan(neg_ll)

      if True in inf_exist:
        print "infinite number"
        print inf_exist.argmax()
        break
        
      if True in nan_exist:
        print "Nan exist"
        print nan_exist.argmax
        break

      # print "use_gpu", use_gpu
      accumulated_loss += tf_loss
      acc_total_loss += total_loss
      acc_domain_loss += domain_loss
      acc_dmrm += domain_loss_reduce_mean

      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency
        print "[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second)
        writer.add_summary(util.make_summary({"original loss": average_loss}), tf_global_step)
        accumulated_loss = 0.0
        
        average_domain_loss = acc_domain_loss / report_frequency
        print "[{}] domain_loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_domain_loss, steps_per_second)
        writer.add_summary(util.make_summary({"domain loss": average_domain_loss}), tf_global_step)
        acc_domain_loss = 0.0

        average_domain_loss_rm = acc_dmrm / report_frequency
        print "[{}] domain_loss_reduce_mean={:.2f}, steps/s={:.2f}".format(tf_global_step, average_domain_loss_rm, steps_per_second)
        writer.add_summary(util.make_summary({"domain loss reduce mean": average_domain_loss_rm}), tf_global_step)
        acc_dmrm = 0.0

        average_total_loss = acc_total_loss / report_frequency
        print "[{}] total_loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_total_loss, steps_per_second)
        writer.add_summary(util.make_summary({"total loss": average_total_loss}), tf_global_step)
        acc_total_loss = 0.0

        print "domain_accuracy", domain_accuracy
        print "l value", model.l

      if tf_global_step > stopping_criteria:
        sv.stop()
  # Ask for all the services to stop.
  sv.stop()
  end_time = time.time()

  running_time = end_time - start_time
  print "Running time:", running_time

# @Time : 2019/6/11 4:10 PM 
# @Author : Kaishun Zhang 
# @File : test.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('model/my-model-0.meta')
  new_saver.restore(sess, 'model/my-model-0')
  # tf.get_collection() return list, just need the first element
  y = tf.get_collection('predict')[0]
  graph = tf.get_default_graph()
  print(graph.get_operations()) # check all the option
  source_input = graph.get_operation_by_name('input/source_input').outputs[0]
  target_input_mete = graph.get_operation_by_name('input/target_input_mete').outputs[0]
  dist_angle = graph.get_operation_by_name('input/dist_angle').outputs[0]
  # predict by using y
  # sess.run(y, feed_dict={input_x:....,  keep_prob:1.0})
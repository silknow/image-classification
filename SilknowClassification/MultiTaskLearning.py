# -*- coding: utf-8 -*-

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# NOTICE: This work was derived from tensorflow/examples/image_retraining
# and modified to use TensorFlow Hub modules.

# pylint: disable=line-too-long

# pylint: enable=line-too-long



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import random
#import re
import sys

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

try:
    import SILKNOW_WP4_library as wp4lib
except:
    try:
        sys.path.insert(0,'./../../') 
        import SILKNOW_WP4_library as wp4lib
    except:
        print("SILKNOW WP4 Library could not be imported!")

#import urllib.parse
#import argparse

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

# The location where variable checkpoints will be stored.
CHECKPOINT_NAME = 'model.ckpt'

# A module is understood as instrumented for quantization with TF-Lite
# if it contains any of these ops.
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')


class SampleHandler:
    """Class for handling the appropriate usage of samples."""
    
    def __init__(self, numTrainSamples, numValidSamples):
        """Creates an object of class SampleHandler
        :Arguments:
          :numTrainSamples:
              Number of available training samples
          :numValidSamples:
              Number of available validation samples
    
        :Returns:
            Object of class SampleHandler
        """
        
        self.numTrain = numTrainSamples
        self.numValid = numValidSamples
        self.trainList = np.arange(0, numTrainSamples)
        self.validList = np.arange(0, numValidSamples)
        self.indexInTrain = 0
        self.indexInValid = 0
        self.epochsCompletedTrain = 0
        self.epochsCompletedValid = 0
        
        
        random.shuffle(self.trainList)
        random.shuffle(self.validList)
        
    def getTrainIndex(self):
        """Returns index of one training sample that has not yet been used in the current epoch
        :Arguments:
    
        :Returns:
            :indexTrain:
        """
        
        # Re-Shuffle all indices if all samples have been used
        if self.indexInTrain == self.numTrain:
            self.indexInTrain = 0
            self.epochsCompletedTrain += 1
            random.shuffle(self.trainList)
            
        # Get 'random' index of sample from indices of unused samples
        indexTrain = self.trainList[self.indexInTrain]
        self.indexInTrain += 1
        return indexTrain
        
    def getValidIndex(self):
        """Returns index of one validation sample that has not yet been used in the current epoch
        :Arguments:
    
        :Returns:
            :indexValid:
        """
        # Re-Shuffle all indices if all samples have been used
        if self.indexInValid == self.numValid:
            self.indexInValid = 0
            self.epochsCompletedValid += 1
            random.shuffle(self.validList)
            
        # Get 'random' index of sample from indices of unused samples
        indexValid = self.validList[self.indexInValid]
        self.indexInValid += 1
        return indexValid
 

def create_module_graph(module_spec):
  """Creates a graph and loads Hub Module into it.

  :Arguments:
      :module_spec:
          The hub.ModuleSpec for the image module being used.

  :Returns:
      :graph:
          The tf.Graph that was created.
      :bottleneck_tensor:
          The bottleneck values output by the module.
      :resized_input_tensor:
          The input images, resized as expected by the module.
      :wants_quantization:
          A boolean, whether the module has been instrumented
          with fake quantization ops.
          
  """
  height, width = hub.get_expected_image_size(module_spec)
  
  with tf.Graph().as_default() as graph:
    resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3]) 
    m = hub.Module(module_spec)                                                 
    bottleneck_tensor = m(resized_input_tensor)                                 
    wants_quantization = any(node.op in FAKE_QUANT_OPS                         
                             for node in graph.as_graph_def().node)
  return graph, bottleneck_tensor, resized_input_tensor, wants_quantization

def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.

  :Arguments:
      :dir_name:
          Path string to the folder we want to create.
  :Returns:
      Nothing.
      
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


def get_random_samples(collections_dict_MTL, how_many, module_name,
                       master_dir, image_2_label_dict,
                       class_count_dict, samplehandler, usage, session, jpeg_data_tensor, decoded_image_tensor):
    """Retrieves raw JPEG samples.

      
    :Arguments:
      :collections_dict_MTL:
          collections_dict_MTL[#label][class_label][img, ..., img]
      :how_many:
          If positive, a random sample of this size will be chosen.
          If negative, all bottlenecks will be retrieved.
      :module_name:
          The name of the image module being used.
      :master_dir (*string*)\::
          This variable is a string and contains the absolute path to the
          master file.
      :image_2_label_dict (*dictionary*)\::
            A dictionary with the image (base) name as key and a list of avaiable
            #labels as value. It's needed for the estimation of the multitask
            loss.
            image_2_label_dict[full_image_name][#label_1, ..., #label_N]
      :class_count_dict:
          xxx.
      :samplehandler:
          Object of type SampleHandler. Makes sure that every sample is used once per epoch
      :usage:
          String that is either 'train' or 'valid'. Determines if samples shall be used from
          the train or the validation set
      :session:
          The current tensorflow session
      :jpeg_data_tensor:
          Tensor to feed JPEG data into it.
      :resized_image_tensor:
            Tensor that contains the result of the preprocessing steps.
          
  :Returns:
      :ground_truths:
          tuple of reference labels for each MTL_task
          
          ground_truths = ((ref_MTL_task_1), ..., (ref_MTL_task_N))
          np.shape(ground_truths) = #MTL_task x batch_size
      :bottlenecks:
          list of feature vectors for all images in the batch.
      :filenames:
          list of the os.path.normpath(full_image_name) for all image in the
          batch.
  """
  
    assert usage == 'train' or usage == 'valid' or how_many < 0, "Usage must be 'train' or 'valid'!"
    all_images    = []
    ground_truths = []
    filenames     = []
    module_name   = (module_name.replace('://', '~')  # URL scheme.
             .replace('/', '~')  # URL and Unix paths.
             .replace(':', '~').replace('\\', '~'))
    print_mapping_index = []
  
    if how_many >= 0:
        # Retrieve a random sample of raw JPEG data
        for unused_i in range(how_many):
            # get index of unused sample
            if usage == 'train':
                image_index = samplehandler.getTrainIndex()
            if usage == 'valid':
                image_index = samplehandler.getValidIndex()
                    
                    # get label of sample
            image_name  = list(image_2_label_dict.keys())[image_index]
            temp_ground_truth = []
            for MTL_label in class_count_dict.keys():
                if MTL_label in list(image_2_label_dict[image_name].keys()):  
                    class_label = list(collections_dict_MTL[MTL_label].keys()
                                  ).index(image_2_label_dict[image_name
                                  ][MTL_label][0])
                    label_name = image_2_label_dict[image_name
                                  ][MTL_label][0]
                else:
                    class_label = -1
                    label_name = 'NaN'
                if label_name not in print_mapping_index:
                    print_mapping_index.append(label_name)
                temp_ground_truth.append(class_label)
              
            # load raw JPEG data from image path
            image_full_path = os.path.abspath(os.path.join(master_dir,image_name))
            if not tf.gfile.Exists(image_full_path):
                tf.logging.fatal('File does not exist %s', image_full_path)
            raw_images = tf.gfile.GFile(image_full_path, 'rb').read()
            try:
                image_data = session.run(decoded_image_tensor, {jpeg_data_tensor: raw_images})
            except:
                print("Failed to decode image", image_name)
                print("Original Error:")
                image_data = session.run(decoded_image_tensor, {jpeg_data_tensor: raw_images})
            all_images.append(image_data)
            ground_truths.append(temp_ground_truth)
            filenames.append(image_name)          
        ground_truths = tuple(zip(*ground_truths))
      
        return all_images, ground_truths, filenames
  
    else:
        # Retrieve all bottlenecks.
        scale_factors = []
        for image_index in range(len(image_2_label_dict.keys())):
            # get label of sample
            image_name  = list(image_2_label_dict.keys())[image_index]
            temp_ground_truth = []
            for MTL_label in class_count_dict.keys():
                if MTL_label in list(image_2_label_dict[image_name].keys()):  
                    class_label = list(collections_dict_MTL[MTL_label].keys()
                                  ).index(image_2_label_dict[image_name
                                  ][MTL_label][0])
                    label_name = image_2_label_dict[image_name
                                  ][MTL_label][0]
                else:
                    class_label = -1
                    label_name = 'NaN'
                if label_name not in print_mapping_index:
                    print_mapping_index.append(label_name)
                temp_ground_truth.append(class_label)
              
            # load raw JPEG data from image path
            image_full_path = os.path.abspath(os.path.join(master_dir,image_name))
            if not tf.gfile.Exists(image_full_path):
                tf.logging.fatal('File does not exist %s', image_full_path)
            raw_images = tf.gfile.GFile(image_full_path, 'rb').read()
            image_data = session.run(decoded_image_tensor, {jpeg_data_tensor: raw_images})
            
            # read unprocessed image to get the ratio of its sides
            orig_img_size = mpimg.imread(image_full_path).shape
            scale_x = orig_img_size[0]
            scale_y = orig_img_size[1]
            scale_factor  = scale_x /scale_y
            
            all_images.append(image_data)
            ground_truths.append(temp_ground_truth)
            filenames.append(image_name)          
            scale_factors.append(scale_factor)
        ground_truths = tuple(zip(*ground_truths))
        
        return all_images, ground_truths, filenames, scale_factors

#def add_final_retrain_ops(class_count, final_tensor_name, bottleneck_tensor,
#                          quantize_layer, is_training, learning_rate):                  
#      
#  """Adds a new softmax and fully-connected layer for training and eval.
#
#  We need to retrain the top layer to identify our new classes, so this function
#  adds the right operations to the graph, along with some variables to hold the
#  weights, and then sets up all the gradients for the backward pass.
#
#  The set up for the softmax and fully-connected layers is based on:
#  https://www.tensorflow.org/tutorials/mnist/beginners/index.html
#
#  :Arguments:
#      :class_count:
#          Integer of how many categories of things we're trying to
#          recognize.
#      :final_tensor_name:
#          Name string for the new final node that produces results.
#      :bottleneck_tensor:
#          The output of the main CNN graph.
#      :quantize_layer:
#          Boolean, specifying whether the newly added layer should be
#          instrumented for quantization with TF-Lite.
#      :is_training:
#          Boolean, specifying whether the newly add layer is for training
#          or eval.
#
#  :Returns:
#      The tensors for the training and cross entropy results, and tensors for the
#      bottleneck input and ground truth input.
#  """
#  batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
#  assert batch_size is None, 'We want to work with arbitrary batch size.'
#  with tf.name_scope('input'):
#    bottleneck_input = tf.placeholder_with_default(
#        bottleneck_tensor,
#        shape=[batch_size, bottleneck_tensor_size],
#        name='BottleneckInputPlaceholder')
#
#    ground_truth_input = tf.placeholder(
#        tf.int64, [batch_size], name='GroundTruthInput')
#    tf.summary.histogram("class_distribution", ground_truth_input)
#    
#  # final layer 
#  init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
#  # Organizing the following ops so they are easier to see in TensorBoard.
#  # (base classes)
#  dense_1 = tf.layers.dense(inputs=bottleneck_input,
#                            units=1000,
#                            use_bias=True,
#                            kernel_initializer = init,
#                            activation=tf.nn.relu,
#                            name='1st_fc_layer')
#  
#  dense_2 = tf.layers.dense(inputs=dense_1,
#                            units=100,
#                            use_bias=True,
#                            kernel_initializer = init,
#                            activation=tf.nn.relu,
#                            name='2nd_fc_layer')
#  
#  logits = tf.layers.dense(inputs=dense_2,
#                            units=class_count,
#                            use_bias=True,
#                            activation=None,
#                            name='3rd_fc_layer')
#  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)              
#
#  # The tf.contrib.quantize functions rewrite the graph in place for
#  # quantization. The imported model graph has already been rewritten, so upon
#  # calling these rewrites, only the newly added final layer will be
#  # transformed.
#  if quantize_layer:
#    if is_training:
#      tf.contrib.quantize.create_training_graph()
#    else:
#      tf.contrib.quantize.create_eval_graph()
#
#  tf.summary.histogram('activations', final_tensor)
#  
#  with tf.name_scope('cross_entropy_sum'):                                       
#    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
#        labels=ground_truth_input, logits=logits, reduction=tf.losses.Reduction.NONE)
#    cross_entropy_sum = tf.reduce_sum(cross_entropy)
#
#  tf.summary.scalar('cross_entropy_sum', cross_entropy_sum)
#
#  # If this is an eval graph, we don't need to add loss ops or an optimizer.
#  if not is_training:
#    return None, None, bottleneck_input, ground_truth_input, final_tensor
#
#  with tf.name_scope('train'):
#    optimizer = tf.train.AdamOptimizer(learning_rate)
#    grad_var_list = optimizer.compute_gradients(cross_entropy_sum,
#                                                tf.trainable_variables())
#    for (grad, var) in grad_var_list:
#        tf.summary.histogram(var.name + '/gradient', grad)
#        tf.summary.histogram(var.op.name, var)
#    train_step = optimizer.apply_gradients(grad_var_list)
##    train_step = optimizer.minimize(cross_entropy_sum)
#    
#
#  return (train_step, cross_entropy_sum, bottleneck_input, ground_truth_input,
#          final_tensor)


def add_final_retrain_ops_MTL(class_count_dict, final_tensor_name, bottleneck_tensor,
                              quantize_layer, is_training, learning_rate,
                              num_joint_fc_layer, num_nodes_joint_fc, nodes_prop_2_num_tasks, 
                              num_task_stop_gradient, weight_decay):                        
  """Adds the classification networks for multitask learning.

  We need to retrain the top layer to identify our new classes, so this function
  adds the right operations to the graph, along with some variables to hold the
  weights, and then sets up all the gradients for the backward pass.

  The set up for the softmax and fully-connected layers is based on:
  https://www.tensorflow.org/tutorials/mnist/beginners/index.html

  :Arguments:
      :class_count_dict:
          A dictionary containing the number of classes per multi learning task.
          
          class_count_dict[im_label][num_class_label]
      :bottleneck_tensor:
          The output tensor of the main CNN graph producing the image features.
      :quantize_layer:
          Boolean, specifying whether the newly added layer should be
          instrumented for quantization with TF-Lite.
      :is_training:
          Boolean, specifying whether the newly add layer is for training
          or eval.
      :learning_rate:
          Float. The learning rate for the backprop algorithm.
      :num_joint_fc_layer:
          Interger. Number of joint FC layers. Each layer will have num_nodes_joint_fc nodes.
      :num_nodes_joint_fc:
          Interger. Number of nodes of each joint FC layer.
      :nodes_prop_2_num_tasks:
          Bool.  If True, the num_nodes_joint_fc will be expanded if more than two
            tasks will be learnt. For each additional task, half of
            num_nodes_joint_fc nodes will be added to each joint
            fully-connected layer.
      :num_task_stop_gradient:
          Integer. 
          If zero, only complete samples will contribute to the joint fc layer. 
          If negative, all samples will contribute to the joint fc layer.
          Else, samples may have up to num_task_stop_gradient 
          missing class labels to contribute to the joint fc layer.
      :weight_decay:
          Scaling factor for the L2-Loss.
          

  :Returns:
      The tensors for the training and cross entropy results, and tensors for the
      bottleneck input and ground truth input.
  """
  # 1.1 Get the size of the bottlenecks and the batch size.
  # 1.2 Create the inputs for the classification networks (bottleneck_input).
  # 1.2 Create one ground_truth_input-tensor for each MTL_task and store it in
  #     a tuple
  ground_truth_list = []
  batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
  assert batch_size is None, 'We want to work with arbitrary batch size.'
  with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor,
        shape=[batch_size, bottleneck_tensor_size],
        name='BottleneckInputPlaceholder')
    filename_input = tf.placeholder(
        tf.string,
        shape=[batch_size],
        name='FilenameInputPlaceholder')
    for MTL_task in class_count_dict.keys():
        ground_truth_input = tf.placeholder(
            tf.int64, [batch_size], name='GroundTruthInput'+MTL_task)
        ground_truth_list.append(ground_truth_input)
    ground_truth_MTL = tuple(ground_truth_list)
    for MTL_ind, MTL_key in enumerate(class_count_dict.keys()):
        tf.summary.histogram("class_distribution"+MTL_key, ground_truth_MTL[MTL_ind])
  
  # 1.3 Initializer for the layer weights
  init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
  
  # 2.1 Create a fc for all tasks together to adapt the features from the other
  #     domain.
  # 2.2 Create one classification network per classification task
  if nodes_prop_2_num_tasks:
      if len(class_count_dict.keys()) > 2:
          number_jfc_nodes = num_nodes_joint_fc + int(
                             num_nodes_joint_fc/2 * (len(class_count_dict.keys())-2))
      else:
          number_jfc_nodes = num_nodes_joint_fc
  else:
      number_jfc_nodes = num_nodes_joint_fc
  
  with tf.variable_scope("joint_layers_MTL"):
      joint_fc = tf.layers.dense(inputs=bottleneck_input,
                                     units=number_jfc_nodes,
                                     use_bias=True,
                                     kernel_initializer = init,
                                     activation=tf.nn.relu,
                                     name='joint_fc_layer')
      if num_joint_fc_layer > 1:
          for num_joint_fc in range(num_joint_fc_layer-1):
              joint_fc = tf.layers.dense(inputs=joint_fc,
                                         units=number_jfc_nodes,
                                         use_bias=True,
                                         kernel_initializer = init,
                                         activation=tf.nn.relu,
                                         name='joint_fc_layer' + str(num_joint_fc))
  final_tensor_MTL = []
  logits_MTL       = []
  
  # Count missing tasks for every sample, i.e. to which degree one sample is incomplete
  assert num_task_stop_gradient <= len(class_count_dict.keys()), "num_task_stop_gradient has to be smaller or equal to the number of tasks!"
  if num_task_stop_gradient < 0: num_task_stop_gradient=len(class_count_dict.keys())
  count_incomplete = tf.fill(tf.shape(ground_truth_MTL[0]), 0.0)
  for MTL_ind, MTL_task in enumerate(class_count_dict.keys()):
      temp_labels = ground_truth_MTL[MTL_ind]
      temp_zero   = tf.fill(tf.shape(temp_labels), 0.0)
      temp_one    = tf.fill(tf.shape(temp_labels), 1.0)
      temp_incomplete = tf.where(tf.equal(temp_labels, -1), temp_one, temp_zero)
      count_incomplete = count_incomplete + temp_incomplete
  mask_contribute = tf.math.greater_equal(tf.cast(num_task_stop_gradient, tf.float32), count_incomplete)
  
  for MTL_ind, MTL_task in enumerate(class_count_dict.keys()):
      
      with tf.variable_scope("stop_incomplete_gradient_"+MTL_task):
          # For each task, split complete from incomplete samples. That way
          # the gradient can be stopped for incomplete samples so that they won't 
          # contribute to the update of the joint fc layer(s)
          temp_labels = ground_truth_MTL[MTL_ind]
          temp_zero   = tf.fill(tf.shape(joint_fc), 0.0)
          contrib_samples   = tf.where(mask_contribute, joint_fc, temp_zero)
          nocontrib_samples = tf.stop_gradient(tf.where(mask_contribute, temp_zero, joint_fc))
          joint_fc_ = contrib_samples + nocontrib_samples
          tf.summary.histogram('activations_FC_contribute_'+ str(MTL_task), contrib_samples) 
          tf.summary.histogram('activations_FC_nocontribute_'+ str(MTL_task), nocontrib_samples) 
      
      dense_0 = tf.layers.dense(inputs=joint_fc_,
                                units=100,
                                use_bias=True,
                                kernel_initializer = init,
                                activation=tf.nn.relu,
                                name='1st_fc_layer_' + str(MTL_task))
      logits_0 = tf.layers.dense(inputs=dense_0,
                               units=class_count_dict[MTL_task],
                               use_bias=True,
                               kernel_initializer = init,
                               activation=None,
                               name='2nd_fc_layer_' + str(
                                        MTL_task) + '_' + str(
                                        class_count_dict[MTL_task]) + '_classes')
      final_tensor_0 = tf.nn.softmax(logits_0,
                                     name=final_tensor_name + '_' + str(MTL_task))
      print("Final Tensor:", final_tensor_name + '_' + str(MTL_task))
      final_tensor_MTL.append(final_tensor_0)
      logits_MTL.append(logits_0)
      
      tf.summary.histogram('activations_'+ str(MTL_task), final_tensor_0) 
      
  final_tensor_MTL = tf.tuple(final_tensor_MTL, name=final_tensor_name)
  logits_MTL       = tuple(logits_MTL)
  
  # The tf.contrib.quantize functions rewrite the graph in place for
  # quantization. The imported model graph has already been rewritten, so upon
  # calling these rewrites, only the newly added final layer will be
  # transformed.
  if quantize_layer:
    if is_training:
      tf.contrib.quantize.create_training_graph()
    else:
      tf.contrib.quantize.create_eval_graph()
  
  # 3.1 Create a joint loss for all classification tasks
  # 3.2 Monitore all losses separately to indentify the progress of the
  #     individual learning tasks.
  cross_entropy_sum = cross_entropy_loss_MTL(ref_tuple      = ground_truth_MTL,
                                              logits_tuple   = logits_MTL,
                                              class_count_dict = class_count_dict,
                                              weight_decay = weight_decay)

  tf.summary.scalar('cross_entropy_overall', cross_entropy_sum)

  # If this is an eval graph, we don't need to add loss ops or an optimizer.
  if not is_training:
    return (None, None, bottleneck_input, ground_truth_MTL, final_tensor_MTL,
            filename_input)
  
  with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grad_var_list = optimizer.compute_gradients(cross_entropy_sum,
                                                tf.trainable_variables())
    for (grad, var) in grad_var_list:
        tf.summary.histogram(var.name + '/gradient', grad)
        tf.summary.histogram(var.op.name, var)
    train_step = optimizer.apply_gradients(grad_var_list)
#    train_step = optimizer.minimize(cross_entropy_sum)

  return (train_step, cross_entropy_sum, bottleneck_input, ground_truth_MTL,
          final_tensor_MTL, filename_input)


def create_computation_graph(module_spec, num_finetune_layers, class_count_dict, final_tensor_name,
                              is_training, learning_rate,
                              num_joint_fc_layer, num_nodes_joint_fc, nodes_prop_2_num_tasks, 
                              num_task_stop_gradient, aug_set_dict, weight_decay):
    """Creates the complete computation graph, including feature extraction,
       data augmentation and classification.
    
    :Arguments:
      :module_spec:
          The hub.ModuleSpec for the image module being used.
      :class_count_dict:
          A dictionary containing the number of classes per multi learning task.         
          class_count_dict[im_label][num_class_label]
      :is_training:
          Boolean, specifying whether the newly add layer is for training
          or eval.
      :learning_rate:
          Float. The learning rate for the backprop algorithm.
      :num_joint_fc_layer:
          Integer. Number of joint FC layers. Each layer will have num_nodes_joint_fc nodes.
      :num_nodes_joint_fc:
          Interger. Number of nodes of each joint FC layer.
      :nodes_prop_2_num_tasks:
          Bool.  If True, the num_nodes_joint_fc will be expanded if more than two
            tasks will be learnt. For each additional task, half of
            num_nodes_joint_fc nodes will be added to each joint
            fully-connected layer.
      :num_task_stop_gradient:
          Integer. 
          If zero, only complete samples will contributeto the joint fc layer. 
          If negative, all samples will contribute to the joint fc layer.
          Else, samples may have up to num_task_stop_gradient 
          missing class labels to contribute to the joint fc layer.
      :aug_set_dict:
          Dictionary with information about data augmentation. 
          For details see function add_data_augmentation in WP4-Library.
      :weight decay:
          Scaling factor for the L2-Loss.
          

    """
    """*********************BEGIN: ADD JPEG DECODING****************************"""
    
    """*********************END: ADD JPEG DECODING****************************"""
    
    """*********************BEGIN: CREATE MODULE GRAPH****************************"""
    height, width = hub.get_expected_image_size(module_spec)
#  print('Input size of the first pre-trained layer:', height, width, 3)
  
    with tf.Graph().as_default() as graph:
        
        with tf.variable_scope("moduleLayers") as scope:
            input_image_tensor = tf.placeholder(tf.float32, [None, height, width, 3], name="input_img") 
            m = hub.Module(module_spec)                                      
            bottleneck_tensor = m(input_image_tensor)                             
            wants_quantization = any(node.op in FAKE_QUANT_OPS                         
                                     for node in graph.as_graph_def().node)
        
        # get scope/names of variables from layers that will be retrained 
        module_vars        = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='moduleLayers')
        pre_names          = '/'.join(module_vars[0].name.split('/')[:3])
        module_vars_names  = np.asarray([v.name.split('/')[3] for v in module_vars])[::-1]
        
        unique_module_vars_names = []
        for n in module_vars_names:
            if len(unique_module_vars_names) == 0 or (not n == unique_module_vars_names[-1]):
                unique_module_vars_names += [n]
                
   # """*********************End: CREATE MODULE GRAPH****************************"""
    
   #"""*********************BEGIN: ADD FINAL RETRAIN OPS MTL*********************"""
    
        with tf.variable_scope("customLayers") as scope:
            (train_step, 
             cross_entropy, 
             bottleneck_input,
             ground_truth_input, 
             final_tensor,
             filename_input) = add_final_retrain_ops_MTL(
                             class_count_dict, final_tensor_name, bottleneck_tensor,
                             wants_quantization, is_training=True,
                             learning_rate = learning_rate, num_joint_fc_layer=num_joint_fc_layer,
                             num_nodes_joint_fc=num_nodes_joint_fc,
                             nodes_prop_2_num_tasks=nodes_prop_2_num_tasks,
                             num_task_stop_gradient = num_task_stop_gradient,
                             weight_decay = weight_decay)
        
    #"""*********************END: ADD FINAL RETRAIN OPS MTL*********************"""
    
    #"""*********************BEGIN: ADD CUSTOM FINETUNING *********************"""
        trainable_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='customLayers')
        
        for v in range(num_finetune_layers):
            trainable_variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=pre_names+'/'+unique_module_vars_names[v]))
    """*********************End: ADD CUSTOM FINETUNING****************************"""
    
    return (train_step, cross_entropy, ground_truth_input,
          final_tensor, filename_input, graph, input_image_tensor, trainable_variables)


def cross_entropy_loss_MTL(ref_tuple, logits_tuple, class_count_dict, weight_decay=1e-3):
    """Computes the cross entropy for multitask learning.
    
    :Arguments:
        :ref_tuple:
            Tuple with ground truth for each sample.
        :logits_tuple:
            Tuple with predicted classes for each sample.
        :class_count_dict:
            A dictionary containing the number of classes per multi learning task.         
            class_count_dict[im_label][num_class_label]
        :weight_decay:
            Scaling factor for the L2-Loss.
    
    :Returns:
        :cross_entropy_MTL:
            The cross entropy loss
    """
    all_cross_entropy = []
    tensor_created = False
    print(ref_tuple)
    print(logits_tuple)
    for MTL_ind in range(np.shape(ref_tuple)[0]):      
        temp_class_count = class_count_dict[
                                list(class_count_dict.keys())[MTL_ind]]
        temp_labels = ref_tuple[MTL_ind]
        temp_logits = logits_tuple[MTL_ind]
        
        x = tf.fill(tf.shape(temp_labels), False)
        y = tf.fill(tf.shape(temp_labels), True)
        data_gap_ind = tf.where(tf.equal(temp_labels, -1), x, y)
        temp_labels  = tf.boolean_mask(temp_labels, data_gap_ind)
        temp_logits  = tf.boolean_mask(temp_logits, data_gap_ind)
        
        temp_labels_one_hot = tf.one_hot(indices=temp_labels,
                                         depth=temp_class_count,
                                         on_value=1.0,
                                         off_value=0.0)        
        temp_cross_entropy = tf.losses.softmax_cross_entropy(
                                        onehot_labels=temp_labels_one_hot,
                                        logits=temp_logits,
                                        reduction=tf.losses.Reduction.NONE)
        temp_cross_entropy = tf.reduce_sum(temp_cross_entropy)
        tf.summary.scalar("cross_entropy_"+list(class_count_dict.keys())[MTL_ind],
                          temp_cross_entropy)        
        if not tensor_created:
            all_cross_entropy = temp_cross_entropy
            tensor_created    = True
        else:
            all_cross_entropy = tf.reduce_sum([all_cross_entropy, temp_cross_entropy])
            # problem with stacking for more than 2 tasks, because stack takes
            # scalars
            # -> tf.concat instead, but not suitable for <=2 tasks
#            all_cross_entropy = tf.stack([all_cross_entropy, temp_cross_entropy], 0)
            
    lossL2 = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    cross_entropy_MTL = tf.reduce_sum(all_cross_entropy) + weight_decay * lossL2
#    cross_entropy_MTL = tf.reduce_sum(all_cross_entropy)
    return cross_entropy_MTL


#def add_evaluation_step(result_tensor, ground_truth_tensor, bool_MTL):
#  """Inserts the operations we need to evaluate the accuracy of our results.
#
#  :Arguments:
#    :result_tensor:
#        The new final node that produces results.
#    :ground_truth_tensor:
#        The node we feed ground truth data into.
#    :bool_MTL:
#        xxx.
#
#  :Returns:
#    Tuple of (evaluation step, prediction).
#  """
#  if not bool_MTL:
#      with tf.name_scope('accuracy'):
#        with tf.name_scope('correct_prediction'):
#          prediction = tf.argmax(result_tensor, 1)
#          correct_prediction = tf.equal(prediction, ground_truth_tensor)
#        with tf.name_scope('accuracy'):
#          evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#      tf.summary.scalar('overall_accuracy', evaluation_step)
#      return evaluation_step, prediction
#  else:
#      1/0
#      return -1

def add_evaluation_step_MTL(result_tensor, ground_truth_tensor, class_count_dict):
    """Inserts the operations we need to evaluate the accuracy of our results.

    :Arguments:
        :result_tensor:
            The new final node that produces results.
        :ground_truth_tensor:
            The node we feed ground truth data into.
        :class_count_dict:
            A dictionary containing the number of classes per multi learning task.         
            class_count_dict[im_label][num_class_label]

     :Returns:
        :prediction_perMTL: 
            Predictions per task.
        :overall_acc_perMTL:
            Overall accuracy per task.
        :prediction_all: 
            All predictions.
        :overall_acc_all: 
            Overall accuracy over all samples.
        :ground_truth_all: 
            All groundtruths.
        :ground_truth_perMTL:
            Groundtruths per task.
        :prediction_perMTL_inc: 
            Predictions per task, including incomplete samples.
        :ground_truth_perMTL_inc: 
            Groundtruth per task, including incomplete samples.
        :correct_prediction_complete:
            Percentage of correctly predicted tasks, per sample.
            
    """
    prediction_perMTL  = []
    overall_acc_perMTL = []
    ground_truth_perMTL = [] # Needed to exclude samples with data gaps (-1)
    first_task_passed = False
    
    # Those arrays will include predictions/groundtruths for incomplete samples
    # unknown GTs will be represented by -1
    prediction_perMTL_inc = []
    ground_truth_perMTL_inc = []
   
    # Bool-Filter for complete samples only
    filter_complete = tf.fill(tf.shape(ground_truth_tensor[0]), True)
        
    for MTL_ind, MTL_task in enumerate(class_count_dict.keys()):
        # Get task-specific result and GT from tensors
        temp_result_tensor = result_tensor[MTL_ind]
        temp_ground_truth  = ground_truth_tensor[MTL_ind]
        
        # Save Prediction/GT
        prediction_perMTL_inc.append(tf.argmax(temp_result_tensor, 1))
        ground_truth_perMTL_inc.append(temp_ground_truth)
        
        # Bool-Filter for complete samples only
        x = tf.fill(tf.shape(temp_ground_truth), False)
        filter_complete = tf.where(tf.equal(temp_ground_truth, -1), x, filter_complete)
        
        # Mask out unkown labels     
        x = tf.fill(tf.shape(temp_ground_truth), False)
        y = tf.fill(tf.shape(temp_ground_truth), True)
        data_gap_ind       = tf.where(tf.equal(temp_ground_truth, -1), x, y)
        temp_result_tensor = tf.boolean_mask(temp_result_tensor, data_gap_ind)
        temp_ground_truth  = tf.boolean_mask(temp_ground_truth, data_gap_ind)
        
        # Prediction and Quality Measure for known labels only
        temp_prediction         = tf.argmax(temp_result_tensor, 1)
        temp_correct_prediction = tf.equal(temp_prediction, temp_ground_truth)
        temp_overall_accuracy   = tf.reduce_mean(tf.cast(temp_correct_prediction, tf.float32))
        
        prediction_perMTL.append(temp_prediction)
        overall_acc_perMTL.append(temp_overall_accuracy)
        ground_truth_perMTL.append(temp_ground_truth)
        tf.summary.scalar("OA_"+MTL_task, temp_overall_accuracy)
        
        if not first_task_passed:
            prediction_all = temp_prediction
            correct_preditcion_all = temp_correct_prediction
            ground_truth_all = temp_ground_truth
            first_task_passed = True
        else:
            prediction_all = tf.concat([prediction_all,
                                        temp_prediction],
                                       0)
            correct_preditcion_all = tf.concat([correct_preditcion_all, 
                                                temp_correct_prediction],
                                       0)
            ground_truth_all = tf.concat([ground_truth_all,
                                          temp_ground_truth],
                                       0)
            
    # get prediction/gt/quality for complete samples only
    for MTL_ind, MTL_task in enumerate(class_count_dict.keys()):
        
         # Get task-specific result and GT from tensors
        temp_result_tensor = result_tensor[MTL_ind]
        temp_ground_truth  = ground_truth_tensor[MTL_ind]
        
        # Prediction and Quality Measure for known labels only
        temp_prediction         = tf.argmax(temp_result_tensor, 1)
        temp_correct_prediction = tf.cast(tf.equal(temp_prediction, temp_ground_truth), tf.float32)
        
        # Sum up the correct predictions per sample
        if MTL_ind == 0:
            correct_prediction_complete = temp_correct_prediction
        else:
            correct_prediction_complete = correct_prediction_complete + temp_correct_prediction
    
    # Set accuracy of incomplete samples to -1
    correct_prediction_complete = correct_prediction_complete / len(class_count_dict.keys())
    y = tf.fill(tf.shape(temp_ground_truth), -1.)
    correct_prediction_complete = tf.where(filter_complete, correct_prediction_complete, y)
            
    overall_acc_all = tf.reduce_mean(tf.cast(correct_preditcion_all,
                                                       tf.float32))
    tf.summary.scalar("OA_all", overall_acc_all)
    prediction_perMTL       = tuple(prediction_perMTL)
    overall_acc_perMTL      = tuple(overall_acc_perMTL)
    ground_truth_perMTL     = tuple(ground_truth_perMTL)
    prediction_perMTL_inc   = tuple(prediction_perMTL_inc)
    ground_truth_perMTL_inc = tuple(ground_truth_perMTL_inc)
    
#    prediction_perMTL = tf.identity(prediction_perMTL, name="prediction_perMTL")
#    overall_acc_perMTL = tf.identity(overall_acc_perMTL, name="overall_acc_perMTL")
#    ground_truth_perMTL = tf.identity(ground_truth_perMTL, name="ground_truth_perMTL")
#    prediction_perMTL_inc = tf.identity(prediction_perMTL_inc, name="prediction_perMTL_inc")
#    ground_truth_perMTL_inc = tf.identity(ground_truth_perMTL_inc, name="ground_truth_perMTL_inc")
#    prediction_all = tf.identity(prediction_all, name="prediction_all")
#    ground_truth_all = tf.identity(ground_truth_all, name="ground_truth_all")
#    correct_prediction_complete = tf.identity(correct_prediction_complete, name="correct_prediction_complete")
    
    return (prediction_perMTL, overall_acc_perMTL,
            prediction_all, overall_acc_all, ground_truth_all, ground_truth_perMTL,
            prediction_perMTL_inc, ground_truth_perMTL_inc, correct_prediction_complete)


## function not used anymore!
#def build_eval_session(module_spec, class_count, final_tensor_name,
#                       learning_rate, bool_MTL, class_count_dict,
#                       num_joint_fc_layer, num_nodes_joint_fc, nodes_prop_2_num_tasks, 
#                       num_finetune_layers, num_task_stop_gradient):
#  """Builds an restored eval session without train operations for exporting.
#
#  :Arguments:
#    :module_spec:
#        The hub.ModuleSpec for the image module being used.
#    :class_count:
#        Number of classes
#    :bool_MTL:
#        xxx.
#    :class_count_dict:
#        xxx.
#
#  :Returns:
#    Eval session containing the restored eval graph.
#    The bottleneck input, ground truth, eval step, and prediction tensors.
#    
#  """
#  # If quantized, we need to create the correct eval graph for exporting.
#  eval_graph, bottleneck_tensor, resized_input_tensor, wants_quantization = (
#      create_module_graph(module_spec))
#
#  eval_sess = tf.Session(graph=eval_graph)
#  with eval_graph.as_default():
#    # Add the new layer for exporting.
#    if not bool_MTL:
#        (_, _, bottleneck_input,
#         ground_truth_input, final_tensor) = add_final_retrain_ops(
#             class_count, final_tensor_name, bottleneck_tensor,
#             wants_quantization, is_training=False, learning_rate=learning_rate)
#    else:
#        (_, _, bottleneck_input,
#         ground_truth_input, final_tensor,
#         _) = add_final_retrain_ops_MTL(
#                 class_count_dict, final_tensor_name, bottleneck_tensor,
#                 wants_quantization, is_training=False,
#                 learning_rate = learning_rate, num_joint_fc_layer=num_joint_fc_layer,
#                 num_nodes_joint_fc=num_nodes_joint_fc,
#                 nodes_prop_2_num_tasks=nodes_prop_2_num_tasks,
#                 num_task_stop_gradient=num_task_stop_gradient)
#
#    # Now we need to restore the values from the training graph to the eval
#    # graph.
#    tf.train.Saver().restore(eval_sess, CHECKPOINT_NAME)
#
#    if not bool_MTL:
#        evaluation_step, prediction = add_evaluation_step(final_tensor,
#                                                      ground_truth_input,
#                                                      bool_MTL)
#    else:
#        (prediction_perMTL, overall_acc_perMTL,
#        prediction_all, overall_acc_all,
#        ground_truth_all, ground_truth_perMTL,
#        prediction_perMTL_inc, ground_truth_perMTL_inc, correct_prediction_complete) = add_evaluation_step_MTL(final_tensor,
#                                                         ground_truth_input,
#                                                         bool_MTL,
#                                                         class_count_dict)
#  if not bool_MTL:
#      return (eval_sess, resized_input_tensor, bottleneck_input, ground_truth_input,
#          evaluation_step, prediction)
#  else:
#      return (eval_sess, resized_input_tensor, bottleneck_input, ground_truth_input,
#              prediction_perMTL, overall_acc_perMTL,
#              prediction_all, overall_acc_all, ground_truth_all,
#              ground_truth_perMTL)
#
## function not used anymore!
#def build_eval_session_NEW(module_spec, class_count, final_tensor_name,
#                       learning_rate, bool_MTL, class_count_dict,
#                       num_joint_fc_layer, num_nodes_joint_fc, nodes_prop_2_num_tasks, 
#                       num_finetune_layers, num_task_stop_gradient, aug_set_dict):
#    (train_step, 
#     cross_entropy,
#     ground_truth_input,
#     final_tensor, 
#     filename_input,
#     graph,
#     input_image_tensor,
#     trainable_variables) = create_computation_graph(module_spec            = module_spec, 
#                                                 num_finetune_layers    = num_finetune_layers, 
#                                                 class_count_dict       = class_count_dict, 
#                                                 final_tensor_name      = final_tensor_name,
#                                                 is_training            = False, 
#                                                 learning_rate          = learning_rate,
#                                                 num_joint_fc_layer     = num_joint_fc_layer, 
#                                                 num_nodes_joint_fc     = num_nodes_joint_fc, 
#                                                 nodes_prop_2_num_tasks = nodes_prop_2_num_tasks, 
#                                                 num_task_stop_gradient = num_task_stop_gradient,
#                                                 aug_set_dict           = aug_set_dict
#                             )
#    eval_sess = tf.Session(graph=graph)
#    
#    tf.train.Saver().restore(eval_sess, CHECKPOINT_NAME)
#
#    if not bool_MTL:
#        evaluation_step, prediction = add_evaluation_step(final_tensor,
#                                                      ground_truth_input,
#                                                      bool_MTL)
#    else:
#        (prediction_perMTL, overall_acc_perMTL,
#        prediction_all, overall_acc_all,
#        ground_truth_all, ground_truth_perMTL,
#        prediction_perMTL_inc, ground_truth_perMTL_inc, correct_prediction_complete) = add_evaluation_step_MTL(final_tensor,
#                                                         ground_truth_input,
#                                                         bool_MTL,
#                                                         class_count_dict)
#
#
#    return (eval_sess, ground_truth_input,
#              prediction_perMTL, overall_acc_perMTL,
#              prediction_all, overall_acc_all, ground_truth_all,
#              ground_truth_perMTL)
#   
    
def save_graph(session, saver, logpath):
    print("Saving graph to", logpath)
    _ = saver.save(session, logpath)
        

#def split_collections_dict(collections_dict, validation_percentage):
#    """Splits the collection into two collections
#    
#    :Arguments:
#        :collections_dict (*dictionary*)\::
#            The keys of this dictionary are the class labels of
#            one specific task, the values are lists of images.
#            collections_dict[task][class label][images]
#        :validation_percentage (*int*)\::
#            Percentage that is used for validation. For example, a value of 20
#            means that 80% of the images in image_2_label_dict will be used for
#            training, the other 20% will be used for validation.
#    
#    :Returns:
#        :collections_dict_train (*dictionary*)\::
#            The keys of this dictionary are the class labels of
#            one specific task, the values are lists of images used for training.
#            collections_dict[task][class label][images]
#        :collections_dict_val (*dictionary*)\::
#            The keys of this dictionary are the class labels of
#            one specific task, the values are lists of images used for validation.
#            collections_dict[task][class label][images]
#    """
#    collections_dict_train = {}
#    collections_dict_val   = {}
#    for coll_key in collections_dict.keys():
#        cur_num_train = int(np.floor(len(collections_dict[coll_key])*\
#                                (100-validation_percentage)/100))
#        cur_num_val = len(collections_dict[coll_key])-cur_num_train        
#        collections_dict_train[coll_key] = collections_dict[coll_key]\
#                                                    [0:cur_num_train]
#        collections_dict_val[coll_key] = collections_dict[coll_key]\
#                                    [cur_num_train:cur_num_train+cur_num_val] 
#    
#    return collections_dict_train, collections_dict_val


def split_collections_dict_MTL(collections_dict_MTL, image_2_label_dict,
                               validation_percentage, master_dir):
    """Splits the collection into two collections.
    
    :Arguments:
        :collections_dict_MTL (*dictionary*)\::
            It's a dict of dicts. The keys are the tasks, the values are again
            dictionaries. The keys of those dictionaries are the class labels of
            the aforementioned task, the values are lists of images.
            collections_dict_MTL[task][class label][images]
        :image_2_label_dict (*dictionary*)\::
            A dictionary with the image (base) name as key and a list of avaiable
            #labels as value. It's needed for the estimation of the multitask
            loss.
            image_2_label_dict[full_image_name][#label_1, ..., #label_N]
        :validation_percentage (*int*)\::
            Percentage that is used for validation. For example, a value of 20
            means that 80% of the images in image_2_label_dict will be used for
            training, the other 20% will be used for validation.
         :master_dir (*string*)\::
            This variable is a string and contains the name of the master file.
            The master file has to contain a list of the "collection.txt".
            All "collection.txt" have to be in the same folder as the master
            file.
            In the "collection.txt" are relative paths to the images and the
            according class label listed. The paths in a "collection.txt" has to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg".
            The "collection.txt" has to have a header "#image\t#Label" and the
            following lines "path_with_image\timage_label".
    
    :Returns:
        :collections_dict_MTL_train (*dictionary*)\::
            It's a dict of dicts. The keys are the tasks, the values are again
            dictionaries. The keys of those dictionaries are the class labels of
            the aforementioned task, the values are lists of images used for training.
            collections_dict_MTL[task][class label][images]
        :collections_dict_MTL_val (*dictionary*)\::
            It's a dict of dicts. The keys are the tasks, the values are again
            dictionaries. The keys of those dictionaries are the class labels of
            the aforementioned task, the values are lists of images used for validation.
            collections_dict_MTL[task][class label][images]
        :image_2_label_dict_train (*dictionary*)\::
            A dictionary with the image (base) name as key and a list of avaiable
            #labels as value. It's needed for the estimation of the multitask
            loss. The images are used for training.
            image_2_label_dict_train[full_image_name][#label_1, ..., #label_N]
        :image_2_label_dict_val (*dictionary*)\::
            A dictionary with the image (base) name as key and a list of avaiable
            #labels as value. It's needed for the estimation of the multitask
            loss. The images are used for validation. 
            image_2_label_dict_val[full_image_name][#label_1, ..., #label_N]
            
    """
    
    collections_dict_MTL_train = {}
    collections_dict_MTL_val   = {}
    image_2_label_dict_train   = {}
    image_2_label_dict_val     = {}
    num_train = int(np.floor(len(list(image_2_label_dict.keys()))*\
                             (100-validation_percentage)/100))
#    num_val   = len(list(image_2_label_dict.keys())) - num_train
    
    
    # Randomly choose samples for training and validation
    idx_all = np.asarray(range(len(list(image_2_label_dict.keys()))))
    random.seed(5517)
    random.shuffle(idx_all)
    idx_train = idx_all[:num_train]
    idx_valid = idx_all[num_train:]
    for image_index in idx_train:
        image_key   = list(image_2_label_dict.keys())[image_index]
        image_2_label_dict_train[image_key] = image_2_label_dict[image_key]
    for image_index in idx_valid:
        image_key   = list(image_2_label_dict.keys())[image_index]
        image_2_label_dict_val[image_key] = image_2_label_dict[image_key]
    
            
    # Iterate over tasks (timespan/place/...)
    for MTL_key in collections_dict_MTL.keys():   
        # Iterate over related class labels
        for class_key in collections_dict_MTL[MTL_key].keys():
            # all images with that class label
            for image in collections_dict_MTL[MTL_key][class_key]:
                fullname_image = os.path.abspath(os.path.join(master_dir,
                                                              image))
                # Part of Training
                if fullname_image in image_2_label_dict_train.keys():
                    if MTL_key not in collections_dict_MTL_train.keys():
                        temp_dict = {}
                        temp_dict[class_key] = [fullname_image]
                        collections_dict_MTL_train[MTL_key] = temp_dict
                    else:
                        if class_key not in collections_dict_MTL_train[MTL_key].keys():
                            collections_dict_MTL_train[MTL_key][class_key] = [fullname_image]
                        else:
                            collections_dict_MTL_train[MTL_key][class_key].append(fullname_image)
                # Part of Validation
                else:
                    if MTL_key not in collections_dict_MTL_val.keys():
                        temp_dict = {}
                        temp_dict[class_key] = [fullname_image]
                        collections_dict_MTL_val[MTL_key] = temp_dict
                    else:
                        if class_key not in collections_dict_MTL_val[MTL_key].keys():
                            collections_dict_MTL_val[MTL_key][class_key] = [fullname_image]
                        else:
                            collections_dict_MTL_val[MTL_key][class_key].append(fullname_image)
#    print('An equal class distribution in training and validation is not',
#          'guaranteed by now.')
    for MTL_task in collections_dict_MTL_train.keys():
        print('\nMTL_task', MTL_task)
        for class_ in collections_dict_MTL_train[MTL_task]:
            print(class_)
            print('all', len(collections_dict_MTL[MTL_task][class_]))
            print('train', len(collections_dict_MTL_train[MTL_task][class_]))
            print('val', len(collections_dict_MTL_val[MTL_task][class_]))
    return (collections_dict_MTL_train, collections_dict_MTL_val,
            image_2_label_dict_train, image_2_label_dict_val)

def sort_out_incomplete_samples(collections_dict_MTL, image_2_label_dict, min_labels=1):
    """Sorts out samples that have less than min_labels labels.
    
    :Arguments\::
        :collections_dict_MTL (*dictionary*)\::
            It's a dict containing the different image labels as keys and the
            according "collections_dict" as value.
            collections_dict_MTL[#label][class label][images]
        :image_2_label_dict (*dictionary*)\::
            A dictionary with the image (base) name as key and a list of avaiable
            #labels as value. It's needed for the estimation of the multitask
            loss.
            image_2_label_dict[full_image_name][#label_1, ..., #label_N]
        :min_labels (*int*)\::
            Integer that defines the minimum number of labels that one sample
            must have. If min_labels is smaller than 2, all samples are valid.
    :Returns\::
        :collections_dict_MTL (*dictionary*)\::
            It's a dict containing the different image labels as keys and the
            according "collections_dict" as value.
            collections_dict_MTL[#label][class label][images].
            The image paths are relative.
        :image_2_label_dict (*dictionary*)\::
            A dictionary with the image (base) name as key and a list of avaiable
            #labels as value. It's needed for the estimation of the multitask
            loss.
            image_2_label_dict[full_image_name][#label_1, ..., #label_N]
            The image paths are absolute.
    """
    if min_labels <= 1:
        return collections_dict_MTL, image_2_label_dict
    image_2_label_dict_filtered = {}
    image_list_filtered = []
    for image in image_2_label_dict.keys():
        num_labels = len(image_2_label_dict[image].keys())
        if num_labels >= min_labels:
            image_2_label_dict_filtered[image] = image_2_label_dict[image]
            image_list_filtered.append(os.path.basename(image))
     
    collections_dict_MTL_filtered = {}
    for im_label in collections_dict_MTL.keys():
        var_dict = {}
        for class_label in collections_dict_MTL[im_label]:
            for image in collections_dict_MTL[im_label][class_label]:
                if os.path.basename(image) in image_list_filtered:
                    try:
                        var_dict[class_label].append(image)
                    except:
                        var_dict[class_label] = [image]
        collections_dict_MTL_filtered[im_label] = var_dict
    
    return collections_dict_MTL_filtered, image_2_label_dict_filtered


def import_control_file_train(control_file_name):
    """Imports the information out of the control file.
    
    All relevant information for the training are contained in the control
    file. This information is fed into the according variable in this
    function.
    Pay attention that all paths in the control file do not contain
    empty spaces!
    
    :Arguments:
        :control_file_name (*string*)\::
            This variable is a string and contains the name of the control
            file. All relevant information for the training are in this file.
    
    :Returns:
        :master_file_name (*string*)\::
            This variable is a string and contains the name of the master file.
            The master file has to contain a list of the "collection.txt".
            All "collection.txt" have to be in the same folder as the master
            file.
            In the "collection.txt" are relative paths to the images and the
            according class label listed. The paths in a "collection.txt" has to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg".
            The "collection.txt" has to have a header "#image\t#Label" and the
            following lines "path_with_image\timage_label".
        :master_dir (*string*)\::
            This variable is a string and contains the absolute path to the
            master file.
        :tfhub_module (*string*)\::
            This variabel is a string and contains the Module URL to the
            desired networks feature vector. For ResNet-152 V2 is has to be
            'https://tfhub.dev/google/imagenet/resnet_v1_152/feature_vector/1'.
            Other posibilities for feature vectors can be found at
            'https://tfhub.dev/s?module-type=image-feature-vector'.
        :bottleneck_dir (*string*)\::
            This variable has to be a string and contains the path to the
            storage location for the feature vectors. The path has to be
            relative to the master_dir. Variable is deprecated and not used anymore.
        :train_batch_size (*int*)\::
            This variable is an int and says how many images shall be used for
            the classifier's training in one training iteration.
            The given number von samples will randomly picked out of all
            given images in the collection.txt. Thereby, the different
            classes will be represented by roughly an equal number of
            samplesin the training.
            Default value is 30.
        :how_many_training_steps (*int*)\::
            An int specifying the number of training iterations.
        :learning_rate (*float*)\::
            A float specifying the learning rate of the Gradient-Descent-
            Optimizer.
            Default value is 1e-4.
        :output_graph (*string*)\::
            A string specifying the absolute path including the name of the
            trained graph.
        :output_labels (*string*)\::
            This variable is a string specifying the absolute path including
            the name of the file that contains all class labels that can be
            predicted by the trained graph.
            Default value is 'output_labels.txt'
        :variables_to_learn (*list*)\::
            A list containing the collection.txt's header terms of the labels
            to be learnt. The terms must not contain blank spaces; use '_'
            instead of blank spaces!
            If more than one label is provided, multitask learning will be
            realized instead of learning the tasks independently of each other.
            If the tasks shall be learnt independently, multpile control files
            leading to multiple independent trainings have to be provided.
            The variable is a list containing all labels to be learnt.
            
            Example (string in control file): #timespan, #place
                
            Example (according list): [timespan, place]
            Default value is ['place', 'material', 'technique', 'depiction', 'timespan'] 
        :bool_MTL (*bool*)\::
            A boolean specifying if multiple task will be learnt together.
            Variable is deprecated and not used anymore.
        :min_samples_per_class (*int*)\::
            An integer that specifies the minimum number of samples per class
            that are desired. If the number of samples is less, the class
            won't be considered in learning and the samples are rejected from
            learning without data gaps. 
            Variable is deprecated and not used anymore.
        :logpath (*string*)\::
            The path where all summarized training information are stored.
            Default value is 'trained_model/'
        :bool_split_train_val (*bool*)\::
            Can be True or False. If True: The given data will be split into
            a training set and a validation set. You have to chosse a
            percentage for the size of thevalidation set
            (validation_percentage)!
            Default value is True.
        :validation_percentage (*int*):
            The amount of data that shall be used for validation instead of
            training. If the value is 20, 20% of the data will be used for
            validation and NOT for training.
            Default value is 25.
        :bool_CrossVal (*bool*)\::
            If True, Cross Validation (cv) will be applied. Thereby, all files
            listed in the master.txt are assumed to provide roughly and equal
            class distribution as well as a (nearly) simillar amount of images.
            In each of the Cross Validation Iteartions, one of those files will
            be used for testing and all others for training (or validation,
            respectively, if bool_split_train_val == True). In the end of each
            cv Iteartion, the trained classifier will be evaluated on the test
            set.
            Default value is False.
        :num_nodes_joint_fc (*int*)\::
            The number of nodes that each joint fully connected layer will have
            in Multi-task learning. Has to be an even number.
            Default value is 1.
        :nodes_prop_2_num_tasks (*bool*)\::
            If True, the num_nodes_joint_fc will be expanded if more than two
            tasks will be learnt. For each additional task, half of
            num_nodes_joint_fc nodes will be added to each joint
            fully-connected layer.
            Default value is False
        :num_finetune_layers (*int*)\::
            Number of layers of the module graph that will be fine-tuned.
            If this is 0, the module will be used as a sole feature extractor.
            If this is 1, the last layer will be fine-tuned, etc.
            Default value is 2.
        :num_task_stop_gradient (*int*)\::
            Samples, that have more than num_task_stop_gradient missing labels,
            will not contribute to the joined feature vector during training.
            If zero, only complete samples will contribute to the joint fc layer. 
            If negative, all samples will contribute to the joint fc layer.
            Default value is -1.
        :crop_aspect_ratio (*float*)\::
            Images will be cropped to their central square when their aspect ratio is
            smaller than crop_aspect_ratio. 
            For example, an image with a height of 200 and a width of 400 has an aspect ratio of 0.5.
            When crop_aspect_ratio is set to 0.9, the exemplary image will be cropped to the region
            hmin=0 hmax=200 wmin=100 wmax=300, as 0.5 < 0.9.
            Default value is 1, which results in no cropping.
        :min_num_labels (*int*)\::
            Integer that defines the minimum number of labels that one sample
            must have. If min_labels is zero or negative, all samples are valid.
            Default value is 0, which results in all samples being valid.
        :how_often_validation (*int*)\::
            Defines how often a validation is carried out. If this is 1, 
            a validation will be carried out after each training iteration, if it is 2, then
            after every second training iteration and so on.
            Default value is 10.
        :aug_set_dict (*dict*)\:: 
            Dictionary that defines what types of data augmentation will be carried out.
            Details on possible augmentations can be found in the documentation of the SILKNOW
            WP4 Library in the function add_data_augmentation.
            By default, horizontal flipping, vertical flipping and random rotation by 90°
            are carried out.
        :weight_decay (*float*)\:: 
            Scale factor for the L2-Loss.
            Default value is 1e-2.
        :evaluate_model (*bool*)\::
            If True, the classifier will be evaluated after training.
            Default value is True.
        :bool_MTL (*bool*)\::
            Used to indicate if multi task learning would be carried out.
            Variable is deprecated and not used anymore.
    """
    
    control_id = open(control_file_name, 'r',encoding='utf-8')
    
    """FILES AND DIRECTORIES"""
    output_labels = r"output_labels.txt"
    logpath = r"trained_model/"
    result_folder_name = r"results/"
    
    """SPECIFICATIONS FOR TRAINING"""
    train_batch_size = 30
    how_many_training_steps = 0 #so it does not need to be set in evaluation case
    learning_rate = 1e-4
    variables_to_learn = ['place', 'material', 'technique', 'depiction', 'timespan'] 
    min_samples_per_class = -1
    if how_many_training_steps == 0:
        bool_split_train_val = False
    else:
        bool_split_train_val = True
    validation_percentage = 25
    bool_CrossVal = False
    how_often_validation = 10
    weight_decay = 1e-2
    evaluate_model = True
    aug_set_dict = {"flip_left_right": True,
                    "flip_up_down": True,
                    "random_rotation90": True}
    
    """SPECIFICATIONS FOR ARCHITECTURE"""
    tfhub_module = "https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1"
    num_joint_fc_layer = 1
    num_nodes_joint_fc = 1500
    nodes_prop_2_num_tasks = False
    num_finetune_layers = 2
    num_task_stop_gradient = -1
    crop_aspect_ratio = 1
    min_num_labels = 0
    
    bottleneck_dir = -1
    output_graph = -1
    bool_MTL = True
    for variable in control_id:
        if variable.split(';')[0] == 'master_file_name':
            master_file_name = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'master_dir':
            master_dir = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'tfhub_module':
            tfhub_module = variable.split(';')[1].strip()
#        if variable.split(';')[0] == 'final_tensor_name':
#            final_tensor_name = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'bottleneck_dir':
            bottleneck_dir = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'train_batch_size' or variable.split(';')[0] == 'batch_size':
            train_batch_size = np.int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'how_many_training_steps':
            how_many_training_steps = np.int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'how_often_validation':
            how_often_validation = np.int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'learning_rate':
            learning_rate = np.float(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'output_graph':
            output_graph = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'output_labels':
            output_labels = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'variables_to_learn':
            variables_to_learn = variable.split(';')[1].replace(',', '')\
                        .replace(' ', '').replace('\n', '')\
                        .replace('\t', '').split('#')[1:]
            print('The following variables shall be considered during training:', variables_to_learn, '\n')
            bool_MTL = True
#            print('Multitask learning will be realized:', bool_MTL, '\n')
        if variable.split(';')[0] == 'min_samples_per_class':
            min_samples_per_class = np.int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'logpath':
            logpath = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'bool_split_train_val':
            bool_split_train_val = variable.split(';')[1].strip()
            if bool_split_train_val == 'True':
                bool_split_train_val = True
            else:
                bool_split_train_val = False
        if variable.split(';')[0] == 'validation_percentage':
            validation_percentage = int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'result_folder_name':
            result_folder_name = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'bool_CrossVal':
            bool_CrossVal = variable.split(';')[1].strip()
            if bool_CrossVal == 'True':
                bool_CrossVal = True
            else:
                bool_CrossVal = False
        if variable.split(';')[0] == 'num_joint_fc_layer':
            num_joint_fc_layer = int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'num_nodes_joint_fc':
            num_nodes_joint_fc = int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'nodes_prop_2_num_tasks':
            nodes_prop_2_num_tasks = variable.split(';')[1].strip()
            if nodes_prop_2_num_tasks == 'True':
                nodes_prop_2_num_tasks = True
            else:
                nodes_prop_2_num_tasks = False
        if variable.split(';')[0] == 'num_finetune_layers':
            num_finetune_layers = int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'num_task_stop_gradient':
            num_task_stop_gradient = int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'crop_aspect_ratio':
            crop_aspect_ratio = int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'min_num_labels':
            min_num_labels = int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'flip_left_right':
            aug_set_dict["flip_left_right"] = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'flip_up_down':
            aug_set_dict["flip_up_down"] = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'random_rotation90':
            aug_set_dict["random_rotation90"] = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'weight_decay':
            weight_decay = np.float(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'evaluate_model':
            evaluate_model = variable.split(';')[1].strip()
            if evaluate_model == 'False':
                evaluate_model = False
                print("No evaluation will be carried out!")
            else:
                evaluate_model = True
            
            
    control_id.close()
    
    return(master_file_name, master_dir, tfhub_module,
           bottleneck_dir, train_batch_size, how_many_training_steps, how_often_validation,
           learning_rate, output_graph, output_labels, variables_to_learn,
           bool_MTL, min_samples_per_class, logpath, bool_split_train_val,
           validation_percentage, result_folder_name, bool_CrossVal,
           num_joint_fc_layer, num_nodes_joint_fc, nodes_prop_2_num_tasks, 
           num_finetune_layers, num_task_stop_gradient,
           crop_aspect_ratio, min_num_labels, aug_set_dict, weight_decay, evaluate_model)
   
def evaluate_CNN_Classifier(control_file_name):
    """Evaluates a pre-trained CNN. This function does the same things like the
    training function if the control file has the same content. To only carry out
    the evaluation, the control file has to include specific values. 
    This function is just introduced to not confuse training and evaluation.
    
    :Arguments\::
        :control_file_name (*string*)\::
            This variable is a string and contains the name of the control
            file. All relevant information for the evaluation is in this file.
            The control file has to be stored in the same location as the
            script executing the training function.
    
    :Returns\::
        No returns. The trained graph (containing the tfhub_module and the
        trained classifier) is stored automatically in the directory given in
        the control file.
    """
    train_CNN_Classifier(control_file_name)
    
def train_CNN_Classifier(control_file_name):
    """Trains a classifier based on top of a pre-trained CNN.
    
    :Arguments\::
        :control_file_name (*string*)\::
            This variable is a string and contains the name of the control
            file. All relevant information for the training is in this file.
            The control file has to be stored in the same location as the
            script executing the training function.
    
    :Returns\::
        No returns. The trained graph (containing the tfhub_module and the
        trained classifier) is stored automatically in the directory given in
        the control file.
    """
    # Feed master file information into variables
    (master_file_name, master_dir, tfhub_module,
     bottleneck_dir, train_batch_size,
     how_many_training_steps, how_often_validation,
     learning_rate,
     output_graph, output_labels, labels_2_learn,
     bool_MTL, min_samples_per_class,
     logpath, bool_split_train_val,
     validation_percentage, result_folder_name,
     bool_CrossVal, num_joint_fc_layer,
     num_nodes_joint_fc, nodes_prop_2_num_tasks, 
     num_finetune_layers, num_task_stop_gradient,
     crop_aspect_ratio, min_num_labels, aug_set_dict, 
     weight_decay, evaluate_model) = import_control_file_train(control_file_name)
    final_tensor_name = 'final_result'
    
    # Get image_lists out of the Master.txt
    master_id = open(os.path.abspath(master_dir + '/' + master_file_name), 'r')
    collections_list = []
    for collection in master_id:
        collections_list.append(collection.strip())
    master_id.close()
    print('Got the following collections:', collections_list, '\n')
###############################################################################
                             # WITH Cross Validation #      
###############################################################################   
    if True:
        if bool_CrossVal:
            num_cv_iter = len(collections_list)
            print('Cross validation iterations:', num_cv_iter)
        else: 
            num_cv_iter = 1
        
        all_pred_testing = []
        all_gt_testing   = []
        perMTL_pred_testing = []
        perMTL_gt_testing   = []
        for cur_cv_iter in range (num_cv_iter):
            if bool_CrossVal:
                print('Current cv iteration:', cur_cv_iter)
            ##########################
            # Single task learning   #
            ##########################
            # WITH Cross Validation  #      
            ##########################
            
            if not bool_MTL:
                bool_MTL = True
               # This part of the code has been omitted.
               # The special case that only one task is considered can
               # be handled implicitly in the MTL-case.
                
                
            ##########################
            # Multi-task learning    #
            ########################## 
            # WITH Cross Validation  #      
            ##########################
            else:
                 # select each cv iteration another test set
                cur_collections_list_test  = [collections_list[cur_cv_iter]]
                cur_collections_list_train = []
                
                # all other collections are for training
                for coll_list in collections_list:
                    if coll_list not in cur_collections_list_test:
                        cur_collections_list_train.append(coll_list)
                collections_list_cv = cur_collections_list_train
                
                # convert the lists into an appropriate data structure
                if not len(collections_list_cv) == 0:
                    (collections_dict_MTL,
                     image_2_label_dict) = wp4lib.collections_list_MTL_to_image_lists(
                                                         collections_list_cv,
                                                         labels_2_learn,
                                                         min_samples_per_class,
                                                         master_dir,
                                                         bool_CrossVal)
                
                    (collections_dict_MTL,
                     image_2_label_dict) = sort_out_incomplete_samples(collections_dict_MTL, image_2_label_dict, min_num_labels)
                    
                    
                
                    with tf.gfile.FastGFile(output_labels, 'w') as f:
                        for task in collections_dict_MTL.keys():
                            f.write(task+';')
                            for c in collections_dict_MTL[task].keys():
                                f.write(' #'+c)
                            f.write('\n')
                    
                    print('\n\nTotal number of images provided for training:',
                          len(list(image_2_label_dict.keys())), '\n')
                    for im_label in collections_dict_MTL.keys():
                        print(im_label)
                        print(collections_dict_MTL[im_label].keys())
                        
                    # split the "training data" (not test) into training and
                    # validation data if a validation set is desired
                    if bool_split_train_val:
                        (collections_dict_MTL_train,
                         collections_dict_MTL_val,
                         image_2_label_dict_train,
                         image_2_label_dict_val
                         ) = split_collections_dict_MTL(collections_dict_MTL,
                                                        image_2_label_dict,
                                                        validation_percentage,
                                                        master_dir)
                        samplehandler = SampleHandler(len(image_2_label_dict_train.keys()), 
                                                      len(image_2_label_dict_val.keys()))
                    else:
                        samplehandler = SampleHandler(len(image_2_label_dict.keys()), 0)
                        
                        
                    # check if there is enough data provided
                    class_count_dict = {}
                    for im_label in collections_dict_MTL.keys():
                        temp_class_count = len(collections_dict_MTL[im_label].keys())
                        class_count_dict[im_label] = temp_class_count
                        if temp_class_count == 0:
                            tf.logging.error('No valid collections of images found at ' + master_file_name)
                            return -1
                        if temp_class_count == 1:
                            tf.logging.error('Only one class was provided via ' +
                                             master_file_name +
                                             ' - multiple classes are needed for classification.')
                            return -1
                        
                    
                    
                (collections_dict_MTL_test,
                 image_2_label_dict_test) = wp4lib.collections_list_MTL_to_image_lists(
                                                     cur_collections_list_test,
                                                     labels_2_learn,
                                                     -1,
                                                     master_dir,
                                                     bool_CrossVal)
                
                (collections_dict_MTL_test,
                 image_2_label_dict_test) = sort_out_incomplete_samples(collections_dict_MTL_test, image_2_label_dict_test, min_num_labels)
                
                if len(collections_list_cv) == 0:
                    class_count_dict = {}
                    for im_label in collections_dict_MTL_test.keys():
                        temp_class_count = len(collections_dict_MTL_test[im_label].keys())
                        class_count_dict[im_label] = temp_class_count
                    samplehandler = SampleHandler(len(image_2_label_dict_test.keys()), 0)
                #collections_dict_MTL[task][class label][images]
                #image_2_label_dict[full_image_name][#label_1, ..., #label_N]

                
                
                    
                    # TO DO: Intersection of image_2_label_dict.keys()
                    # -> train, val, test
                
                
                
                # Set up the pre-trained graph.
                # Sometimes in the following line a problem occurs. If so, please check,
                # wehre the tfhub_module has been saved and delete it. un the code
                # afterwards again.
                module_spec = hub.load_module_spec(str(tfhub_module)) 
                
                (train_step, 
                 cross_entropy,
                 ground_truth_input,
                 final_tensor, 
                 filename_input,
                 graph,
                 input_image_tensor,
                 trainable_variables) = create_computation_graph(module_spec            = module_spec, 
                                                             num_finetune_layers    = num_finetune_layers, 
                                                             class_count_dict       = class_count_dict, 
                                                             final_tensor_name      = final_tensor_name,
                                                             is_training            = True, 
                                                             learning_rate          = learning_rate,
                                                             num_joint_fc_layer     = num_joint_fc_layer, 
                                                             num_nodes_joint_fc     = num_nodes_joint_fc, 
                                                             nodes_prop_2_num_tasks = nodes_prop_2_num_tasks, 
                                                             num_task_stop_gradient = num_task_stop_gradient,
                                                             aug_set_dict           = aug_set_dict,
                                                             weight_decay           = weight_decay
                                         )
                
                with tf.Session(graph=graph) as sess:                                    # initialize the weights (pretrained/random) ---> graph contains loaded module
                    # Initialize all weights: for the module to their pretrained values,
                    # and for the newly added retraining layer to random initial values.
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    print("The following variables will be optimized during training:")
                    for var in trainable_variables:
                        print("\t",var.name)

                    
                    jpeg_data_tensor, decoded_image_tensor = wp4lib.add_jpeg_decoding(module_spec=module_spec,
                                                                      bool_hub_module=True,
                                                                      input_height=0,
                                                                      input_width=0,
                                                                      input_depth=0,
                                                                      bool_data_aug=True,
                                                                      aug_set_dict=aug_set_dict,
                                                                      crop_aspect_ratio=crop_aspect_ratio)

                    
                    # Create the operations we need to evaluate the accuracy of our new layer.
                    with tf.variable_scope("evaluationLayers") as scope:
                        (prediction_perMTL, overall_acc_perMTL,
                        prediction_all, overall_acc_all,
                        ground_truth_all,
                        ground_truth_perMTL,
                        prediction_perMTL_inc, 
                        ground_truth_perMTL_inc, 
                        correct_prediction_complete) = add_evaluation_step_MTL(
                                                                     final_tensor,
                                                                     ground_truth_input,
                                                                     class_count_dict)
                        
                    # Merge all the summaries and write them out to the logpath
                    merged = tf.summary.merge_all()
                    train_writer = tf.summary.FileWriter(
                            logpath + '/cv_'+str(cur_cv_iter) + '/train', sess.graph)
                    
                    # Create a train saver that is used to restore values into an eval graph
                    # when exporting models.
                    train_saver = tf.train.Saver()
                    
                    # after all bottlenecks are cached, the collections_dict_train will
                    # be renamed to collections_dict, if a validation is desired
                    if bool_split_train_val:
                        collections_dict_MTL = collections_dict_MTL_train
                        val_writer = tf.summary.FileWriter(
                                logpath + '/cv_'+str(cur_cv_iter) + '/val', sess.graph)
                        
                    best_validation = 0
                    for i in range(how_many_training_steps):
                        print('\ncurrent training Iteration in cvMTL:', i)

                        (image_data, 
                         train_ground_truth, 
                         train_filenames) = get_random_samples(collections_dict_MTL = collections_dict_MTL, 
                                                             how_many             = train_batch_size,
                                                             module_name          = tfhub_module,
                                                             master_dir           = master_dir, 
                                                             image_2_label_dict   = image_2_label_dict_train,
                                                             class_count_dict     = class_count_dict, 
                                                             samplehandler        = samplehandler, 
                                                             usage                = 'train',
                                                             session              = sess,
                                                             jpeg_data_tensor     = jpeg_data_tensor,
                                                             decoded_image_tensor = decoded_image_tensor)
                        feed_dictionary = {}
                        feed_dictionary[input_image_tensor] = image_data
                        feed_dictionary[filename_input] = train_filenames
                        for ind in range(np.shape(train_ground_truth)[0]):
                            feed_dictionary[ground_truth_input[ind]] = train_ground_truth[ind]                    
                        (train_summary,
                         cross_entropy_value, _) = sess.run(
                                            [merged, cross_entropy, train_step],
                                            feed_dict=feed_dictionary)
                        train_writer.add_summary(train_summary, i)
                        train_writer.flush()
                        
                        # If a validation is desired
                        if bool_split_train_val and (i%how_often_validation==0):
                            
                            # Validate on whole validation set
                            (val_image_data, 
                             val_ground_truth, 
                             val_filenames,_) = get_random_samples(collections_dict_MTL = collections_dict_MTL_val, 
                                                                 how_many             = -1, 
                                                                 module_name          = tfhub_module,
                                                                 master_dir           = master_dir, 
                                                                 image_2_label_dict   = image_2_label_dict_val,
                                                                 class_count_dict     = class_count_dict, 
                                                                 samplehandler        = samplehandler, 
                                                                 usage                ='valid',
                                                                 session              = sess,
                                                                 jpeg_data_tensor     = jpeg_data_tensor,
                                                                 decoded_image_tensor = decoded_image_tensor)
                            
                            OA_all = 0
                            OA_perMTL = {}
                            var_samples_total = {}
                            for val_iter in range(len(val_image_data)):
                        
                                feed_dictionary = {}
                                feed_dictionary[input_image_tensor] = val_image_data[train_batch_size*val_iter:train_batch_size*val_iter+train_batch_size]
                                feed_dictionary[filename_input]   = val_filenames[train_batch_size*val_iter:train_batch_size*val_iter+train_batch_size]
                                val_ground_truth_ = np.asarray(val_ground_truth)[:,train_batch_size*val_iter:train_batch_size*val_iter+train_batch_size]
                                if(len(feed_dictionary[input_image_tensor])) == 0: break
                                for ind in range(np.shape(val_ground_truth_)[0]):
                                    feed_dictionary[ground_truth_input[ind]] = val_ground_truth_[ind]
                                    
                                pred_perMTL, OA_perMTL_, _, OA_all_ = sess.run(
                                                [prediction_perMTL,
                                                 overall_acc_perMTL,
                                                 prediction_all, overall_acc_all],
                                                feed_dict=feed_dictionary)
                                
                                # kompliziertes Aufsummieren der OA, jeweils gesamt und per Task
                                var_dict_OA = {}
                                var_samples = {}
                                for MTL_iter, MTL_ele in enumerate(OA_perMTL_):
                                    var_samples[MTL_iter] = len(pred_perMTL[MTL_iter])
                                    
                                    if val_iter == 0:
                                        var_samples_total[MTL_iter] = var_samples[MTL_iter]
                                        OA_perMTL[MTL_iter] = OA_perMTL_[MTL_iter]*var_samples[MTL_iter]
                                    else:
                                        var_samples_total[MTL_iter] += var_samples[MTL_iter]
                                        OA_current_MTL         = OA_perMTL_[MTL_iter]
                                        samples_current_MTL    = var_samples[MTL_iter]
                                        OA_overall_current_MTL = OA_perMTL[MTL_iter]
                                        OA_perMTL[MTL_iter] = OA_current_MTL*samples_current_MTL + OA_overall_current_MTL
#                                OA_perMTL = tuple(var_dict_OA.values())
                                OA_all += OA_all_*sum(var_samples.values())
                                
                            OA_all /= sum(var_samples_total.values())
                            var_dict_OA = {}
                            for MTL_iter, MTL_ele in enumerate(OA_perMTL.keys()):
                                var_dict_OA[MTL_iter] = OA_perMTL[MTL_iter]/var_samples_total[MTL_iter]
                            OA_perMTL = tuple(var_dict_OA.values())

                            summary_val_OA = [tf.Summary.Value(tag='OA_all',
                                                               simple_value=OA_all)]
                            val_writer.add_summary(tf.Summary(value=summary_val_OA), i)
                            val_writer.flush()
                            for MTL_index, MTL_task in enumerate(class_count_dict):
                                summary_MTL = [tf.Summary.Value(tag='OA_'+MTL_task,
                                                               simple_value=OA_perMTL[MTL_index])]
                                val_writer.add_summary(tf.Summary(value=summary_MTL), i)
                                val_writer.flush()
       
                        if bool_split_train_val:
                            if best_validation < OA_all:
                                print("New best overall accuracy: %2.2f %%" % (OA_all*100))
                                train_saver.save(sess, logpath+CHECKPOINT_NAME)
                                best_validation = OA_all
                        else:
                            train_saver.save(sess, logpath+CHECKPOINT_NAME)
#                    
                    # Load latest checkpoint, i.e. the model that performed best
                    # on the validation set
#                    if how_many_training_steps == 0:
#                        test_saver = tf.train.import_meta_graph(logpath + CHECKPOINT_NAME + '.meta',
#                                               clear_devices=True)
#                        test_saver.restore(sess, logpath + CHECKPOINT_NAME)
#                        
#                    else:
#                        latest_checkpoint = tf.train.latest_checkpoint(logpath)
#                        train_saver.restore(sess, latest_checkpoint)
                    
                    
                    # Perform "empty" training step for variable initialization?!
                    if evaluate_model:
                        
                        if how_many_training_steps == 0:
                            print("No training will be carried out! Proceeding with Evaluation")
                            (image_data, 
                             train_ground_truth, 
                             train_filenames) = get_random_samples(collections_dict_MTL = collections_dict_MTL_test, 
                                                                 how_many             = 1,
                                                                 module_name          = tfhub_module,
                                                                 master_dir           = master_dir, 
                                                                 image_2_label_dict   = image_2_label_dict_test,
                                                                 class_count_dict     = class_count_dict, 
                                                                 samplehandler        = samplehandler, 
                                                                 usage                = 'train',
                                                                 session              = sess,
                                                                 jpeg_data_tensor     = jpeg_data_tensor,
                                                                 decoded_image_tensor = decoded_image_tensor)
                            feed_dictionary = {}
                            feed_dictionary[input_image_tensor] = image_data
                            feed_dictionary[filename_input] = train_filenames
                            for ind in range(np.shape(train_ground_truth)[0]):
                                feed_dictionary[ground_truth_input[ind]] = train_ground_truth[ind]                    
                            (train_summary,
                             cross_entropy_value) = sess.run(
                                                [merged, cross_entropy],
                                                feed_dict=feed_dictionary)
                        
                        latest_checkpoint = tf.train.latest_checkpoint(logpath)
                        train_saver.restore(sess, latest_checkpoint)
                        
                        # Get test samples
                        (test_image_data, 
                         test_ground_truth, 
                         test_filenames,
                         test_scale_factors) = get_random_samples(collections_dict_MTL = collections_dict_MTL_test, 
                                                             how_many             = -1, 
                                                             module_name          = tfhub_module,
                                                             master_dir           = master_dir, 
                                                             image_2_label_dict   = image_2_label_dict_test,
                                                             class_count_dict     = class_count_dict, 
                                                             samplehandler        = None, 
                                                             usage                = None,
                                                             session              = sess,
                                                             jpeg_data_tensor     = jpeg_data_tensor,
                                                             decoded_image_tensor = decoded_image_tensor)
                        # Test all data with batchsize
                        test_pred_perMTL, test_pred_all, test_gt_perMTL, test_gt_all = [], [], [], []
                        test_correct_pred_comp = []
                        for test_iter in range(len(test_image_data)):
                            
                            feed_dictionary = {}
                            feed_dictionary[input_image_tensor] = test_image_data[train_batch_size*test_iter:train_batch_size*test_iter+train_batch_size]
                            feed_dictionary[filename_input]   = test_filenames[train_batch_size*test_iter:train_batch_size*test_iter+train_batch_size]
                            test_ground_truth_ = np.asarray(test_ground_truth)[:,train_batch_size*test_iter:train_batch_size*test_iter+train_batch_size]
                            if(len(feed_dictionary[input_image_tensor])) == 0: break
                            for ind in range(np.shape(test_ground_truth_)[0]):
                                feed_dictionary[ground_truth_input[ind]] = test_ground_truth_[ind]
                                
                            (test_pred_perMTL_, test_pred_all_,
                             test_gt_perMTL_, test_gt_all_,
                             test_pred_perMTL_inc_, 
                             test_gt_perMTL_inc_,
                             test_correct_pred_comp_) = sess.run([prediction_perMTL,
                                                             prediction_all,
                                                             ground_truth_perMTL,
                                                             ground_truth_all,
                                                             prediction_perMTL_inc, 
                                                             ground_truth_perMTL_inc,
                                                             correct_prediction_complete],
                                                             feed_dict=feed_dictionary)
                            if test_iter == 0:
                                test_pred_perMTL = test_pred_perMTL_
                                test_gt_perMTL   = test_gt_perMTL_
                                test_pred_perMTL_inc = test_pred_perMTL_inc_
                                test_gt_perMTL_inc   = test_gt_perMTL_inc_
                            else:
                                var_dict_pred = {}
                                for MTL_iter, MTL_ele in enumerate(test_pred_perMTL):
                                    var_dict_pred[MTL_iter] = np.concatenate((MTL_ele,test_pred_perMTL_[MTL_iter]), axis=-1)
                                test_pred_perMTL = tuple(var_dict_pred.values())
                                
                                var_dict_gt = {}
                                for MTL_iter, MTL_ele in enumerate(test_gt_perMTL):
                                    var_dict_gt[MTL_iter] = np.concatenate((MTL_ele,test_gt_perMTL_[MTL_iter]), axis=-1)
                                test_gt_perMTL = tuple(var_dict_gt.values())
                                
                                var_dict_pred = {}
                                for MTL_iter, MTL_ele in enumerate(test_pred_perMTL_inc):
                                    var_dict_pred[MTL_iter] = np.concatenate((MTL_ele,test_pred_perMTL_inc_[MTL_iter]), axis=-1)
                                test_pred_perMTL_inc = tuple(var_dict_pred.values())
                                
                                var_dict_gt = {}
                                for MTL_iter, MTL_ele in enumerate(test_gt_perMTL_inc):
                                    var_dict_gt[MTL_iter] = np.concatenate((MTL_ele,test_gt_perMTL_inc_[MTL_iter]), axis=-1)
                                test_gt_perMTL_inc = tuple(var_dict_gt.values())
                            test_correct_pred_comp = np.concatenate((test_correct_pred_comp, test_correct_pred_comp_), axis=0)
                            
                            test_pred_all = np.concatenate((test_pred_all,
                                                               test_pred_all_), axis=0)
                            test_gt_all = np.concatenate((test_gt_all,
                                                             test_gt_all_), axis=0)
                        
                        # Save all predictions for all samples as well as the (possibly unkown) labels
                        if not os.path.exists(logpath+'cv_'+str(cur_cv_iter)+'/test/'):
                            os.makedirs(logpath+'cv_'+str(cur_cv_iter)+'/test/')
                        test_pred_perMTL_inc_asarray = []
                        test_gt_perMTL_inc_asarray   = []
                        for e1, e2 in zip(test_pred_perMTL_inc, test_gt_perMTL_inc):
                            test_pred_perMTL_inc_asarray.append(e1)
                            test_gt_perMTL_inc_asarray.append(e2)
                        np.save(logpath+'cv_'+str(cur_cv_iter)+'/test/prediction.npy', test_pred_perMTL_inc_asarray)
                        np.save(logpath+'cv_'+str(cur_cv_iter)+'/test/groundtruth.npy', test_gt_perMTL_inc_asarray)
                        
                        # Save plot for correlation between scale factor and prediction accuracy for complete samples
                        plt.figure(num=None, figsize=(13, 6), dpi=140, facecolor='w', edgecolor='k')
                        plt.plot(test_scale_factors, test_correct_pred_comp, 'r.')
                        plt.rcParams.update({'font.size': 20})
                        plt.xlabel('Side Ratio of original image')
                        plt.ylabel('Correctly predicted labels[%]')
                        plt.grid(axis='y', linewidth=2)
                        plt.savefig(logpath+'cv_'+str(cur_cv_iter)+'/test/sideratio2accuracy.png')
                        
                        # Save quality measure (per task) to disk
                        for task_index in range(np.shape(test_pred_perMTL)[0]):
                          # TO DO: Check assignment of index and name for MTL evaluation
                #          for temp_index, temp_name in enumerate(collections_dict[
                #                                            list(class_count)[task_index]].keys()):
                #              print(temp_index, temp_name)
                            wp4lib.estimate_quality_measures(test_gt_perMTL[task_index],
                                                    test_pred_perMTL[task_index],
                                                    list(collections_dict_MTL_test[
                                                            list(class_count_dict)[task_index]].keys()),
                                                    'Testing_cv_'+str(cur_cv_iter) + list(class_count_dict)[task_index],
                                                    result_folder_name,
                                                    how_many_training_steps, bool_MTL)
                        
                        all_pred_testing = np.concatenate((all_pred_testing,
                                                           test_pred_all), axis=0)
                        all_gt_testing = np.concatenate((all_gt_testing,
                                                         test_gt_all), axis=0)
                        
                        temp_array2tuple_pred = []
                        temp_array2tuple_gt   = []
                        if cur_cv_iter > 0:
                            for MTL_ind in range(np.shape(perMTL_pred_testing)[0]):
                                temp_pred_perMTL = np.concatenate(
                                        (perMTL_pred_testing[MTL_ind],
                                         test_pred_perMTL[MTL_ind]), axis=0)
                                temp_array2tuple_pred.append(temp_pred_perMTL)
                                
                                temp_gt_perMTL = np.concatenate(
                                        (perMTL_gt_testing[MTL_ind],
                                         test_gt_perMTL[MTL_ind]), axis=0)
                                temp_array2tuple_gt.append(temp_gt_perMTL)
                                
                            perMTL_pred_testing = tuple(temp_array2tuple_pred)
                            perMTL_gt_testing   = tuple(temp_array2tuple_gt)
                        else:
                            perMTL_pred_testing = test_pred_perMTL
                            perMTL_gt_testing   = test_gt_perMTL
                            
                        save_graph(sess, train_saver, logpath=logpath+'cv_'+str(cur_cv_iter)+'/'+CHECKPOINT_NAME)
#        
#        #############          
#                    # TO DO: Integriere collections_dict aus collections_dict_MTL
#                    with tf.gfile.FastGFile(output_labels, 'w') as f:
#                        f.write('\n'.join(collections_dict.keys()) + '\n')
#        ############# 
                
        # Estimate the quality measures for all cv iterations (summed up)        
        if not bool_MTL:
            # estimate_quality_measures!????!!!!!
            wp4lib.estimate_quality_measures(all_gt_testing, all_pred_testing,
                                      list(collections_dict.keys()),
                                      'Testing_AVERAGE_', result_folder_name,
                                      how_many_training_steps, bool_MTL) 
        else:
#            strings_allMTL = [str(int(i)) for i in list(set(all_gt_testing))]
            strings_allMTL = labels_2_learn
            wp4lib.estimate_quality_measures(all_gt_testing, all_pred_testing,
                                      strings_allMTL,
                                      'Testing_AVERAGE_', result_folder_name,
                                      how_many_training_steps, False)
            for MTL_ind, MTL_task in enumerate(class_count_dict.keys()):
                wp4lib.estimate_quality_measures(perMTL_gt_testing[MTL_ind],
                                          perMTL_pred_testing[MTL_ind],
                                          list(collections_dict_MTL_test[MTL_task].keys()),
                                          'Testing_'+'_AVERAGE_'+MTL_task, result_folder_name,
                                          how_many_training_steps, bool_MTL)
                

###############################################################################
                             # NO Cross Validation #      
###############################################################################        
                
    else:
        print('No Cross Validation')
        ########################
        # Single task learning #
        ########################
        # NO Cross Validation  #      
        ########################
        
        if not bool_MTL:
            bool_MTL = True
               # This part of the code has been omitted.
               # The special case that only one task is considered can
               # be handled implicitly in the MTL-case.
               
        #######################
        # Multi-task learning #
        ####################### 
        # NO Cross Validation #      
        #######################        
        else:
         # select each cv iteration another test set
            cur_collections_list_test  = [collections_list[random.randint(0,len(collections_list)-1)]]
            cur_collections_list_train = []
            
            # all other collections are for training
            for coll_list in collections_list:
                if coll_list not in cur_collections_list_test:
                    cur_collections_list_train.append(coll_list)
            collections_list_cv = cur_collections_list_train
            
            # convert the lists into an appropriate data structure
            (collections_dict_MTL,
             image_2_label_dict) = wp4lib.collections_list_MTL_to_image_lists(
                                                 collections_list_cv,
                                                 labels_2_learn,
                                                 min_samples_per_class,
                                                 master_dir,
                                                 bool_CrossVal)
            (collections_dict_MTL_test,
             image_2_label_dict_test) = wp4lib.collections_list_MTL_to_image_lists(
                                                 cur_collections_list_test,
                                                 labels_2_learn,
                                                 -1,
                                                 master_dir,
                                                 bool_CrossVal)
            (collections_dict_MTL,
             image_2_label_dict) = sort_out_incomplete_samples(collections_dict_MTL, image_2_label_dict, min_num_labels)
            
            (collections_dict_MTL_test,
             image_2_label_dict_test) = sort_out_incomplete_samples(collections_dict_MTL_test, image_2_label_dict_test, min_num_labels)
            
            print('\n\nTotal number of images provided for training:',
                  len(list(image_2_label_dict.keys())), '\n')
            if bool_split_train_val:
                (collections_dict_MTL_train,
                 collections_dict_MTL_val,
                 image_2_label_dict_train,
                 image_2_label_dict_val
                 ) = split_collections_dict_MTL(collections_dict_MTL,
                                                image_2_label_dict,
                                                validation_percentage,
                                                master_dir) 
                samplehandler = SampleHandler(len(image_2_label_dict_train.keys()), 
                                              len(image_2_label_dict_val.keys()))
            else:
                samplehandler = SampleHandler(len(image_2_label_dict.keys()), 0)
    #        1/0
            class_count_dict = {}
            for im_label in collections_dict_MTL.keys():
                temp_class_count = len(collections_dict_MTL[im_label].keys())
                class_count_dict[im_label] = temp_class_count
                if temp_class_count == 0:
                    tf.logging.error('No valid collections of images found at ' + master_file_name)
                    return -1
                if temp_class_count == 1:
                    tf.logging.error('Only one class was provided via ' +
                                     master_file_name +
                                     ' - multiple classes are needed for classification.')
                    return -1
    #        print(class_count_dict)
    #        1/0
            
            # Set up the pre-trained graph.
            # Sometimes in the following line a problem occurs. If so, please check,
            # wehre the tfhub_module has been saved and delete it. un the code
            # afterwards again.
            module_spec = hub.load_module_spec(str(tfhub_module))
            """
            graph, bottleneck_tensor, resized_image_tensor, wants_quantization = (
                  create_module_graph(module_spec))
          
            with graph.as_default():    
                (train_step, cross_entropy, bottleneck_input,
                 ground_truth_input, final_tensor,
                 filename_input) = add_final_retrain_ops_MTL(
                     class_count_dict, final_tensor_name, bottleneck_tensor,
                     wants_quantization, is_training=True,
                     learning_rate = learning_rate, num_joint_fc_layer=num_joint_fc_layer,
                     num_nodes_joint_fc=num_nodes_joint_fc,
                     nodes_prop_2_num_tasks=nodes_prop_2_num_tasks)
            """
            
            (train_step, 
             cross_entropy,
             ground_truth_input,
             final_tensor, 
             filename_input,
             graph,
             input_image_tensor,
             trainable_variables) = create_computation_graph(module_spec        = module_spec, 
                                                         num_finetune_layers    = num_finetune_layers, 
                                                         class_count_dict       = class_count_dict, 
                                                         final_tensor_name      = final_tensor_name,
                                                         is_training            = True, 
                                                         learning_rate          = learning_rate,
                                                         num_joint_fc_layer     = num_joint_fc_layer, 
                                                         num_nodes_joint_fc     = num_nodes_joint_fc, 
                                                         nodes_prop_2_num_tasks = nodes_prop_2_num_tasks, 
                                                         num_task_stop_gradient = num_task_stop_gradient,
                                                         aug_set_dict           = aug_set_dict
                                     )
            
            with tf.Session(graph=graph) as sess:                                    # initialize the weights (pretrained/random) ---> graph contains loaded module
                # Initialize all weights: for the module to their pretrained values,
                # and for the newly added retraining layer to random initial values.
                init = tf.global_variables_initializer()
                sess.run(init)
                print("The following variables will be optimized during training:")
                for var in trainable_variables:
                    print("\t",var.name)
            
                # Set up the image decoding sub-graph.
                jpeg_data_tensor, decoded_image_tensor = wp4lib.add_jpeg_decoding(module_spec=module_spec,
                                                                                      bool_hub_module=True,
                                                                                      input_height=0,
                                                                                      input_width=0,
                                                                                      input_depth=0,
                                                                                      bool_data_aug=False,
                                                                                      aug_set_dict=aug_set_dict,
                                                                                      crop_aspect_ratio=crop_aspect_ratio)  
               
                # We'll make sure we've calculated the 'bottleneck' image summaries and
                # cached them on disk.
                for MTL_task in collections_dict_MTL.keys():
                    collections_dict = collections_dict_MTL[MTL_task]
                
                # Create the operations we need to evaluate the accuracy of our new layer.
                (prediction_perMTL, overall_acc_perMTL,
                prediction_all, overall_acc_all,
                ground_truth_all,
                ground_truth_perMTL,
                prediction_perMTL_inc, 
                ground_truth_perMTL_inc, 
                correct_prediction_complete) = add_evaluation_step_MTL(
                                                             final_tensor,
                                                             ground_truth_input,
                                                             bool_MTL,
                                                             class_count_dict)
                # Merge all the summaries and write them out to the summaries_dir
                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(logpath + '/train', sess.graph)
                
                # Create a train saver that is used to restore values into an eval graph
                # when exporting models.
                train_saver = tf.train.Saver()
                
                if bool_split_train_val:
                    collections_dict_MTL = collections_dict_MTL_train
                    val_writer = tf.summary.FileWriter(logpath + '/val',
                                                         sess.graph)
                
                best_validation = 0
                for i in range(how_many_training_steps):
                    print('\ncurrent training Iteration:', i)
                    
                    (image_data, 
                     train_ground_truth, 
                     train_filenames) = get_random_samples(collections_dict_MTL = collections_dict_MTL, 
                                                         how_many             = train_batch_size,
                                                         module_name          = tfhub_module,
                                                         master_dir           = master_dir, 
                                                         image_2_label_dict   = image_2_label_dict_train,
                                                         class_count_dict     = class_count_dict, 
                                                         samplehandler        = samplehandler, 
                                                         usage                = 'train',
                                                         session              = sess,
                                                         jpeg_data_tensor     = jpeg_data_tensor,
                                                         decoded_image_tensor = decoded_image_tensor)
                    feed_dictionary = {}
                    feed_dictionary[input_image_tensor] = image_data
                    feed_dictionary[filename_input] = train_filenames
                    for ind in range(np.shape(train_ground_truth)[0]):
                        feed_dictionary[ground_truth_input[ind]] = train_ground_truth[ind]                    
                    (train_summary,
                     cross_entropy_value, _) = sess.run(
                                        [merged, cross_entropy, train_step],
                                        feed_dict=feed_dictionary)
                    train_writer.add_summary(train_summary, i)
                    train_writer.flush()
                    
                    # If a validation is desired
                    if bool_split_train_val:
                        
                        # Validate on whole validation set
                        (val_image_data, 
                         val_ground_truth, 
                         val_filenames,_) = get_random_samples(collections_dict_MTL = collections_dict_MTL_val, 
                                                             how_many             = -1, 
                                                             module_name          = tfhub_module,
                                                             master_dir           = master_dir, 
                                                             image_2_label_dict   = image_2_label_dict_val,
                                                             class_count_dict     = class_count_dict, 
                                                             samplehandler        = samplehandler, 
                                                             usage                ='valid',
                                                             session              = sess,
                                                             jpeg_data_tensor     = jpeg_data_tensor,
                                                             decoded_image_tensor = decoded_image_tensor)
                        OA_all = 0
                        OA_perMTL = {}
                        var_samples_total = {}
                        for val_iter in range(len(val_image_data)):
                            feed_dictionary = {}
                            feed_dictionary[input_image_tensor] = val_image_data[train_batch_size*val_iter:train_batch_size*val_iter+train_batch_size]
                            feed_dictionary[filename_input]   = val_filenames[train_batch_size*val_iter:train_batch_size*val_iter+train_batch_size]
                            val_ground_truth_ = np.asarray(val_ground_truth)[:,train_batch_size*val_iter:train_batch_size*val_iter+train_batch_size]
                            if(len(feed_dictionary[input_image_tensor])) == 0: break
                            for ind in range(np.shape(val_ground_truth_)[0]):
                                feed_dictionary[ground_truth_input[ind]] = val_ground_truth_[ind]
                                
                            pred_perMTL, OA_perMTL_, _, OA_all_ = sess.run(
                                            [prediction_perMTL,
                                             overall_acc_perMTL,
                                             prediction_all, overall_acc_all],
                                            feed_dict=feed_dictionary)
                            
                            # kompliziertes Aufsummieren der OA, jeweils gesamt und per Task
                            var_dict_OA = {}
                            var_samples = {}
                            for MTL_iter, MTL_ele in enumerate(OA_perMTL_):
                                var_samples[MTL_iter] = len(pred_perMTL[MTL_iter])
                                
                                if val_iter == 0:
                                    var_samples_total[MTL_iter] = var_samples[MTL_iter]
                                    OA_perMTL[MTL_iter] = OA_perMTL_[MTL_iter]*var_samples[MTL_iter]
                                else:
                                    var_samples_total[MTL_iter] += var_samples[MTL_iter]
                                    OA_current_MTL         = OA_perMTL_[MTL_iter]
                                    samples_current_MTL    = var_samples[MTL_iter]
                                    OA_overall_current_MTL = OA_perMTL[MTL_iter]
                                    OA_perMTL[MTL_iter] = OA_current_MTL*samples_current_MTL + OA_overall_current_MTL
#                                OA_perMTL = tuple(var_dict_OA.values())
                            OA_all += OA_all_*sum(var_samples.values())
                            
                        OA_all /= sum(var_samples_total.values())
                        var_dict_OA = {}
                        for MTL_iter, MTL_ele in enumerate(OA_perMTL.keys()):
                            var_dict_OA[MTL_iter] = OA_perMTL[MTL_iter]/var_samples_total[MTL_iter]
                        OA_perMTL = tuple(var_dict_OA.values())
                        
                        
                        summary_val_OA = [tf.Summary.Value(tag='OA_all',
                                                           simple_value=OA_all)]
                        val_writer.add_summary(tf.Summary(value=summary_val_OA), i)
                        val_writer.flush()
                        for MTL_index, MTL_task in enumerate(class_count_dict):
                            summary_MTL = [tf.Summary.Value(tag='OA_'+MTL_task,
                                                           simple_value=OA_perMTL[MTL_index])]
                            val_writer.add_summary(tf.Summary(value=summary_MTL), i)
                            val_writer.flush()
                            
                    if bool_split_train_val:
                        if best_validation < OA_all:
                            print("New best overall accuracy: %2.2f %%" % (OA_all*100))
                            train_saver.save(sess, logpath+CHECKPOINT_NAME)
                            best_validation = OA_all
                    else:
                        train_saver.save(sess, logpath+CHECKPOINT_NAME)
                        
                
                latest_checkpoint = tf.train.latest_checkpoint(logpath)
                train_saver.restore(sess, latest_checkpoint)
                (test_image_data, 
                 test_ground_truth, 
                 test_filenames,
                 test_scale_factors) = get_random_samples(collections_dict_MTL = collections_dict_MTL_test, 
                                                     how_many             = -1, 
                                                     module_name          = tfhub_module,
                                                     master_dir           = master_dir, 
                                                     image_2_label_dict   = image_2_label_dict_test,
                                                     class_count_dict     = class_count_dict, 
                                                     samplehandler        = None, 
                                                     usage                = None,
                                                     session              = sess,
                                                     jpeg_data_tensor     = jpeg_data_tensor,
                                                     decoded_image_tensor = decoded_image_tensor)
                # Test all data with batchsize
                test_pred_perMTL, test_pred_all, test_gt_perMTL, test_gt_all = [], [], [], []
                test_correct_pred_comp = []
                for test_iter in range(len(test_image_data)):
                    
                    feed_dictionary = {}
                    feed_dictionary[input_image_tensor] = test_image_data[train_batch_size*test_iter:train_batch_size*test_iter+train_batch_size]
                    feed_dictionary[filename_input]   = test_filenames[train_batch_size*test_iter:train_batch_size*test_iter+train_batch_size]
                    test_ground_truth_ = np.asarray(test_ground_truth)[:,train_batch_size*test_iter:train_batch_size*test_iter+train_batch_size]
                    if(len(feed_dictionary[input_image_tensor])) == 0: break
                    for ind in range(np.shape(test_ground_truth_)[0]):
                        feed_dictionary[ground_truth_input[ind]] = test_ground_truth_[ind]
                        
                    (test_pred_perMTL_, test_pred_all_,
                     test_gt_perMTL_, test_gt_all_,
                     test_pred_perMTL_inc_, 
                     test_gt_perMTL_inc_,
                     test_correct_pred_comp_) = sess.run([prediction_perMTL,
                                                     prediction_all,
                                                     ground_truth_perMTL,
                                                     ground_truth_all,
                                                     prediction_perMTL_inc, 
                                                     ground_truth_perMTL_inc,
                                                     correct_prediction_complete],
                                                     feed_dict=feed_dictionary)
                    if test_iter == 0:
                        test_pred_perMTL = test_pred_perMTL_
                        test_gt_perMTL   = test_gt_perMTL_
                        test_pred_perMTL_inc = test_pred_perMTL_inc_
                        test_gt_perMTL_inc   = test_gt_perMTL_inc_
                    else:
                        var_dict_pred = {}
                        for MTL_iter, MTL_ele in enumerate(test_pred_perMTL):
                            var_dict_pred[MTL_iter] = np.concatenate((MTL_ele,test_pred_perMTL_[MTL_iter]), axis=-1)
                        test_pred_perMTL = tuple(var_dict_pred.values())
                        
                        var_dict_gt = {}
                        for MTL_iter, MTL_ele in enumerate(test_gt_perMTL):
                            var_dict_gt[MTL_iter] = np.concatenate((MTL_ele,test_gt_perMTL_[MTL_iter]), axis=-1)
                        test_gt_perMTL = tuple(var_dict_gt.values())
                        
                        var_dict_pred = {}
                        for MTL_iter, MTL_ele in enumerate(test_pred_perMTL_inc):
                            var_dict_pred[MTL_iter] = np.concatenate((MTL_ele,test_pred_perMTL_inc_[MTL_iter]), axis=-1)
                        test_pred_perMTL_inc = tuple(var_dict_pred.values())
                        
                        var_dict_gt = {}
                        for MTL_iter, MTL_ele in enumerate(test_gt_perMTL_inc):
                            var_dict_gt[MTL_iter] = np.concatenate((MTL_ele,test_gt_perMTL_inc_[MTL_iter]), axis=-1)
                        test_gt_perMTL_inc = tuple(var_dict_gt.values())
                    test_correct_pred_comp = np.concatenate((test_correct_pred_comp, test_correct_pred_comp_), axis=0)
                    
                    test_pred_all = np.concatenate((test_pred_all,
                                                       test_pred_all_), axis=0)
                    test_gt_all = np.concatenate((test_gt_all,
                                                     test_gt_all_), axis=0)
                
                # Save all predictions for all samples as well as the (possibly unkown) labels
                if not os.path.exists(logpath+'/test/'):
                    os.makedirs(logpath+'/test/')
                test_pred_perMTL_inc_asarray = []
                test_gt_perMTL_inc_asarray   = []
                for e1, e2 in zip(test_pred_perMTL_inc, test_gt_perMTL_inc):
                    test_pred_perMTL_inc_asarray.append(e1)
                    test_gt_perMTL_inc_asarray.append(e2)
                np.save(logpath+'/test/prediction.npy', test_pred_perMTL_inc_asarray)
                np.save(logpath+'/test/groundtruth.npy', test_gt_perMTL_inc_asarray)
                
                # Save plot for correlation between scale factor and prediction accuracy for complete samples
                plt.figure(num=None, figsize=(13, 6), dpi=140, facecolor='w', edgecolor='k')
                plt.plot(test_scale_factors, test_correct_pred_comp, 'r.')
                plt.rcParams.update({'font.size': 20})
                plt.xlabel('Side Ratio of original image')
                plt.ylabel('Correctly predicted labels[%]')
                plt.grid(axis='y', linewidth=2)
                plt.savefig(logpath+'/test/sideratio2accuracy.png')
                
                # Save quality measure (per task) to disk
                for task_index in range(np.shape(test_pred_perMTL)[0]):
                  # TO DO: Check assignment of index and name for MTL evaluation
        #          for temp_index, temp_name in enumerate(collections_dict[
        #                                            list(class_count)[task_index]].keys()):
        #              print(temp_index, temp_name)
                    wp4lib.estimate_quality_measures(test_gt_perMTL[task_index],
                                            test_pred_perMTL[task_index],
                                            list(collections_dict_MTL_test[
                                                    list(class_count_dict)[task_index]].keys()),
                                            'Testing' + list(class_count_dict)[task_index],
                                            result_folder_name,
                                            how_many_training_steps, bool_MTL)
                        
                save_graph(sess, train_saver, logpath=logpath+'/'+CHECKPOINT_NAME)
    #############          
#                # TO DO: Integriere collections_dict aus collections_dict_MTL
#                with tf.gfile.FastGFile(output_labels, 'w') as f:
#                    f.write('\n'.join(collections_dict.keys()) + '\n')
    #############    
                
                with tf.gfile.FastGFile(output_labels, 'w') as f:
                    for task in collections_dict_MTL.keys():
                        f.write(task+';')
                        for c in collections_dict_MTL[task].keys():
                            f.write(' #'+c)
                        f.write('\n')



def read_tensor_from_image_file(file_name,
                                input_height,
                                input_width):
    """Loads and prepares the images for the graph.
    
    :Arguments:
        :file_name:
            The name of the file containing the image data.
        :input_height:
            An integer specifying the expeted image height of the graph's
            input layer.
        :input_width:
            An integer specifying the expeted image width of the graph's
            input layer.
            
    :Returns:
        :image_tensor:
            A tensor containing the image data.
    """
    file_reader = tf.read_file(file_name, "file_reader")
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                           name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name="jpeg_reader")
    float_caster  = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized       = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    input_mean    = 0
    input_std     = 255
    normalized    = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess          = tf.Session()
    image_tensor  = sess.run(normalized)

    return image_tensor


def image_lists_to_image_array(image_lists, master_dir):
    """Converts an array of image lists into an array of imagess.
    
    :Arguments:
        :image_lists:
            It's an array containing all file names of the "image_file.txt" (the
            master file in array form). Each "image_file.txt" contains a list
            of the images to be classified.
    
    :Returns:
        :image_array:
            It's an array containing all images with their relative path from
            the master directory to their storage location.
    """
    image_array = []
    for image_list in image_lists:
        im_id = open(os.path.join(master_dir, image_list), 'r')
        first_line_passed = False
        for im_line in im_id:
            if first_line_passed == False:
#                print(im_line)
                first_line_passed = True
                continue
            rel_im_path = im_line.replace('\n', '')
            image_array.append(rel_im_path)
#        1/0
    return image_array
    

# Code ausgeliehen von: https://bretahajek.com/2017/04/importing-multiple-tensorflow-models-graphs/
class ImportGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc, output_name, task_list):
        """Creates an object of class ImportGraph
        
        :Arguments:
          :loc:
              The absolute path to the storage location of the trained graph
            including the name of the graph.
          :output_name:
            The name of the output classification layer in the retrained graph.
            It has to be the same name as it was given in the training.
        :task_list:
            Names of the tasks to be considered for the classification. The
            wanted tasks have to be contained in the label_file, i.e. they must
            have been considered during training, too. Task names should begin
            with a # and be separated by commas, e.g. '#timespan, #place'
    
        :Returns:
            Object of class ImportGraph
        """
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + CHECKPOINT_NAME + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, loc + CHECKPOINT_NAME)
            self.task_list = task_list
            self.output_operations = [self.graph.get_operation_by_name('customLayers/'+output_name+'_'+task).outputs[0] for task in task_list]
            self.eval_operations   = [self.graph.get_operation_by_name]

    def run(self, data):
        """ Running the activation operation previously imported.
        
        :Arguments:
            :data:
                The image data, i.e. the output from read_tensor_from_image_file.
                
        :Returns:
            :output:
                The result of the specified layer (output_name).
        """
        # The 'x' corresponds to name of input placeholder
        feed_dict={"moduleLayers/input_img:0": data}
        for task in self.task_list:
            feed_dict['customLayers/input/GroundTruthInput'+task+":0"] = [0.]
        output = self.sess.run(self.output_operations, feed_dict=feed_dict)
        return output
    
#    def evaluate(self, data):
#        """ Running the activation operation previously imported.
#        
#        :Arguments:
#            :data:
#                The image data, i.e. the output from read_tensor_from_image_file.
#                
#        :Returns:
#            :output:
#                The result of the specified layer (output_name).
#        """
#        
#        (test_pred_perMTL_, test_pred_all_,
#         test_gt_perMTL_, test_gt_all_,
#         test_pred_perMTL_inc_, 
#         test_gt_perMTL_inc_,
#         test_correct_pred_comp_) = sess.run([prediction_perMTL,
#                                         prediction_all,
#                                         ground_truth_perMTL,
#                                         ground_truth_all,
#                                         prediction_perMTL_inc, 
#                                         ground_truth_perMTL_inc,
#                                         correct_prediction_complete],
#                                         feed_dict=feed_dictionary)

def apply_CNN_Classifier(control_file_name):
    """Applies the trained classifier to new data.
    
    
    :Arguments:
        :control_file_name:
            This variable is a string and contains the name of the control
            file. All relevant information for applying the trained classifier
            is in this file.
            The control file has to be stored in the same location as the
            script executing the classification function.
        
    :Returns:
        No returns. The classification result will be written automatically
        into a result file in the master direction. The name of this file can
        be chosen by the user in the control file.
        
    """
    (master_file_name, master_dir, model_file, label_file, tfhub_module,
     classification_result, task_list
     )            = import_control_file_apply(control_file_name)
    module_spec   = hub.load_module_spec(str(tfhub_module))
    height, width = hub.get_expected_image_size(module_spec)
    output_layer = 'final_result'
    
    # Get image_lists out of the Master.txt
    master_id = open(os.path.abspath(master_dir + '/' + master_file_name), 'r')
    image_lists = []
    for image_list in master_id:
        image_lists.append(image_list)
    master_id.close()
    print('Got the following image lists for classification:', image_lists)
    (image_file_array) = image_lists_to_image_array(image_lists, master_dir)
    
    model = ImportGraph(model_file, output_layer, task_list)
    
    class_res_id = open(os.path.abspath(master_dir + '/'
                                        + classification_result), 'w')
    class_res_id.write('#image_file')
    for task in task_list:
        class_res_id.write('\t#'+task)
    class_res_id.write('\n')                      
    
    for image_file in image_file_array:
        print('current image:\n', image_file)       
        
        im_file_full_path = os.path.abspath(os.path.join(master_dir,
                                                         image_file))
        image_tensor      = read_tensor_from_image_file(im_file_full_path,
                                                        height,
                                                        width)
        
        results = model.run(image_tensor)
        
        
        # Get most probable result per task
        resultTasks = {}
        for i, task in enumerate(task_list):
            resultTasks[task] = np.argmax(results[i])
        # Read actual class names and map results to those names
        resultClasses = {}
        control_id = open(label_file, 'r',encoding='utf-8')
        for variable in control_id:
            for task in task_list:
                if variable.split(';')[0] == task:
                    class_list = variable.split(';')[1].replace(',', '')\
                                .replace(' ', '').replace('\n', '')\
                                .replace('\t', '').split('#')[1:]
                    resultClasses[task] = class_list[resultTasks[task]]
        
        print(resultClasses)
        # Write class names to file
        class_res_id.write("%s" % (image_file))
        for task in task_list:
            class_res_id.write("\t%s" % (resultClasses[task]))
        class_res_id.write("\n")
            
    class_res_id.close()

    
def import_control_file_apply(control_file_name):
    """Imports the information out of the control file.
    
    All relevant information for applying the classifier are contained in the
    control file. This information is fed into the according variables in this
    function.
    Pay attention that all paths in the control file do not contain
    empty spaces!
    
    :Arguments:
        :control_file_name:
            This variable is a string and contains the name of the control
            file. All relevant information for the training are in this file.
    
    :Returns:
        :model_file:
            The absolute path to the storage location of the trained graph
            including the name of the graph.
            For example: some_absolute_path/output_graph.pb
        :label_file:
            The absolute path to the storage location of the text file with the
            label names including the name of the graph.
            For example: some_absolute_path/output_lables.txt
        :master_file_name:
            This variable is a string and contains the name of the master file.
            The master file has to contain a list of the "image_file.txt".
            All "image_file.txt" have to be in the same folder as the master
            file.
            In the "image_file.txt" are relative paths to the images listed,
            which shall be classified. The paths in a "image_file.txt" has to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg".
            The first line in a "image_file.txt" is a header (for example:
            "#image") and the following lines contain "path_with_image".
        :master_dir (*string*)\::
            This variable is a string and contains the absolute path to the
            master file.
        :tfhub_module:
            This variabel is a string and contains the Module URL to the
            desired networks feature vector. For ResNet-152 V2 is has to be
            'https://tfhub.dev/google/imagenet/resnet_v1_152/feature_vector/1'.
            Other posibilities for feature vectors can be found at
            'https://tfhub.dev/s?module-type=image-feature-vector'.
#        :final_tensor_name:
#            The name of the output classification layer in the retrained graph.
#            It has to be the same name as it was given in the training.
        :classification_result:
            The name of the file, which shall contain the classification
            results. It will have a header line and all subsequent lines
            contain the images (with their according relative path from the
            master file) as well as the predicted class label. It will be
            stored in the "master_dir".
        :task_list:
            Names of the tasks to be considered for the classification. The
            wanted tasks have to be contained in the label_file, i.e. they must
            have been considered during training, too. Task names should begin
            with a # and be separated by commas, e.g. '#timespan, #place'
    """
    
    # Default Values
    tfhub_module = "https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1"
    task_list    = ['place', 'material', 'technique', 'depiction', 'timespan'] 
    
    control_id = open(control_file_name, 'r',encoding='utf-8')
    for variable in control_id:
        if variable.split(';')[0] == 'master_file_name':
            master_file_name = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'master_dir':
            master_dir = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'model_file':
            model_file = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'label_file':
            label_file = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'tfhub_module':
            tfhub_module = variable.split(';')[1].strip()
#        if variable.split(';')[0] == 'final_tensor_name':
#            final_tensor_name = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'classification_result':
            classification_result = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'task_list':
            task_list = variable.split(';')[1].replace(',', '')\
                        .replace(' ', '').replace('\n', '')\
                        .replace('\t', '').split('#')[1:]
            print('The following labels shall be classified:', task_list, '\n')
            
    control_id.close()

    return(master_file_name, master_dir, model_file, label_file, 
           tfhub_module, classification_result, task_list)
    
    
    
def main(application, controlFile):
    """Example for calling the CNN classifier functions.
    
    All relevant information for the functions "train_CNN_Classifier()" and
    "apply_CNN_Classifier()" has to be given in the according control files.
    This means that at least the paths in the control files have to be adapted
    by the user.
    
    If you run the train function for the first time on your data, it will take
    some time because the images' features have to be estimated and stored. The
    same yields for the apply function; don't worry if it takes some time.
    
    """
    # Zwei Bilder in unterschiedlichen collections.txt dürfen aktuell nicht denselben Dateinamen haben!!!
    if application == 'classification':
        print("Commencing with application of trained model.")
        apply_CNN_Classifier(controlFile)
    elif application == 'evaluation':
        print("Commencing with evaluation of trained model.")
        evaluate_CNN_Classifier(controlFile)
    elif application == 'training':
        print("Commencing with training of new model.")
        train_CNN_Classifier(controlFile)
    else:
        print(application, controlFile)
        print("Incorrect use of parameters! \n",
              "The first parameter must be 'classification', 'evaluation' or 'training'. \n",
              "The second parameter must be the relative path to a control file.")
        
    
#    control_file_name_apply = 'control_file_applyClassifier.txt'
#    apply_CNN_Classifier(control_file_name_apply)
    
    
  
if __name__ == '__main__':
    application = sys.argv[1]
    controlFile = sys.argv[2]
    main(application, controlFile)
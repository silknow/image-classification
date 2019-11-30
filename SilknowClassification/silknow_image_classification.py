# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 18:37:39 2019

@author: clermont
"""

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
#import re
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd


try:
    import SILKNOW_WP4_library as wp4lib
except:
    try:
        sys.path.insert(0,'./src') 
        import SILKNOW_WP4_library as wp4lib
    except:
        print("SILKNOW WP4 Library could not be imported!")

try:
    from image_classification_utility import *
except:
    sys.path.insert(0,'./src') 
    from image_classification_utility import *
    
try:
    from create_dataset_utility import *
except:
    sys.path.insert(0,'./src') 
    from create_dataset_utility import *

#import urllib.parse
#import argparse

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

# The location where variable checkpoints will be stored.
CHECKPOINT_NAME = 'model.ckpt'

# A module is understood as instrumented for quantization with TF-Lite
# if it contains any of these ops.
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')


def CreateDataset(configfile):
    """Creates the dataset for the CNN.

    :Arguments\::
        :configfile (*string*)\::
            This variable is a string and contains the name of the configfile. 
            All relevant information for the dataset creation are in this file.
            The configfile has to be stored in the same location as the
            script executing the classification function.
        
    :Returns\::
        No returns. The classification result will be written automatically
        into result file in the master direction. The name of this file can
        be chosen by the user in the control file.
    
    
    """
    ImageCSV, FieldCSV_list, Mapping_table, MasterfilePath, \
    csvPath, minNumSamples, skipDownload, onlyFromCollection = importDatasetConfiguration(configfile)
    
    imagePath = MasterfilePath+r"/img/"
    
    # make sure all paths exist
    if not os.path.isdir(MasterfilePath):
        os.mkdir(MasterfilePath)
    if not os.path.isdir(imagePath):
        os.mkdir(imagePath)
            
        
    
    # Load ImageCSV 
    image_data = pd.read_csv(csvPath+ImageCSV)
        
    # filter out all URLs that link to silknow.org
    image_data = image_data[~image_data.url.str.contains("silknow.org")]
    
    # OPTIONAL: Only use records from one collection
    if not onlyFromCollection == "":
        onlyFromCollection = onlyFromCollection.lower()
        print("Only records from collection/graph", onlyFromCollection, "will be considered!")
        image_data = image_data[image_data.g == "http://data.silknow.org/"+onlyFromCollection]
    
    # set identifier
    image_index = make_id(image_data)
    image_data["identifier"] = image_index
    image_data = (image_data.groupby('identifier')["url"].apply(lambda x: list(set(x))).reset_index()).set_index("identifier")
    
    # Load field data, map to class structure, merge with other dataframes
    for field in FieldCSV_list:
        print("Loading and Mapping data from",field[0])
        data = LoadAndMap(field, Mapping_table, csvPath)
        image_data = pd.merge(image_data, data, left_index=True, right_index=True)
    
    # save merged dataframe, might contain dead links
#    image_data.to_csv(csvPath+"dataset_cleaned_incdeadlinks.csv")
    
    if not skipDownload:
        downloadNewImages(image_data, imagePath)
    
    # Delete records from dataframe for which no image was downloaded
    onlyfiles = [f[:-4] for f in listdir(imagePath) if isfile(join(imagePath, f))]
    onlyfiles = list(dict.fromkeys(onlyfiles))
    image_data_ = pd.DataFrame(columns=list(image_data))
    for f in onlyfiles:
        if f in image_data.index:
            image_data_ = image_data_.append(image_data.loc[f])
    image_data_.index.name='identifier'
    image_data = image_data_
    
    # Check if there are enough samples for each class
    for field in FieldCSV_list:
        print("Checking number of samples for", field[0])
        fieldname = field[1]
        image_data = checkNumSamples(image_data, fieldname, minNumSamples)
    
    # Sort out records with only NaN
    image_data = image_data.dropna(thresh=2)
        
    
#    # Save dataset as .csv
#    if not onlyFromCollection == "":
#        image_data.to_csv(csvPath+"dataset_cleaned_"+onlyFromCollection+".csv")
#    else:
#        image_data.to_csv(csvPath+"dataset_cleaned.csv")
    
    
    # Save dataset in five collection files, to be usable in MTL 
    print("Saving dataframe into collections files.")
#    image_data["identifier"] = image_data.index
    dataChunkList = np.array_split(image_data, 5)
    variable_list = np.asarray(FieldCSV_list)[:,1]
    
    for i, chunk in enumerate(dataChunkList):
        collection = open(MasterfilePath+"collection_"+str(i+1)+".txt","w+")
#        string = ["#"+name+"\t" for name in list(image_data)[1:]]
        string = ["#"+name+"\t" for name in variable_list]
        collection.writelines(['#image_file\t']+string+["\n"])
        
        for index, row in chunk.iterrows():
            imagefile = str(row.name)+".jpg\t"
            
            # Skip improperly formatted filenames
            if "/" in imagefile: continue
    
#            string = [(str(row[label])+"\t").replace('nan','NaN') for label in list(image_data)[1:]]
            string = [(str(row[label])+"\t").replace('nan','NaN') for label in variable_list]
    
            collection.writelines(["./img/"+imagefile]+string+["\n"])
    
        collection.close()
    
    # Write collection files to masterfile and save it in the same path
    master = open(MasterfilePath+"Masterfile.txt","w+")
    for i in range(len(dataChunkList)):
        master.writelines(["collection_"]+[str(i+1)]+[".txt\n"])
    master.close()
    
    # Print label statistics
    for field in FieldCSV_list:
        fieldname = field[1] 
        print("Classes for variable", fieldname,":")
        print(image_data[fieldname].value_counts(dropna=False))
    
    # Create default CNN Configuration File
    image_data = image_data[np.asarray(FieldCSV_list)[:,1]]
    createCNNConfigFile(image_data, MasterfilePath)
    
    # If Masterfiles for Evaluation and Classification do net exist yet, create them
    if not os.path.isfile(MasterfilePath+"Masterfile_Classification.txt"):
        collection = open(MasterfilePath+"Masterfile_Classification.txt","w+")
        collection.write("Collection_Classification.txt")
        collection.close()
    if not os.path.isfile(MasterfilePath+"Collection_Classification.txt"):
        collection = open(MasterfilePath+"Collection_Classification.txt","w+")        
        collection.write("./img/imatex__0011654a-8ae5-30fa-bf48-ada9e2996e8c.jpg\n")
        collection.write("http://imatex.cdmt.cat/fotografies/0000000680.jpg\n")
        collection.close()
    if not os.path.isfile(MasterfilePath+"Masterfile_Evaluation.txt"):
        collection = open(MasterfilePath+"Masterfile_Evaluation.txt","w+")
        collection.write("collection_1.txt")
        collection.close()


def crossvalidate_CNN_Classifier(configfile):
    """Carries out training of the CNN with cross validation.

    :Arguments\::
          :configfile (*string*)\::
            This variable is a string and contains the name of the configfile. 
            All relevant information for the dataset creation are in this file.
            The configfile has to be stored in the same location as the
            script executing the classification function.
    
    :Returns\::
        No returns. The results of the evaluation are stored automatically in 
        the directory given in the configfile.
    """
    (master_file_name, master_dir, tfhub_module,
     bottleneck_dir, train_batch_size,
     how_many_training_steps, how_often_validation,
     learning_rate,
     output_graph, labels_2_learn,
     bool_MTL, min_samples_per_class,
     logpath,
     validation_percentage, result_folder_name,
     evaluation_index, num_joint_fc_layer,
     num_nodes_joint_fc, nodes_prop_2_num_tasks, 
     num_finetune_layers, num_task_stop_gradient,
     crop_aspect_ratio, min_num_labels, aug_set_dict, 
     weight_decay, evaluate_model) = import_control_file_train(configfile)
    
    # Get image_lists out of the Master.txt
    master_id = open(os.path.abspath(master_dir + '/' + master_file_name), 'r')
    collections_list = []
    for collection in master_id:
        collections_list.append(collection.strip())
    master_id.close()
    print('Got the following collections for cross validation:', collections_list, '\n')
    
    num_cv_iter = len(collections_list)
    print('Cross validation iterations:', num_cv_iter)
    
    bonus_parameters = {"callFromCrossVal": True}
    
    perMTL_pred_testing = []
    perMTL_gt_testing   = []
    
    # Call Training Function in every CV iteration, save results and concatenate them
    for cv_iter in range(num_cv_iter):
        bonus_parameters["evaluation_index"] = cv_iter+1
        test_pred_perMTL, test_gt_perMTL, \
        class_count_dict, collections_dict_MTL_test = train_CNN_Classifier(configfile, bonus_parameters)
        
        
        # Concatenate results from CV iterations
        temp_array2tuple_pred = []
        temp_array2tuple_gt   = []
        if cv_iter > 1:
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
            
    for MTL_ind, MTL_task in enumerate(class_count_dict.keys()):
        wp4lib.estimate_quality_measures(perMTL_gt_testing[MTL_ind],
                                  perMTL_pred_testing[MTL_ind],
                                  list(collections_dict_MTL_test[MTL_task].keys()),
                                  'Testing_'+'_AVERAGE_'+MTL_task, result_folder_name,
                                  how_many_training_steps, bool_MTL)
            
def evaluate_CNN_Classifier(configfile):
    """Evaluates a pre-trained CNN.
    
    :Arguments\::
        :configfile (*string*)\::
            This variable is a string and contains the name of the configfile.
            All relevant information for the evaluation is in this file.
            The configfile has to be stored in the same location as the
            script executing the evaluation function.
    
    :Returns\::
        No returns. The results of the evaluation are stored automatically in 
        the directory given in the configfile.
    """
    evaluation_parameters = {"how_many_training_steps": 0,
                             "bool_split_train_val": False,
                             "bool_CrossVal": False,
                             "evaluate_model": True,
                             "callFromEvaluate": True,
                             "evaluation_index": 1}
    
    train_CNN_Classifier(configfile, evaluation_parameters)
    
def train_CNN_Classifier(configfile, bonus_parameters = {}):
    """Trains a classifier based on top of a pre-trained CNN.
    
    :Arguments\::
        :configfile (*string*)\::
            This variable is a string and contains the name of the configfile.
            All relevant information for the training is in this file.
            The configfile has to be stored in the same location as the
            script executing the training function.
        :bonus_parameters (*dict*)\::
            Dictionary with parameters for internal handling of evaluation
            and cross validation. Must not be defined manually.
    
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
     output_graph, labels_2_learn,
     bool_MTL, min_samples_per_class,
     logpath,
     validation_percentage, result_folder_name,
     evaluation_index, num_joint_fc_layer,
     num_nodes_joint_fc, nodes_prop_2_num_tasks, 
     num_finetune_layers, num_task_stop_gradient,
     crop_aspect_ratio, min_num_labels, aug_set_dict, 
     weight_decay, evaluate_model) = import_control_file_train(configfile, bonus_parameters)
    final_tensor_name = 'final_result'
    
    
    # Set parameters from bonus_parameters for call from evaluation function
    if "how_many_training_steps" in bonus_parameters.keys(): how_many_training_steps = bonus_parameters["how_many_training_steps"]
    if "bool_split_train_val" in bonus_parameters.keys(): bool_split_train_val = bonus_parameters["bool_split_train_val"]
    if "evaluate_model" in bonus_parameters.keys(): evaluate_model = bonus_parameters["evaluate_model"]
    if "callFromEvaluate" in bonus_parameters.keys(): evaluation_index = bonus_parameters["evaluation_index"]
    
    # Set parameters from bonus_parameters for call from crossvalidation function
    if "callFromCrossVal" in bonus_parameters.keys(): 
        evaluation_index = bonus_parameters["evaluation_index"]
        print('Current cv iteration:', evaluation_index)
        bool_CrossVal = True
    else:
        bool_CrossVal = False
        
        
        
    # Handle CV index, if CV is carried out
    cur_cv_iter = evaluation_index
        
        
        
    # Set deprecated parameters
    bool_MTL = True
    if validation_percentage > 0 and not how_many_training_steps == 0:
        bool_split_train_val = True
    else:
        bool_split_train_val = False
        
        
    # Get image_lists out of the Master.txt
    master_id = open(os.path.abspath(master_dir + '/' + master_file_name), 'r')
    collections_list = []
    for collection in master_id:
        collections_list.append(collection.strip())
    master_id.close()
    print('Got the following collections:', collections_list, '\n')


    all_pred_testing = []
    all_gt_testing   = []
    perMTL_pred_testing = []
    perMTL_gt_testing   = []
    
    
    
    # Select Training and Evaluation Data
    if cur_cv_iter == -1:
        # select all data for training
        print("No evaluation will be carried out. All samples will be used for training.")
        cur_collections_list_train = [collections_list]
        cur_collections_list_test = []
        evaluate_model = False
        bool_CrossVal = False
        
    else:
         # select each cv iteration another test set
        cur_collections_list_test  = [collections_list[cur_cv_iter-1]]
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
            
        
    
    if evaluate_model:
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
        if how_many_training_steps > 0:
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
        
        
        if bool_CrossVal:
            logpath_ = logpath+'cv_'+str(cur_cv_iter)
        else:
            logpath_ = logpath
        
        # Merge all the summaries and write them out to the logpath
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logpath_ + '/train', sess.graph)
        
        # Create a train saver that is used to restore values into an eval graph
        # when exporting models.
        train_saver = tf.train.Saver()
        
        # after all bottlenecks are cached, the collections_dict_train will
        # be renamed to collections_dict, if a validation is desired
        if bool_split_train_val:
            collections_dict_MTL = collections_dict_MTL_train
            val_writer = tf.summary.FileWriter(logpath_+ '/val', sess.graph)
            
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
                
            if not os.path.exists(logpath_+'/test/'):
                os.makedirs(logpath_+'/test/')
            test_pred_perMTL_inc_asarray = []
            test_gt_perMTL_inc_asarray   = []
            for e1, e2 in zip(test_pred_perMTL_inc, test_gt_perMTL_inc):
                test_pred_perMTL_inc_asarray.append(e1)
                test_gt_perMTL_inc_asarray.append(e2)
            np.save(logpath_+'/test/prediction.npy', test_pred_perMTL_inc_asarray)
            np.save(logpath_+'/test/groundtruth.npy', test_gt_perMTL_inc_asarray)
            
#            # Save plot for correlation between scale factor and prediction accuracy for complete samples
#            plt.figure(num=None, figsize=(13, 6), dpi=140, facecolor='w', edgecolor='k')
#            plt.plot(test_scale_factors, test_correct_pred_comp, 'r.')
#            plt.rcParams.update({'font.size': 20})
#            plt.xlabel('Side Ratio of original image')
#            plt.ylabel('Correctly predicted labels[%]')
#            plt.grid(axis='y', linewidth=2)
#            plt.savefig(logpath+'cv_'+str(cur_cv_iter)+'/test/sideratio2accuracy.png')
            
            # Save quality measure (per task) to disk
            if bool_CrossVal:
                logstring = 'Testing_cv_'+str(cur_cv_iter)
            else: 
                logstring = 'Testing_'
            for task_index in range(np.shape(test_pred_perMTL)[0]):
                wp4lib.estimate_quality_measures(test_gt_perMTL[task_index],
                                        test_pred_perMTL[task_index],
                                        list(collections_dict_MTL_test[
                                                list(class_count_dict)[task_index]].keys()),
                                        logstring + list(class_count_dict)[task_index],
                                        result_folder_name,
                                        how_many_training_steps, bool_MTL)
            
    
    if bool_CrossVal:
        return test_pred_perMTL, test_gt_perMTL, \
                class_count_dict, collections_dict_MTL_test
                
            



def apply_CNN_Classifier(configfile):
    """Applies the trained classifier to new data.
    
    :Arguments\::
        :configfile (*string*)\::
            This variable is a string and contains the name of the configfile. 
            All relevant information for applying the trained classifier is in this file.
            The configfile has to be stored in the same location as the
            script executing the classification function.
        
    :Returns:
        No returns. The classification result will be written automatically
        into result file in the master direction. The name of this file can
        be chosen by the user in the control file.
        
    """
    (master_file_name, master_dir, model_file, tfhub_module,
     classification_result_path, task_dict
     )            = import_control_file_apply(configfile)
    module_spec   = hub.load_module_spec(str(tfhub_module))
    height, width = hub.get_expected_image_size(module_spec)
    output_layer = 'final_result'
    if not os.path.isdir(classification_result_path):
        os.mkdir(classification_result_path)
    classification_result = classification_result_path+"classification_result.txt"
    classification_scores = classification_result_path+"classification_scores.txt"
    
    # Get image_lists out of the Master.txt
    master_id = open(os.path.abspath(master_dir + '/' + master_file_name), 'r')
    image_lists = []
    for image_list in master_id:
        image_lists.append(image_list)
    master_id.close()
    print('Got the following image lists for classification:', image_lists)
    (image_file_array) = image_lists_to_image_array(image_lists, master_dir)
    
    task_list = task_dict.keys()
    model = ImportGraph(model_file, output_layer, task_list)
    
    class_res_id = open(os.path.abspath(classification_result), 'w')
    class_score_file = open(os.path.abspath(classification_scores), 'w')
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
        for task in task_dict:
            class_list = task_dict[task]
            resultClasses[task] = class_list[resultTasks[task]]
        
        print(resultClasses)
        # Write class names to file
        class_res_id.write("%s" % (image_file))
        for task in task_list:
            class_res_id.write("\t%s" % (resultClasses[task]))
        class_res_id.write("\n")
            
        
        """TODO: INSERT OUTPUT OF CLASS SCORES"""
        class_score_file.write("****"+image_file +"****"+ "\n")
        for ti, task in enumerate(task_dict):
            class_score_file.write(task+": \t \t")
            for ci, c in enumerate(task_dict[task]):
                class_score_file.write(c + ": "+ str(np.around(results[ti][0][ci]*100,2))+"% \t \t")
                
            class_score_file.write("\n")
        class_score_file.write("\n")
        class_score_file.write("\n")
        
        
    class_res_id.close()
    class_score_file.close()


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
        import silknow_image_classification.src.SILKNOW_WP4_library as wp4lib

try:
    from image_classification_utility import *
except:
    try:
        sys.path.insert(0,'./src') 
        from image_classification_utility import *
    except:
        from silknow_image_classification.src.image_classification_utility import *
    
try:
    from create_dataset_utility import *
except:
    try:
        sys.path.insert(0,'./src') 
        from create_dataset_utility import *
    except:
        from silknow_image_classification.src.create_dataset_utility import *

#import urllib.parse
#import argparse

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

# The location where variable checkpoints will be stored.
CHECKPOINT_NAME = 'model.ckpt'

# A module is understood as instrumented for quantization with TF-Lite
# if it contains any of these ops.
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')


def createDataset(configfile):
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
    csvPath, minNumSamples, skipDownload, onlyFromCollection, \
    numFC, numNodes = importDatasetConfiguration(configfile)
    
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
    numVars = len(FieldCSV_list)
    image_data = image_data[np.asarray(FieldCSV_list)[:,1]]
    image_data = image_data.replace("NaN", np.nan)
    image_data["nancount"] = image_data.isnull().sum(axis=1)
    image_data = image_data[image_data.nancount != numVars]
    image_data = image_data[np.asarray(FieldCSV_list)[:,1]]
        
    
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
#    image_data = image_data[np.asarray(FieldCSV_list)[:,1]]
    
    createCNNConfigFile(image_data, MasterfilePath, numFC, numNodes)
    createTrainingConfigFile(MasterfilePath)
    createEvaluationConfigFile(MasterfilePath)
    createClassificationConfigFile(MasterfilePath)
    createCrossvalidationConfigFile(MasterfilePath)
    
    # If Masterfiles for Evaluation and Classification do net exist yet, create them
    if not os.path.isfile(MasterfilePath+"Masterfile_Classification.txt"):
        collection = open(MasterfilePath+"Masterfile_Classification.txt","w+")
        collection.write("Collection_Classification.txt")
        collection.close()
    if not os.path.isfile(MasterfilePath+"Collection_Classification.txt"):
        collection = open(MasterfilePath+"Collection_Classification.txt","w+")   
        collection.write("#image_name \n")     
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
     weight_decay, evaluate_model,_) = import_control_file_train(configfile)
    
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
        class_count_dict, collections_dict_MTL_test = training_CNN_Classifier(configfile, bonus_parameters)
        
        
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
                                  'Testing_average_'+MTL_task, result_folder_name,
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
    
    training_CNN_Classifier(configfile, evaluation_parameters)
    
    
def train_CNN_Classifier(configfile):
    """Trains a classifier based on top of a pre-trained CNN.
    
    :Arguments\::
        :configfile (*string*)\::
            This variable is a string and contains the name of the configfile.
            All relevant information for the training is in this file.
            The configfile has to be stored in the same location as the
            script executing the training function.
    
    :Returns\::
        No returns. The trained graph (containing the tfhub_module and the
        trained classifier) is stored automatically in the directory given in
        the control file.
    """
    training_CNN_Classifier(configfile)
    
   
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
    classification_result = classification_result_path+"classification_results.txt"
    classification_scores = classification_result_path+"classification_scores.txt"
    
    # Get image_lists out of the Master.txt
    # If an collection file is passed (header includes '#'), skip
    master_id = open(os.path.abspath(master_dir + '/' + master_file_name), 'r')
    image_lists = []
    for line, image_list in enumerate(master_id):
        if "#" in image_list and line==0:
            image_lists.append(master_dir + '/' + master_file_name)
            break
        else:
            image_lists.append(image_list)
    master_id.close()
#    print('Got the following image lists for classification:', image_lists)
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


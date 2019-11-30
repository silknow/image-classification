# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 20:43:30 2019

@author: clermont
"""

import urllib.request
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm


def make_id(data):
    """Creates unique identifier from graph and URI.

  :Arguments:
    :data:
        Pandas dataframe with fields "g" and "obj"

  :Returns:
    :identifier:
        Unique combined identifier for all entries in data.
    
  """
    identifier = (data["g"].apply(lambda x: x.rsplit('/', 1)[-1])+"__"+
                 data["obj"].apply(lambda x: x.rsplit('/', 1)[-1]))
    return identifier

def LoadAndMap(FieldCSV, Mapping_table, csvPath=""):
    """ Loads .csv for one field, maps its content to the class structure and
        saves it as a "cleaned" .csv.
    :Arguments:
        :FieldCSV:
            Path to .csv file for a single field. This is one entry from FieldCSV_list.
        :Mapping_table:
            Path to Excel table that has sheets named according to the files in FieldCSV_list.
            Every sheets contains two columns, i.e. Class and Values. The strings in the column
            Values will be mapped to the respective strings in the column Class.
        :csvPath:
            Path to the folder containing the .csv file.
    :Returns:
        :data:
            Dataframe with data from .csv, mapped to class structure
    """
    # Load .csv
    fieldname = FieldCSV[1]
    data = pd.read_csv(csvPath+FieldCSV[0]).drop_duplicates()
    index = make_id(data)
    data["identifier"] = index
    data = data.set_index("identifier").drop(columns=["id","obj","g"])
    
    # load mapping table
    xl_file = pd.ExcelFile(csvPath+Mapping_table)
    dfs = {sheet_name: xl_file.parse(sheet_name) 
              for sheet_name in xl_file.sheet_names}
    Mapping_dataframe = dfs[fieldname]
    
    # Map to classstructure, fill empty values with "NaN"
    for _, row in Mapping_dataframe.iterrows():
        data[fieldname] = data[fieldname].replace(row.Values, row.Class)
    data[fieldname] = data[fieldname].fillna('NaN')
    
    # Group duplicates together, concatenate values into list
    data = (data.groupby('identifier')[fieldname].apply(lambda x: list(set(x))).reset_index()).set_index("identifier")
    
    # only keep values that comply to the class structure, map the rest to "Other"
    valid_list = np.append(Mapping_dataframe.Class.unique(), "NaN")
    
    other_name = "Other_"+fieldname.capitalize()
    
    # #Handle multiple values in one field
    for idx in data.index:
        
        # get material list
        value_list = np.asarray(data.loc[idx].tolist(), dtype='<U100').flatten()
        
            
        # Filter NaNs, if empty, only NaN was there
        vl = value_list[value_list != "NaN"]
        if len(vl)==0:
            data.at[idx, fieldname] = "NaN"
            continue
        
        # Replace invalid strings with "Other"
        for i, value in enumerate(vl):
            if value not in valid_list:
                vl[i] = other_name
        
        # Filter Others, if empty, only Others was there
        vl = vl[vl != other_name]
        if len(vl)==0:
            data.at[idx, fieldname] = other_name
            continue
        vl.sort()
        data.at[idx, fieldname] = vl[0]
    
    # Do not include Other class for timespan
    if fieldname == "timespan":
        data.timespan = data.timespan.replace(r'Other_Timespan', 'NaN')
    
    """!!!"""
    data[fieldname] = data[fieldname].replace(other_name, 'NaN')
    
    #data.to_csv(csvPath+fieldname+"_cleaned.csv")
    
    return data

def checkNumSamples(data, fieldname, minNumSamples):
    """ Replace values with too few occurences with other class (exception for timespan)
    :Arguments:
        :data:
            Dataframe in which values will be replaced
        :fieldname:
            Field in Datamframe in which values will be replaced
        :minNumSamples:
            Values with less than minNumSamples occurences will be replaced with other class.
            For Timespan, those values will instead be mapped to NaN
            
    :Returns:
        :data:
            Dataframe with replaced values.
    """
    
    """!!!"""
    if fieldname=="timespan" or True: 
        for value, count in data[fieldname].value_counts().iteritems():
            if count <= minNumSamples and not value=="NaN":
                data[fieldname] = data[fieldname].replace(value, "NaN")
                
    else:
        for value, count in data[fieldname].value_counts().iteritems():
            if count <= minNumSamples and not value=="NaN":
                data[fieldname] = data[fieldname].replace(value, "Other_"+fieldname.capitalize())
            
    return data

def downloadNewImages(data, savepath):
    """ Download images that don't exist already.
        :Arguments:
            :data:
                Dataframe which has a list of URLs per record
            :savepath:
                Path to where the images will be downloaded
    """
    data["identifier"] = data.index
    # Create savepath
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    # Iterate over all records
    print("Starting download of images.")
    deadlinks = 0
    for index, row in tqdm(data.iterrows(), total=len(data.index)):
    
        # Skip record if image already exists
        if os.path.isfile(savepath+row.identifier+".jpg"): continue
        
        # Try to download from URL until one URL works
        try:
            url_list = list(row.url.strip('[').strip(']').split("'"))[1::2]
        except:
            url_list = row.url
        for url in url_list:
            try:
                urllib.request.urlretrieve(url, savepath+row.identifier+".jpg")
                break
            except:
                deadlinks += 1
                
        
    print("In total,",deadlinks, "records have no functioning image link!")
    

def importDatasetConfiguration(configfile):
    """Imports the dataset configuration from the configfile file.
    
    All relevant information for creating the dataset are contained in the
    configfile. This information is fed into the according variables in this
    function.
    Pay attention that all paths in the control file do not contain
    empty spaces!


    :Arguments:
        :configfile:
            This variable is a string and contains the name of the configfile. 
            All relevant information for the dataset creation are in this file.
    
    :Returns:
        :ImageCSV:
            Path to .csv file with image URLs for each record.
        :FieldCSV_list:
            Array of tuples. Every tuple contains (as its first element) the filename of 
            a .csv file for a target variable that will be included in the dataset. 
            The second element of every tuple is the name of the target variable.
            Example:[("material.csv","material"),("timespan.csv","timespan")]
        :MappingTable:
            Path to Excel table that has sheets named according to the names of the target variables in FieldCSV_list.
            Every sheets contains two columns, namely Class and Values. The strings
            in Values will be mapped to the respective strings in Class.
        :OutputDataPath:
            Path to the folder where the masterfile, controlfiles and images (in subfolder img) will be saved.
        :InputDataPath:
            Path to the folder containing the .csv files and the mapping table.
        :minNumSamples:
            Classes with less than minNumSamples occurrences will be replaced with their respective 'Other' class
        :skipDownload:
            If True, the images will not be downloaded.
        :onlyFromCollection:
            This string can be set to a name of a collection. Only record from this
            collection will be used for creating the dataset.
    """
    
    
    FieldCSV_list = [(r"depiction.csv","depiction"),
                     (r"material.csv","material"),
                     (r"place.csv","place"),
                     (r"technique.csv","technique"),
                     (r"timespan.csv","timespan")]
    FieldCSV_list_default = True
    OutputDataPath = r"./master_collection_images_csv/"
    onlyFromCollection = "imatex"
    InputDataPath = r"./master_collection_images_csv/"
    MappingTable = r"ClassMapping.xlsx"
    ImageCSV = r"imagelinks.csv"
    minNumSamples = 150
    
    control_id = open(configfile, 'r',encoding='utf-8')
    for variable in control_id:
        if variable.split(';')[0] == 'ImageCSV':
            ImageCSV = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'FieldCSV':
            if FieldCSV_list_default:
                FieldCSV_list_default = False
                FieldCSV_list = []
            csvfile = variable.split(';')[1].strip()
            csvname = variable.split(';')[2].strip()
            FieldCSV_list.append((csvfile, csvname))
        if variable.split(';')[0] == 'MappingTable':
            MappingTable = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'OutputDataPath':
            OutputDataPath = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'InputDataPath':
            InputDataPath = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'minNumSamples':
            minNumSamples = int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'skipDownload':
            skipDownload = variable.split(';')[1].strip()
            if skipDownload == 'True':
                skipDownload = True
            else:
                skipDownload = False
        if variable.split(';')[0] == 'onlyFromCollection':
            onlyFromCollection = variable.split(';')[1].strip()
            
            
    return ImageCSV, FieldCSV_list, MappingTable, OutputDataPath, \
            InputDataPath, minNumSamples, skipDownload, onlyFromCollection
    
def createCNNConfigFile(data, MasterfilePath):
    """ Creates a default CNN Configuration File.
        :Arguments:
            :data:
                Pandas Dataframe which lists all samples.
            :ConfigPath:
                Path to the master file.
    """
    config = open("CNNConfigurationFile.txt","w+")
    
    # Write Default Paths
    config.writelines(["****************FILES AND DIRECTORIES**************** \n"])
    config.writelines(["master_file_name; Masterfile.txt\n"])
    config.writelines(["master_dir; " +  MasterfilePath +"\n"])
    config.writelines(["logpath; "+  r"./logfiles/CNN_Default/"+"\n"])
    
    # Write Default Training Configuration
    config.writelines(["\n****************TRAINING SPECIFICATIONS**************** \n"])
    config.writelines(["train_batch_size; 300\n"])
    config.writelines(["how_many_training_steps; 300\n"])
    config.writelines(["learning_rate; 1e-4\n"])
    config.writelines(["validation_percentage; 25\n"])
    config.writelines(["how_often_validation; 10\n"])
    config.writelines(["weight_decay; 1e-2\n"])
    config.writelines(["num_finetune_layers; 2\n"])
    
    # Write Default Architecture Configuration
    config.writelines(["\n****************ARCHITECTURE SPECIFICATIONS**************** \n"])
    config.writelines(["tfhub_module; https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1\n"])
    config.writelines(["num_joint_fc_layer; 1\n"])
    config.writelines(["num_nodes_joint_fc; 1500\n"])
    
    # Write Default Data Augmentation Configuration
    config.writelines(["\n****************DATA AUGMENTATION SPECIFICATIONS**************** \n"])
    config.writelines(["flip_left_right; True\n"])
    config.writelines(["flip_up_down; True\n"])
    config.writelines(["random_rotation90; True\n"])
    
    
    # Write variables and their classes
    config.writelines(["\n****************CLASS STRUCTURE**************** \n"])
    variables = np.asarray(data.columns)
    variables.sort()
    for var in variables:
        classes = np.asarray(list(filter(lambda v: v==v, np.asarray(data[var].unique()))))
        classes = classes[classes != 'NaN']
        classes.sort()
        string = [" #"+name for name in classes]
        config.writelines(["variable_and_class; #"+var]+string+["\n"])
    
    # Write Default Paths for Evaluation
    config.writelines(["\n****************EVALUATION SPECIFICATIONS**************** \n"])
    config.writelines(["evaluation_result_path; ./evaluation_result/\n"])
    config.writelines(["evaluation_master_file_name; Masterfile_Evaluation.txt\n"])
    config.writelines(["evaluation_master_dir; " +  MasterfilePath +"\n"])
    config.writelines(["evaluation_index; 1\n"])
    
    # Write Default Paths for Classification
    config.writelines(["\n****************CLASSIFICATION SPECIFICATIONS**************** \n"])
    config.writelines(["classification_result_path; ./classification_result/\n"])
    config.writelines(["classification_master_file_name; Masterfile_Classification.txt\n"])
    config.writelines(["classification_master_dir; " +  MasterfilePath +"\n"])

    
    config.close()
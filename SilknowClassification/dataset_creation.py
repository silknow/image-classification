# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:28:07 2019

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
    fieldname = FieldCSV[:-4]
    data = pd.read_csv(csvPath+FieldCSV).drop_duplicates()
    index = make_id(data)
    data["identifier"] = index
    data = data.set_index("identifier").drop(columns=["id","obj","g"])
    
    # load mapping table
    xl_file = pd.ExcelFile(Mapping_table)
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
    
    data.to_csv(fieldname+"_cleaned.csv")
    
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
    if fieldname=="timespan":
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
        url_list = list(row.url.strip('[').strip(']').split("'"))[1::2]
        for url in url_list:
            try:
                urllib.request.urlretrieve(url, savepath+row.identifier+".jpg")
                break
            except:
                deadlinks += 1
                
        
    print("In total,",deadlinks, "records have no functioning image link!")

def CreateDataset(ImageCSV, FieldCSV_list, Mapping_table, imagePath, collectionsPath, csvPath="",
                  minNumSamples = 150, skipDownload = True, onlyFromCollection = ""):
    """Creates the dataset for the CNN.

  :Arguments:
    :ImageCSV:
        Path to .csv file with URLs for each record
    :FieldCSV_list:
        List of paths to .csv files from fields that will be included in the
        dataset.
        Example:["material.csv","timespan.csv"]
    :Mapping_table:
        Path to Excel table that has sheets named according to the files in FieldCSV_list.
        Every sheets contains two columns, namely Class and Values. The strings
        in Values will be mapped to the respective strings in Class.
    :imagePath:
        Path to the folder where the downloaded images will be saved.
    :collectionsPath:
        Path to the folder where the collection files will be saved.
    :csvPath:
        Path to the folder containing the .csv files .
    :minNumSamples:
        Classes with less than minNumSamples occurrences will be replaced with their respective 'Other' class
    :skipDownload:
        If True, the images will not be downloaded.
    :onlyFromCOllection:
        This string can be set to a name of a collection. Only record from this
        collection will be used for creating the dataset.

  :Returns:
    
  """
  

    
    # Load ImageCSV 
    image_data = pd.read_csv(ImageCSV)
    
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
        print("Loading and Mapping data from",field)
        data = LoadAndMap(field, Mapping_table, csvPath)
        image_data = pd.merge(image_data, data, left_index=True, right_index=True)
    
    # save merged dataframe, might contain dead links
    image_data.to_csv(r"dataset_cleaned_deadlinks.csv")
    
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
        print("Checking number of samples for", field)
        fieldname = field[:-4]
        image_data = checkNumSamples(image_data, fieldname, minNumSamples)
    
    # Save dataset as .csv
    if not onlyFromCollection == "":
        image_data.to_csv(r"dataset_cleaned_"+onlyFromCollection+".csv")
    else:
        image_data.to_csv(r"dataset_cleaned.csv")
    
    
    # Save dataset in five collection files, to be usable in MTL 
    print("Saving dataframe into collections files.")
    image_data["identifier"] = image_data.index
    dataChunkList = np.array_split(image_data, 5)
    for i, chunk in enumerate(dataChunkList):
        collection = open(collectionsPath+"collection_"+str(i+1)+".txt","w+")
        string = ["#"+name+"\t" for name in list(image_data)[1:]]
        collection.writelines(['#image_file\t']+string+["\n"])
        
        for index, row in chunk.iterrows():
            imagefile = str(row.identifier)+".jpg\t"
            
            # Skip improperly formatted filenames
            if "/" in imagefile: continue
    
            string = [(str(row[label])+"\t").replace('nan','NaN') for label in list(image_data)[1:]]
    
            collection.writelines([imagePath+imagefile]+string+["\n"])
    
        collection.close()
    
    # Print label statistics
    for field in FieldCSV_list:
        fieldname = field[:-4] 
        print("Classes for variable", fieldname,":")
        print(image_data[fieldname].value_counts(dropna=False))
    
    

if __name__ == '__main__':
    ImageCSV = r"imagelinks.csv"
    FieldCSV_list = [r"depiction.csv",
                     r"material.csv",
                     r"place.csv",
                     r"technique.csv",
                     r"timespan.csv"]
    Mapping_table = r"ClassMapping.xlsx"
    imagePath = r"../images/"
    collectionsPath = r"collections/"
    onlyFromCollection = "imatex"
    csvPath = r"../tests/"
    CreateDataset(ImageCSV, FieldCSV_list, Mapping_table, imagePath, collectionsPath, csvPath,
                  onlyFromCollection=onlyFromCollection)
    
    
    
    

    
    
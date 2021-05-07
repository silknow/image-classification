import silknow_image_classification as sic

# creates a dataset out of the knowledge graph export
# "collection[1-5].txt" are created
sic.create_dataset_parameter(csvfile=r"./samples/total_post.csv",
                             imgsavepath=r"./samples/data/",
                             master_file_dir=r"./samples/")

# performs training on a part of the dataset
# trains on "collection[1-4].txt"
sic.train_model_parameter(master_file_name="master_file_train.txt",
                          master_file_dir=r"./samples/",
                          log_dir=r"./output_files/log_dir/")

# performs classification on unseen images
# "collection_5.txt"
sic.classify_images_parameter(masterfile_name="master_file_test.txt",
                              masterfile_dir=r"./samples/",
                              model_dir=r"./output_files/log_dir/",
                              result_dir=r"./output_files/log_dir/")

# evaluates the classification
# "collection_5.txt"
sic.evaluate_model_parameter(pred_gt_dir=r"./output_files/log_dir/",
                             result_dir=r"./output_files/log_dir/")

# performs cross validation
# on all data
sic.crossvalidation_parameter(masterfile_name="master_file.txt",
                              masterfile_dir=r"./samples/",
                              log_dir=r"./output_files/log_dir/")
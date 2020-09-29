import sys

sys.path.insert(0, r"./src/")

import silk_classification_func as scf

"""--------------------------------------------- create dataset -----------------------------------------------------"""
# retainCollections = ['garin', 'imatex', 'joconde', 'mfa', 'risd','mad']
rawCSVFile = "./samples/total_post_08_June_2020.csv"
imageSaveDirectory = "./samples/data/"
masterfile_dir = "./samples/"
minNumSamplesPerClass = 100
retainCollections = ['garin', 'imatex', 'joconde', 'mfa', 'risd']
minNumLabelsPerSample = 1
flagDownloadImages = True
flagRescaleImages = True
fabricListFile = "fabricImages.txt"

scf.create_dataset_from_csv_parameter(rawCSVFile, imageSaveDirectory, masterfile_dir, minNumSamplesPerClass,
                                      retainCollections, minNumLabelsPerSample, flagDownloadImages, flagRescaleImages,
                                      fabricListFile)

"""--------------------------------------------- train model --------------------------------------------------------"""
aug_set_dict = {}
aug_set_dict["random_rotation90"] = True
aug_set_dict["flip_left_right"] = True
aug_set_dict["flip_up_down"] = True
aug_set_dict["gaussian_noise"] = 0.05

aug_set_dict["random_brightness"] = 10
aug_set_dict["random_crop"] = (0.5, 1.)
aug_set_dict["random_contrast"] = (0.9, 1.1)
aug_set_dict["random_hue"] = 0.1
aug_set_dict["random_saturation"] = (0.9, 1.1)

masterfile_name_train = "Masterfile_train.txt"
# masterfile_dir = "./samples/"
log_dir = "./output_files/focal/"
num_joint_fc_layer = 1
num_nodes_joint_fc = 500
num_finetune_layers = 2
relevant_variables = ["place", "timespan", "technique", "material", "depiction"]
batchsize = 300
how_many_training_steps = 1
how_often_validation = 10
validation_percentage = 25
learning_rate = 1e-4
weight_decay = 1e-3
num_task_stop_gradient = -1
aug_set_dict = aug_set_dict
image_based_samples = True
dropout_rate = 0.1
nameOfLossFunction = "focal"

scf.train_model_parameter(masterfile_name_train, masterfile_dir, log_dir, num_joint_fc_layer,
                          num_nodes_joint_fc, num_finetune_layers, relevant_variables,
                          batchsize, how_many_training_steps, how_often_validation,
                          validation_percentage, learning_rate, weight_decay,
                          num_task_stop_gradient, aug_set_dict, image_based_samples,
                          dropout_rate, nameOfLossFunction)

"""--------------------------------------------- classify images ----------------------------------------------------"""
masterfile_name_classify = "Masterfile_test.txt"
# masterfile_dir = "./samples/"
model_dir = log_dir
result_dir = log_dir
image_based_samples = True

scf.classify_images_parameter(masterfile_name_classify, masterfile_dir, model_dir,
                              result_dir, image_based_samples)

"""--------------------------------------------- evaluate model -----------------------------------------------------"""
pred_gt_dir = log_dir
result_dir = log_dir

scf.evaluate_model_parameter(pred_gt_dir, result_dir)

"""--------------------------------------------- 5-fold cross validation --------------------------------------------"""
masterfile_name_cv="Masterfile.txt"
# masterfile_dir="./samples/"
# log_dir="./output_files/focal/"
# num_joint_fc_layer=1
# num_nodes_joint_fc=500
# num_finetune_layers=2
# relevant_variables=["place", "timespan", "technique", "material", "depiction"]
# batchsize=300
# how_many_training_steps=1
# how_often_validation=10
# validation_percentage=25
# learning_rate=1e-4
# weight_decay=1e-3
# num_task_stop_gradient=-1
# aug_set_dict=aug_set_dict
# image_based_samples=True
# dropout_rate = 0.1
# nameOfLossFunction = "focal"
scf.crossvalidation_parameter(masterfile_name_cv, masterfile_dir, log_dir,
                              num_joint_fc_layer, num_nodes_joint_fc, num_finetune_layers,
                              relevant_variables, batchsize, how_many_training_steps,
                              how_often_validation, validation_percentage, learning_rate,
                              weight_decay, num_task_stop_gradient, aug_set_dict,
                              image_based_samples, dropout_rate, nameOfLossFunction)

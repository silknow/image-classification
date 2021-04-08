# import silk_classification_class as scc
# import DatasetCreation
from . import silk_classification_class as scc
from . import DatasetCreation


def create_dataset_from_csv_parameter(csvfile, imgsavepath, master_file_dir,
                                      minnumsamples, retaincollections, num_labeled,
                                      multi_label_variables=None):
    """Creates a dataset

    :Arguments\::
        :csvfile (*string*)\::
            Path including the filename of the csv file containing the data
            exported from the SILKNOW knowledge graph (by EURECOM) or to
            data that is structured in the same way. An example is given in
            https://github.com/silknow/image-classification/tree/master/silknow_image_classification/samples.
        :imgsavepath (*string*)\::
            Path to were the images shall be downloaded or were the beforehand
            downloaded images are stored. All images in the csv file (rawCSVFile)
            that fulfill the further user-defined requirements are considered,
            i.e. images in that director that are not part of the csv file won't be considered.
            All images have to be in the folder; subfolders are not considered.
        :master_file_dir (*string*)\::
            Path where the created master file will be stored. The master file contains all
            created collection files. These collection files contain the relative paths
            from masterfileDirectory to the images in imageSaveDirectory as well as the
            images' annotations.
        :minnumsamples (*int*)\::
            The minimum number of images that shall contribute to a certain class.
            Classes (e.g. Italy) with less than minNumSamplesPerClass samples are not considered
            for the variable (e.g. place) with that class (e.g. Italy), i.e. the class label is set
            to 'nan' (unknown).
        :retaincollections (*list of strings*)\::
            A list of the names of the museums in the csv file that shall be considered
            for the dataset creation.
        :num_labeled (*int*)\::
            The minimum number of labels that shall be available for a sample. If a sample
            has less labels, it won't be part of the created dataset. This number of labels is counted
            after potentially setting labels to 'nan' (unknown) due to the restrictions in
            minNumSamplesPerClass. The maximum number of labels is 5 in the current software
            implementation, i.e. one label for each of the five semantic variables
            (relevant_variables in other functions of silknow_image_classification).
        :multi_label_variables (*list of strings*)\::
            A list of the SILKNOW knowledge graph names of the five semantic variables that
            have multiple class labels per variable to be used in subsequent functions. A complete list
            would be ["material_group", "place_country_code", "time_label", "technique_group",
            "depict_group"].

    :Returns\::
        No returns. This function produces all files needed for running the subsequent software.
    """
    DatasetCreation.createDataset(rawCSVFile = csvfile,
                                  imageSaveDirectory=imgsavepath,
                                  masterfileDirectory=master_file_dir,
                                  minNumSamplesPerClass=minnumsamples,
                                  retainCollections=retaincollections,
                                  minNumLabelsPerSample=num_labeled,
                                  flagDownloadImages=True,
                                  flagRescaleImages=True,
                                  fabricListFile=None,
                                  multiLabelsListOfVariables=multi_label_variables)

def train_model_parameter(masterfile_name, masterfile_dir, log_dir, num_joint_fc_layer,
                          num_nodes_joint_fc, num_finetune_layers,
                          relevant_variables, batchsize,
                          how_many_training_steps, how_often_validation,
                          validation_percentage, learning_rate,
                          weight_decay, num_task_stop_gradient,
                          random_crop, random_rotation90, gaussian_noise,
                          flip_left_right, flip_up_down,
                          image_based_samples, dropout_rate,
                          nameOfLossFunction, multi_label_variables, lossParameters = {}):
    """Trains a new classifier.

        :Arguments\::
            :masterfile_name (*string*)\::
                Filename of the masterfile which states the collection files used for training and validation.
            :masterfile_dir (*string*)\::
                Directory where the masterfile is stored.
            :log_dir (*string*)\::
                Directory where the trained CNN will be stored.
            :num_joint_fc_layer (int)\::
                Number of joint fully connected layers.
            :num_nodes_joint_fc (int)\::
                Number of nodes in each joint fully connected layer.
            :num_finetune_layers (int)\::
                Number of layers of the pretrained feature extraction network that will be finetuned.
            :relevant_variables (list)\::
                List of strings that defines the relevant variables.
            :batchsize (int)\::
                Number of samples per training iteration.
            :how_many_training_steps (int)\::
                Number of training iterations.
            :how_often_validation (int)\::
                Number of training iterations between validation steps.
            :validation_percentage (int)\::
                Percentage of training samples that will be used for validation.
            :learning_rate (float)\::
                Learning rate.
            :weight_decay (float)\::
                Factor for the L2-loss of the weights.
            :num_task_stop_gradient (int)\::
                Samples with up to num_task_stop_gradient missing labels are used for training the joint layer.
                Samples with more missing labels will only be used for training the task-specific branches.
            :random_crop (*list*)\::
                Range of float fractions for centrally cropping the image. The crop fraction
                is drawn out of the provided range [lower bound, upper bound],
                i.e. the first and second values of random_crop. If [0.8, 0.9] is given,
                a crop fraction of e.g. 0.85 is drawn meaning that the crop for an image with
                the dimensions 200 x 400 pixels consists of the 170 x 340 central pixels.
            :random_rotation90 (*bool*)\::
                Data augmentation: should rotations by 90° be used (True) or not (False)?
            :gaussian_noise (*float*)\::
                Data augmentation: Standard deviation of the Gaussian noise
            :flip_left_right (*bool*)\::
                Data augmentation: should horizontal flips be used (True) or not (False)?
            :flip_up_down (*bool*)\::
                Data augmentation: should vertical flips be used (True) or not (False)?.
            :image_based_samples (bool)\::
                Has to be True if the collection files state image based samples (i.e. one image per sample).
                Has to be False if the collection files state record based samples (i.e. multiple images per sample).
            :dropout_rate (float)\::
                Value between 0 and 1 that defines the probability of randomly dropping activations between the
                pretrained feature extractor and the fully connected layers.
            :nameOfLossFunction (bool)\::
                Indicates the loss function that shall be used:
                    - If "sce": Softmax cross entropy loss for multi-task learning with incomplete samples.
                    (Note: both single-task learning and the complete samples case are special cases of "sce")
                    - If "focal": Focal softmax cross entropy loss for multi-task learning with incomplete samples.
                    (Note: both single-task learning and the complete samples case are special cases of "focal")
                    - If "mixed_sce": Softmax cross entropy loss (for variables listed in relevant_variables, but
                    not in multi_label_variables) combined with Sigmoid cross entropy loss
                    (for variables listed both in relevant_variables and multi_label_variables) for
                    multi-task learning with incomplete samples.
                    (Note: both single-task learning and the complete samples case are special cases of "mixed_sce")
            :multi_label_variables (*list of strings*)\::
                A list of variable names of the five semantic variables (relevant_variables) that
                have multiple class labels per variable to be used. A complete list
                would be ["material", "place", "timespan", "technique", "depiction"].
            :lossParameters (bool)\::
                Indicates (optional) parameters for the chosen loss function.
                Specifications can be found in LossCollections.py.

        :Returns\::
            No returns. This function produces all files needed for running the software.
        """

    # create new classifier object
    sc = scc.SilkClassifier()

    # set parameters
    sc.masterfile_name = masterfile_name
    sc.masterfile_dir = masterfile_dir
    sc.log_dir = log_dir

    sc.num_joint_fc_layer = num_joint_fc_layer
    sc.num_nodes_joint_fc = num_nodes_joint_fc
    sc.num_finetune_layers = num_finetune_layers

    sc.relevant_variables = relevant_variables
    sc.batchsize = batchsize
    sc.how_many_training_steps = how_many_training_steps
    sc.how_often_validation = how_often_validation
    sc.validation_percentage = validation_percentage
    sc.learning_rate = learning_rate
    sc.weight_decay = weight_decay
    sc.num_task_stop_gradient = num_task_stop_gradient
    sc.dropout_rate = dropout_rate
    sc.nameOfLossFunction = nameOfLossFunction
    sc.lossParameters = lossParameters

    sc.aug_set_dict['random_crop'] = random_crop
    sc.aug_set_dict['random_rotation90'] = random_rotation90
    sc.aug_set_dict['gaussian_noise'] = gaussian_noise
    sc.aug_set_dict['flip_left_right'] = flip_left_right
    sc.aug_set_dict['flip_up_down'] = flip_up_down

    sc.image_based_samples = image_based_samples
    sc.multiLabelsListOfVariables = multi_label_variables

    # call main function
    sc.train_model()


def classify_images_parameter(masterfile_name, masterfile_dir, model_dir, result_dir, bool_unlabeled_dataset = None,
                              image_based_samples=True, multi_label_variables=None, sigmoid_activation_thresh=0.5):
    """Classifies images.

    :Arguments\::
        :masterfile_name (*string*)\::
            Filename of the masterfile which states the collection files used for training and validation.
        :masterfile_dir (*string*)\::
            Directory where the masterfile is stored.
        :model_dir (*string*)\::
            Directory where the trained CNN will is stored. It is identical to log_dir in the
            training function.
        :result_dir (*string*)\::
            Directory where the results of the performed classification will be stored.
        :image_based_samples (bool)\::
            Has to be True if the collection files state image based samples (i.e. one image per sample).
            Has to be False if the collection files state record based samples (i.e. multiple images per sample).
        :multi_label_variables (*list of strings*)\::
            A list of variable names of the five semantic variables (relevant_variables) that
            have multiple class labels per variable to be used. A complete list
            would be ["material", "place", "timespan", "technique", "depiction"].
            This list has to match the list in multi_label_variables that was used for the training
            of the selected model in model_dir.
        :sigmoid_activation_thresh (*float*)\::
            Only if multi_label_variables is not None.
            A float threshold defining the minimum value of the sigmoid activation in case of a
            multi-label classification that a class needs to have to be predicted.


    :Returns\::
        No returns. This function produces all files needed for running the subsequent software.
    """
    # create new classifier object
    sc = scc.SilkClassifier()

    # set parameters
    sc.masterfile_name = masterfile_name
    sc.masterfile_dir = masterfile_dir
    sc.model_dir = model_dir
    sc.result_dir = result_dir
    sc.bool_unlabeled_dataset = bool_unlabeled_dataset

    sc.image_based_samples = image_based_samples
    sc.multiLabelsListOfVariables = multi_label_variables
    sc.sigmoid_activation_thresh=sigmoid_activation_thresh

    # call main function
    sc.classify_images()


def evaluate_model_parameter(pred_gt_dir, result_dir, multi_label_variables=None):
    """Evaluates a model

    :Arguments\::
        :pred_gt_dir (*string*)\::
            Directory containing the results of the classification function (result_dir in
            classify_images_parameter).
        :result_dir (*string*)\::
            Directory where the evaluation results will be stored.
        :multi_label_variables (*list of strings*)\::
            A list of variable names of the five semantic variables (relevant_variables) that
            have multiple class labels per variable to be used. A complete list
            would be ["material", "place", "timespan", "technique", "depiction"].
            This list has to match the list in multi_label_variables that was used for the training
            of the selected model to be evaluated.

    :Returns\::
        No returns.
    """
    # create new classifier object
    sc = scc.SilkClassifier()

    # set parameters
    sc.pred_gt_dir = pred_gt_dir
    sc.result_dir = result_dir
    sc.multiLabelsListOfVariables = multi_label_variables

    # call main function
    sc.evaluate_model()


def crossvalidation_parameter(masterfile_name, masterfile_dir, log_dir, num_finetune_layers,
                              relevant_variables, batchsize,
                              how_many_training_steps, how_often_validation,
                              validation_percentage, learning_rate, random_crop, random_rotation90, gaussian_noise,
                              flip_left_right, dropout_rate,
                              flip_up_down, nameOfLossFunction, multi_label_variables, sigmoid_activation_thresh):
    """Perform 5-fold crossvalidation

    :Arguments\::
        :masterfile_name (*string*)\::
            Filename of the masterfile which states the collection files used for training and validation.
        :masterfile_dir (*string*)\::
            Directory where the masterfile is stored.
        :log_dir (*string*)\::
            Directory where the trained CNN will be stored.
        :num_joint_fc_layer (int)\::
            Number of joint fully connected layers.
        :num_nodes_joint_fc (int)\::
            Number of nodes in each joint fully connected layer.
        :num_finetune_layers (int)\::
            Number of layers of the pretrained feature extraction network that will be finetuned.
        :relevant_variables (list)\::
            List of strings that defines the relevant variables.
        :batchsize (int)\::
            Number of samples per training iteration.
        :how_many_training_steps (int)\::
            Number of training iterations.
        :how_often_validation (int)\::
            Number of training iterations between validation steps.
        :validation_percentage (int)\::
            Percentage of training samples that will be used for validation.
        :learning_rate (float)\::
            Learning rate.
        :random_crop (*list*)\::
            Range of float fractions for centrally cropping the image. The crop fraction
            is drawn out of the provided range [lower bound, upper bound],
            i.e. the first and second values of random_crop. If [0.8, 0.9] is given,
            a crop fraction of e.g. 0.85 is drawn meaning that the crop for an image with
            the dimensions 200 x 400 pixels consists of the 170 x 340 central pixels.
        :random_rotation90 (*bool*)\::
            Data augmentation: should rotations by 90° be used (True) or not (False)?
        :gaussian_noise (*float*)\::
            Data augmentation: Standard deviation of the Gaussian noise
        :flip_left_right (*bool*)\::
            Data augmentation: should horizontal flips be used (True) or not (False)?
        :flip_up_down (*bool*)\::
            Data augmentation: should vertical flips be used (True) or not (False)?.
        :image_based_samples (bool)\::
            Has to be True if the collection files state image based samples (i.e. one image per sample).
            Has to be False if the collection files state record based samples (i.e. multiple images per sample).
        :dropout_rate (float)\::
            Value between 0 and 1 that defines the probability of randomly dropping activations between the
            pretrained feature extractor and the fully connected layers.
        :nameOfLossFunction (bool)\::
            Indicates the loss function that shall be used.
            Available functions are listed in LossCollections.py.
        :multi_label_variables (*list of strings*)\::
            A list of variable names of the five semantic variables (relevant_variables) that
            have multiple class labels per variable to be used. A complete list
            would be ["material", "place", "timespan", "technique", "depiction"].
        :sigmoid_activation_thresh (*float*)\::
            Only if multi_label_variables is not None.
            A float threshold defining the minimum value of the sigmoid activation in case of a
            multi-label classification that a class needs to have to be predicted.

    :Returns\::
        No returns.
    """
    # create new classifier object
    sc = scc.SilkClassifier()

    # set parameters
    sc.masterfile_name_cv = masterfile_name
    sc.masterfile_dir = masterfile_dir
    sc.log_dir_cv = log_dir

    sc.num_joint_fc_layer = 1
    sc.num_nodes_joint_fc = 1500
    sc.num_finetune_layers = num_finetune_layers
    sc.dropout_rate = dropout_rate

    sc.relevant_variables = relevant_variables
    sc.batchsize = batchsize
    sc.how_many_training_steps = how_many_training_steps
    sc.how_often_validation = how_often_validation
    sc.validation_percentage = validation_percentage
    sc.learning_rate = learning_rate
    sc.weight_decay = 1e-3
    sc.num_task_stop_gradient = -1

    sc.aug_set_dict['random_crop'] = random_crop
    sc.aug_set_dict['random_rotation90'] = random_rotation90
    sc.aug_set_dict['gaussian_noise'] = gaussian_noise
    sc.aug_set_dict['flip_left_right'] = flip_left_right
    sc.aug_set_dict['flip_up_down'] = flip_up_down

    sc.nameOfLossFunction = nameOfLossFunction
    sc.lossParameters = {}

    sc.image_based_samples = True
    sc.multiLabelsListOfVariables = multi_label_variables
    sc.sigmoid_activation_thresh=sigmoid_activation_thresh

    # call main function
    sc.crossvalidation()


# def trainWithDomainCheck(masterfile_name, masterfile_dir, log_dir,
#                          masterfileTarget,
#                           num_joint_fc_layer,
#                           num_nodes_joint_fc, num_finetune_layers,
#                           relevant_variables, batchsize,
#                           how_many_training_steps, how_often_validation,
#                           validation_percentage, learning_rate,
#                           weight_decay, num_task_stop_gradient,
#                           aug_set_dict, image_based_samples, dropout_rate,
#                           nameOfLossFunction, lossParameters = {},
#                          ):
#     """Trains a new classifier.
#
#         :Arguments\::
#             :configfile (*string*)\::
#                 Filename of the configuration file which defines all function parameters.
#
#         :Returns\::
#             No returns. This function produces all files needed for running the software.
#         """
#
#     # create new classifier object
#     sc = scc.SilkClassifier()
#
#     # set parameters
#     sc.masterfile_name = masterfile_name
#     sc.masterfile_dir = masterfile_dir
#     sc.log_dir = log_dir
#
#     sc.num_joint_fc_layer = num_joint_fc_layer
#     sc.num_nodes_joint_fc = num_nodes_joint_fc
#     sc.num_finetune_layers = num_finetune_layers
#
#     sc.relevant_variables = relevant_variables
#     sc.batchsize = batchsize
#     sc.how_many_training_steps = how_many_training_steps
#     sc.how_often_validation = how_often_validation
#     sc.validation_percentage = validation_percentage
#     sc.learning_rate = learning_rate
#     sc.weight_decay = weight_decay
#     sc.num_task_stop_gradient = num_task_stop_gradient
#     sc.aug_set_dict = aug_set_dict
#     sc.dropout_rate = dropout_rate
#     sc.nameOfLossFunction = nameOfLossFunction
#     sc.lossParameters = lossParameters
#
#     sc.image_based_samples = image_based_samples
#
#     sc.masterfileTarget = masterfileTarget
#
#     # call main function
#     sc.trainWithDomainCheck()
# import silk_classification_class as scc
# import DatasetCreation
from . import silk_classification_class as scc
from . import DatasetCreation


def create_dataset_parameter(csvfile,
                             imgsavepath,
                             master_file_dir,
                             minnumsamples=150,
                             retaincollections=['cer', 'garin', 'imatex', 'joconde', 'mad', 'met',
                                                'mfa', 'mobilier', 'mtmad', 'paris-musees', 'risd',
                                                'smithsonian', 'unipa', 'vam', 'venezia', 'versailles'],
                             num_labeled=1,
                             multi_label_variables=["material"]):
    """Creates a dataset

    :Arguments\::
        :csvfile (*string*)\::
            The name (including the path) of the CSV file containing the data exported from the SILKNOW knowledge graph.
        :imgsavepath (*string*)\::
            The path to the directory that will contain the downloaded images. The original images will be downloaded
            to the folder img_unscaled in that directory and the rescaled images (the smaller side will be 448 pixels)
            will be saved to the folder img. It has to be relative to the current working directory.
        :master_file_dir (*string*)\::
            Directory where the collection files and masterfile will be created. The storage location can now be chosen
            by the user.
        :minnumsamples (*int*)\::
            The minimum number of samples that has to be available for a single class or, in case the parameter
            multi_label_variables is not None, for every class combination for the variables contained in that list.
            The dataset is restricted to class combinations that occur at least minnumsamples times in the dataset
            exported from the knowledge graph. Classes or class combinations with fewer samples will not be considered
            in the generated dataset.
        :retaincollections (*list of strings*)\::
            A list containing the museums/collections in the knowledge graph that shall be considered for the data set
            creation. Data from museums/collections not stated in this list will be omitted. Possible values in the list
            according to EURECOM’s export from the SILKNOW knowledge graph (19.02.2021) are: cer, garin, imatex,
            joconde, mad, met, mfa, mobilier, mtmad, paris-musee, risd, smithsonian, unipa, vam, venezia, versailles.
        :num_labeled (*int*)\::
            A variable that indicates how many labels per sample should be available so that a sample is a valid sample
            and thus, part of the created dataset. The maximum value is 5, as five semantic variables are considered in
            the current implementation of this function. Choosing this maximum number means that only complete samples
            will form the dataset, while choosing a value of 0 means that records without annotations will also be
            considered. The value of num_labeled must not be smaller than 0.
        :multi_label_variables (*list of strings*)\::
            A list of keywords indicating those among the five semantic variables in the input CSV file (see csvfile)
            that may have multiple class labels per variable to be predicted. A complete list would be ["material",
            "place", "timespan", "technique", "depiction"]. If the value is None, all variables will have mutually
            exclusive labels, i.e. the generated dataset will not contain any samples with a class combination as a
            label.

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

def train_model_parameter(masterfile_name,
                          masterfile_dir,
                          log_dir,
                          num_finetune_layers=5,
                          relevant_variables=["material", "timespan", "technique", "depiction", "place"],
                          batchsize=300,
                          how_many_training_steps=500,
                          how_often_validation=10,
                          validation_percentage=25,
                          learning_rate=1e-3,
                          random_crop=[1., 1.],
                          random_rotation90=False,
                          gaussian_noise=0.0,
                          flip_left_right=False,
                          flip_up_down=False,
                          weight_decay=1e-3,
                          nameOfLossFunction="focal",
                          multi_label_variables=None,
                          num_joint_fc_layer=1,
                          num_nodes_joint_fc=1500):
    """Trains a new classifier.

        :Arguments\::
            :masterfile_name (*string*)\::
                Name of the master file that lists the collection files with the available samples that will be used
                for training the CNN. This file has to exist in directory master_dir.
            :masterfile_dir (*string*)\::
                Path to the directory containing the master file.
            :log_dir (*string*)\::
                Path to the directory to which the trained model and the log files will be saved.
            :num_finetune_layers (int)\::
                Number of residual blocks (each containing 3 convo- lutional layers) of ResNet 152 that shall be
                retrained.
            :relevant_variables (list)\::
                A list containing the names of the variables to be learned. These names have to be those (or a subset
                of those) listed in the header sections of the collection files collection_n.txt.
            :batchsize (int)\::
                Number of samples that are used during each training iteration.
            :how_many_training_steps (int)\::
                Number of training iterations.
            :how_often_validation (int)\::
                Number of training iterations between two computations of the validation loss.
            :validation_percentage (int)\::
                Percentage of training samples that are used for validation. The value has to be in the range [0, 100).
            :learning_rate (float)\::
                Learning rate for the training procedure.
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
            :weight_decay (float)\::
                Weight of the regularization term in the loss function.
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
                A list of those among the variables to be predicted (cf. relevant_variables) that may have multiple
                class labels per variable to be used in subsequent functions. A complete list would be ["material",
                "place", "timespan", "technique", "depiction"].
            :num_nodes_joint_fc (int)\::
                Number of nodes in each joint fully connected layer.
            :num_finetune_layers (int)\::
                Number of joint fully connected layers.

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
    sc.num_task_stop_gradient = -1
    sc.dropout_rate = 0.1
    sc.nameOfLossFunction = nameOfLossFunction
    sc.lossParameters = {}

    sc.aug_set_dict['random_crop'] = random_crop
    sc.aug_set_dict['random_rotation90'] = random_rotation90
    sc.aug_set_dict['gaussian_noise'] = gaussian_noise
    sc.aug_set_dict['flip_left_right'] = flip_left_right
    sc.aug_set_dict['flip_up_down'] = flip_up_down

    sc.image_based_samples = True
    sc.multiLabelsListOfVariables = multi_label_variables

    # call main function
    sc.train_model()


def classify_images_parameter(masterfile_name,
                              masterfile_dir,
                              model_dir,
                              result_dir,
                              multi_label_variables=None,
                              sigmoid_activation_thresh=0.5):
    """Classifies images.

    :Arguments\::
        :masterfile_name (*string*)\::
            Name of the master file that lists the collection files with the available samples that will be classified
            by the trained CNN in model_dir. This file has to exist in directory master_dir.
        :masterfile_dir (*string*)\::
            Path to the directory containing the master file master_file_name.
        :model_dir (*string*)\::
            Path to the directory with the trained model to be used for the classification. This directory is
            equivalent to log_dir in the function crossvalidation_parameter.
        :result_dir (*string*)\::
            Path to the directory to which the classification results will be saved. This directory is equivalent to
            log_dir in the function crossvalidation_parameter.
        :multi_label_variables (*list of strings*)\::
            A list of variable names of the semantic variables that have multiple class labels per variable to be used.
            A complete list would be ["material", "place", "timespan", "technique", "depiction"]. The performed
            classification of variables listed in this parameter is a multi-label classification where one binary
            classification per class is performed. All classes with a sigmoid activation larger than
            sigmoid_activation_thresh are part of the prediction.
            Note that this parameter setting has to be the same as the setting of multi_label_variables in the function
            train_model_parmeter at training time of the CNN that is loaded via model_dir!
        :sigmoid_activation_thresh (*float*)\::
            This variable is a float threshold defining the minimum value of the sigmoid activation in case of a
            multi-label classification that a class needs to have to be predicted. It is 0.5 per default in case that
            the user does not change the value. This parameter is only used, if multi_label_variables is different from
            None.

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
    sc.bool_unlabeled_dataset = True

    sc.image_based_samples = True
    sc.multiLabelsListOfVariables = multi_label_variables
    sc.sigmoid_activation_thresh=sigmoid_activation_thresh

    # call main function
    sc.classify_images()


def evaluate_model_parameter(pred_gt_dir,
                             result_dir,
                             multi_label_variables=None):
    """Evaluates a model

    :Arguments\::
        :pred_gt_dir (*string*)\::
            Path to the directory where the classification results to be evaluated are saved. This directory is
            equivalent to log_dir in the function crossvalidation_parameter.
        :result_dir (*string*)\::
            Path to the directory to which the evaluation results will be saved. This directory is equivalent to
            log_dir in the function crossvalidation_parameter.
        :multi_label_variables (*list of strings*)\::
            A list of variable names of the semantic variables that have multiple class labels per variable to be used.
            A complete list would be ["material", "place", "timespan", "technique", "depiction"]. This list has to be
            identical to the one used in the function classify_model_parmeter at the time of the classification that
            produced the predictions in pred_gt_dir.

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


def crossvalidation_parameter(masterfile_name,
                              masterfile_dir,
                              log_dir,
                              num_finetune_layers=5,
                              relevant_variables=["material", "timespan", "technique", "depiction", "place"],
                              batchsize=300,
                              how_many_training_steps=500,
                              how_often_validation=10,
                              validation_percentage=25,
                              learning_rate=1e-3,
                              random_crop=[1., 1.],
                              random_rotation90=False,
                              gaussian_noise=0.0,
                              flip_left_right=False,
                              flip_up_down=False,
                              weight_decay=1e-3,
                              nameOfLossFunction="focal",
                              multi_label_variables=None,
                              num_joint_fc_layer=1,
                              num_nodes_joint_fc=1500,
                              sigmoid_activation_thresh=0.5):
    """Perform 5-fold crossvalidation

    :Arguments\::
        :masterfile_name (*string*)\::
            Name of the master file that lists the collection files with the available samples that will be used
            for training the CNN. This file has to exist in directory master_dir.
        :masterfile_dir (*string*)\::
            Path to the directory containing the master file.
        :log_dir (*string*)\::
            Path to the directory to which the output files will be saved.
        :num_joint_fc_layer (int)\::
            Number of joint fully connected layers.
        :num_nodes_joint_fc (int)\::
            Number of nodes in each joint fully connected layer.
        :num_finetune_layers (int)\::
            Number of residual blocks (each containing 3 convo- lutional layers) of ResNet 152 that shall be
                retrained.
        :relevant_variables (list)\::
            A list containing the names of the variables to be learned. These names have to be those (or a subset
                of those) listed in the header sections of the collection files collection_n.txt.
        :batchsize (int)\::
            Number of samples that are used during each training iteration.
        :how_many_training_steps (int)\::
            Number of training iterations.
        :how_often_validation (int)\::
            Number of training iterations between two computations of the validation loss.
        :validation_percentage (int)\::
            Percentage of training samples that are used for validation. The value has to be in the range [0, 100).
        :learning_rate (float)\::
            Learning rate for the training procedure.
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
        :weight_decay (float)\::
            Weight of the regularization term in the loss function.
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
            A list of those among the variables to be predicted (cf. relevant_variables) that may have multiple
            class labels per variable to be used in subsequent functions. A complete list would be ["material",
            "place", "timespan", "technique", "depiction"].
        :sigmoid_activation_thresh (*float*)\::
            This variable is a float threshold defining the minimum value of the sigmoid activation in case of a
            multi-label classification that a class needs to have to be predicted. It is 0.5 per default in case that
            the user does not change the value. This parameter is only used, if multi_label_variables is different from
            None.

    :Returns\::
        No returns.
    """
    # create new classifier object
    sc = scc.SilkClassifier()

    # set parameters
    sc.masterfile_name_cv = masterfile_name
    sc.masterfile_dir = masterfile_dir
    sc.log_dir_cv = log_dir

    sc.num_joint_fc_layer = num_joint_fc_layer
    sc.num_nodes_joint_fc = num_nodes_joint_fc
    sc.num_finetune_layers = num_finetune_layers
    sc.dropout_rate = 0.1

    sc.relevant_variables = relevant_variables
    sc.batchsize = batchsize
    sc.how_many_training_steps = how_many_training_steps
    sc.how_often_validation = how_often_validation
    sc.validation_percentage = validation_percentage
    sc.learning_rate = learning_rate
    sc.weight_decay = weight_decay
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
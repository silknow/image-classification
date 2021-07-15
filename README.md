silknow_image_classification
===============================

version number: 0.0.3
author: LUH

Overview
--------

This software provides python functions for the classification of images and training and evaluation of classification models. It consists of five main parts:
    1. The creation of a dataset,
    2. the training of a new classifier,
    3. the evaluation of an existing classifier,
    4. the classification of images using an existing classifier and
    5. the combined training and evaluation in a five-fold cross validation.
    
All functions take explicit parameter settings as an input and generally write their results in speciﬁed paths. A documentation of the functions' parameters can be found in [documentation](https://github.com/silknow/image-classification/tree/master/silknow_image_classification/documentation) and further details are described in the SILKNOW Deliverable D4.6.

Installation / Usage
--------------------

To install clone the repo:

    $ git clone https://github.com/silknow/image-classification
    $ cd ./image-classification
    $ pip install --upgrade .

A pre-trained model that was created using this software can be download from https://doi.org/10.5281/zenodo.5091813. The training of that model is based on the `focal` loss using a mutli-task CNN architecture.

User Guidelines
-----------------

The user can download the [provided classification model](https://doi.org/10.5281/zenodo.5091813) and directly start to classify new images by means of the function `silknow_image_classification.classify_images_parameter`.

Alternatively, the user can train an own image classification model using the provided software for a subsequent image classification. Therefore, example calls of all functions are provided in [main.py](https://github.com/silknow/image-classification/blob/master/silknow_image_classification/main.py) using the [provided data files](https://github.com/silknow/image-classification/tree/master/silknow_image_classification/samples). These function calls will perform all steps listed in the overview above using the [provided knowledge graph export](https://github.com/silknow/image-classification/blob/master/silknow_image_classification/samples/total_post.csv).
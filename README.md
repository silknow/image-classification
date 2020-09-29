silknow_image_classification
===============================

version number: 0.0.2
author: LUH

Overview
--------

This software provides python functions for the classification of images and training and evaluation of classification models. It consists of five main parts: The creation of a dataset, the training of a new classifier, the evaluation of an existing classifier, the classification of images using an existing classifier and the combined training and evaluation in a five-fold cross validation. All functions take configuration files as an input and generally write their results in specified paths. The format required for the configuration files is described in Deliverable D4.4.  A description of the software's main functions can be found in the documentation at silknow_image_classification/documentation/(https://github.com/silknow/image-classification/tree/master/silknow_image_classification/documentation). 

Installation / Usage
--------------------

To install clone the repo:

    $ git clone https://github.com/Dorozynski/silknow_image_classification.git
    $ python setup.py install

Or:

    $ git clone https://github.com/Dorozynski/silknow_image_classification.git
    
    change the working directory to the folder of the setup.py
    
    $ pip install --upgrade .

Example
-------

An example of all function calls described in the documentation (see Overview) is provided in silknow_image_classification/main.py.
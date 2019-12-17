SILKNOW Image Classification
===============================

version number: 0.0.1
author: LUH

Overview
--------
This software provides python functions for the classification of images and training and evaluation of classification models. It consists of five main parts: The creation of a dataset, the training of a new classifier, the evaluation of an existing classifier, the classification of images using an existing classifier and the combined training and evaluation in a five-fold cross validation. All functions take configuration files as an input and generally write their results in specified paths. The format required for the configuration files is described in Deliverable D4.4.  A description of the software's main functions can be found in the documentation at silknow_image_classification/documentation/documentation.pdf. 


Installation / Usage
--------------------

To install the package silknow-image-classification use pip:

    $ pip install .
	
The installation will make sure that the following required packages are installed:
- Sphinx
- sphinx_rtd_theme
- nose
- coverage
- pypi-publisher
- urllib3
- numpy
- pandas
- tqdm
- tensorflow (version 1.13.1)
- tensorflow-hub (version 0.6.0)
- matplotlib
- sklearn
- xlrd

The pre-trained model that was created using this software can be download from http://doi.org/10.5281/zenodo.3577299.
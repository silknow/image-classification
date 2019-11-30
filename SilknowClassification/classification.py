# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 12:04:08 2019

@author: clermont
"""

import sys
try:
    import silknow_image_classification as sic
except:
    sys.path.insert(0,'./src') 
    import silknow_image_classification as sic

if __name__ == '__main__':
    configFile = sys.argv[1]
    sic.apply_CNN_Classifier(configFile)
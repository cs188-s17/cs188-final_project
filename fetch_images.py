"""Modified Olivetti faces dataset images set
 
   returns an array of images with 64x64 images. 

"""
# Copyright (c) 2011 David Warde-Farley <wardefar at iro dot umontreal dot ca>
# License: BSD 3 clause

from io import BytesIO
from os.path import exists
from os import makedirs
import os

import numpy as np
from scipy.io.matlab import loadmat

# Import for loading a file

from .base import get_data_home
from .base import Bunch
from .base import _pkl_filepath
from ..utils import check_random_state
from ..externals import joblib

# add our library
from img2array import img2array_converter

TARGET_FILENAME = "angiograms.pkz"

# Grab the module-level docstring to use as a description of the
# dataset
MODULE_DOCS = __doc__

def fetch_angiogram_images(data_home=None, shuffle=False, random_state=0,
                         download_if_missing=True):
    # get the data home
    data_home = get_data_home(data_home=data_home)

    # if the directory doesn't exist, create it
    if not exists(data_home):
        makedirs(data_home)

    # set the filepath to the target filename, a .pkz file
    filepath = _pkl_filepath(data_home, TARGET_FILENAME)

    if not exists(filepath):
    	print(' file does not exist, creating file ')

        # since file doesn't exist, create a file path for the future pkz file. 
	filepath = _pkl_filepath(data_home, TARGET_FILENAME)
	
	# create the array of faces, calling the function from img2array
	faces = img2array_converter()

	# joblib compresses the array in a file with the given file path, a .pkz file
        joblib.dump(faces, filepath, compress=6)

    else:
        faces = joblib.load(filepath)

    # We want floating point data, but float32 is enough (there is only
    # one byte of precision in the original uint8s anyway)
    # process the images
    faces = np.float32(faces)
    faces = faces - faces.min()

    # this is equivalent to divide AND
    faces /= faces.max()

    # reshape the array
    faces = faces.reshape((400, 64, 64)).transpose(0, 2, 1)

    # 10 images per class, 400 images total, each class is contiguous.
    target = np.array([i // 10 for i in range(400)])

    if shuffle:
        random_state = check_random_state(random_state)
        order = random_state.permutation(len(faces))
        faces = faces[order]
        target = target[order]
    return Bunch(data=faces.reshape(len(faces), -1),
                 images=faces,
                 target=target,
                 DESCR=MODULE_DOCS)

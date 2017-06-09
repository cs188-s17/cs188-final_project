"""
Processing angiogram image data.
"""
print(__doc__)

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.datasets
import pdb
from skimage.io import imread

from PIL import Image
#from pkl import _pkl_filepath

from sklearn.utils.validation import check_random_state
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV


"""
Image processing: 

   Data required is in a folder called edited_data in the directory from where the program is run.
	
"""
def fetch_images():
	# Grab the module-level docstring to use as a description of the
	# dataset
	MODULE_DOCS = __doc__

	# a list of images to be stitched into an array
	image_list = glob.glob('before/*.png')
	n_images = len(image_list)

	# create a numpy array of the images
	image_array = np.array([imread(f, True) for f in image_list])
	#image_array = np.array([Image.open(fname).convert("L") for fname in image_list])
	faces = np.float32(image_array)
		
	faces = faces - faces.min()
	faces /= faces.max()
	faces = faces.reshape((n_images, 64, 64)).transpose(0, 2, 1)

	# create the target 
	target = np.array([i // 1 for i in range(n_images)])

	# create the data part
	data=faces.reshape(len(faces), -1)

	# create the descr
	DESCR=MODULE_DOCS

	data_group = sklearn.datasets.base.Bunch(data=data, images=faces, target=target, DESCR=DESCR)
	return data_group
	#return Bunch(data=data, images=faces, target=target, DESCR=DESCR)


"""
There are 2 different images (a before and an 
after) for 14 different images. 
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
def fetch_drawings():
	# Grab the module-level docstring to use as a description of the
	# dataset
	MODULE_DOCS = __doc__

	# number of images
	#n_images = 28

	# a list of images to be stitched into an array
	image_list = glob.glob('final_after_drawings/*.png')
	n_images = len(image_list)

	# create a numpy array of the images
	image_array = np.array([imread(f, True) for f in image_list])
	#image_array = np.array([Image.open(fname).convert("L") for fname in image_list])
	faces = np.float32(image_array)

	
	#print(faces.shape())
	faces = faces - faces.min()
	faces /= faces.max()
	faces = faces.reshape((n_images, 64, 64)).transpose(0, 2, 1)
	#faces = faces.reshape((28, 64, 64))
	"""
	print(faces)
	"""

	# create the target 
	#target = np.array([i // 2 for i in range(n_images)])
	target = np.array([i // 1 for i in range(n_images)])
	#print(target)

	# create the data part
	data=faces.reshape(len(faces), -1)
	#print(data)

	# create the descr
	DESCR=MODULE_DOCS

	data_group = sklearn.datasets.base.Bunch(data=data, images=faces, target=target, DESCR=DESCR)
	return data_group
	#return Bunch(data=data, images=faces, target=target, DESCR=DESCR)


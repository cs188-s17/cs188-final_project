import numpy as np
from PIL import Image

# function to take in images, convert them to array of arrays
def img2array_converter():

	# initialize an empty array
	list_of_arrays = []
	    
	# attempt to open files 
	try: 
		f = [Image.open("im_%d.png" % i) for i in range(1, 50)]
		# action on each image 
		converted_img = f.convert('RGBA')
		arr = np.array(converted_img)

		# record the original shape
		shape = arr.shape

		# make a 1-dimensional view of arr
		flat_arr = arr.ravel()

		# convert it to a matrix
		vector = np.matrix(flat_arr)

		# reform a numpy array of the original shape
		fin_array = np.asarray(vector).reshape(shape)
		list_of_arrays.append(fin_array)
			 
	finally:
		# make an array of arrays of the image data
		mat = numpy.array(list_of_arrays)
		for fh in f:
			fh.close()

	return mat

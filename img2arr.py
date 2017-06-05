import numpy
from PIL import Image

import os

print("start")
# function to take in images, convert them to array of arrays
def img2array_converter():
	# initialize an empty array
	list_of_arrays = []
	
	f = []
	    
	# attempt to open files 
	try: 
		f = [Image.open(os.path.join("edited data", "im_%d.png") % i).convert('RGBA') for i in range(1, 29)]
		for img in f:
			# action on each image
			arr = numpy.array(img)

			# record the original shape
			shape = arr.shape

			# make a 1-dimensional view of arr
			flat_arr = arr.ravel()

			# convert it to a matrix
			vector = numpy.matrix(flat_arr)

			# reform a numpy array of the original shape
			fin_array = numpy.asarray(vector).reshape(shape)
			list_of_arrays.append(fin_array)
		
		# action on each image 
		# converted_img = Image.open(os.path.join("edited data", "im_3.png")).convert('RGBA')
		#converted_img = f.convert('RGBA')
		# arr = numpy.array(converted_img)
		# arr = numpy.array(f)

		# record the original shape
		# shape = arr.shape

		# make a 1-dimensional view of arr
		# flat_arr = arr.ravel()

		# convert it to a matrix
		# vector = numpy.matrix(flat_arr)

		# reform a numpy array of the original shape
		# fin_array = numpy.asarray(vector).reshape(shape)
		# list_of_arrays.append(fin_array)
		
			 
	finally:
		# make an array of arrays of the image data
		mat = numpy.array(list_of_arrays)
		for fh in f:
			fh.close()
	print("reached end of program")	
	return mat

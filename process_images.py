"""
There are 2 different images (a before and an 
after) for 14 different images. 
"""
import numpy as np
import os
import glob
from PIL import Image
from pkl import _pkl_filepath

"""
Image processing: 

   Data required is in a folder called edited_data in the directory from where the program is run.
	
"""
# number of images
n_images = 30

# a list of images to be stitched into an array
image_list = glob.glob('edited_data/*.png')

# create a numpy array of the images
image_array = np.array([np.array(Image.open(fname).convert('RGBA')) for fname in image_list])

# create the target 
target = np.array([i // 2 for i in range(n_images)])
print(target)

"""
  faces = np.float32(faces)
    faces = faces - faces.min()
    faces /= faces.max()
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
"""

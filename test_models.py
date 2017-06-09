"""
==============================================
Face completion with a multi-output estimators
==============================================
This example shows the use of multi-output estimator to complete images.
The first column of images shows true output. The next columns illustrate
the method using randomized trees, k nearest neighbors, linear
regression and ridge regression..
"""
print(__doc__)

import pdb
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils.validation import check_random_state
from old_process_images import old_fetch_images 
from process_images import fetch_images 
from process_drawings import fetch_drawings 

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# import the image before angiogram data
data_images = fetch_images()

# similarly, import the corresponding annotations
data_drawings = fetch_drawings()

# get the image target values
im_targets = data_images.target

# get the drawing target values
draw_targets = data_drawings.target

# reshape the images array
data_images = data_images.images.reshape((len(data_images.images), -1))

# reshape the annotations array
data_drawings = data_drawings.images.reshape((len(data_drawings.images), -1))

# train on the before image, and on the after drawing
train_im = data_images[im_targets < 11]
train_draw = data_drawings[draw_targets < 11]

# test the after drawing on the before images
test_im = data_images[im_targets >= 11]
test_draw = data_drawings[draw_targets >= 11] 

# Test on a subset of people
n_vals = 5
rng = check_random_state(4)
img_ids = rng.randint(test_im.shape[0], size=(n_vals, ))

test_im = test_im[img_ids, :]
test_draw = test_draw[img_ids, :]

n_pixels = data_images.shape[1]

# train on the images
X_train = train_im[:, :]  

# train on the annotations
y_train = train_draw[:, :]

# test using the images ... 
X_test = test_im[:, :]

# on the drawings
y_test = test_draw[:, :]
#pdb.set_trace()

# Fit estimators
ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32,
                                       random_state=0),
    "K-nn": KNeighborsRegressor(),
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
}

y_test_predict = dict()

# fit the training data to the estimator models above
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)

# Plot the completed faces
image_shape = (64, 64)

# How many columns your output file has
n_cols = 1 + len(ESTIMATORS)

# plot details
plt.figure(figsize=(2. * n_cols, 2.26 * n_vals))

# plot title
plt.suptitle("Image prediction with multi-output estimators", size=16)

for i in range(n_vals):

    # the true drawing would be the annotation we had matched 
    true_pred = y_test[i]

    if i:
        sub = plt.subplot(n_vals, n_cols, i * n_cols + 1)
    else: 
        sub = plt.subplot(n_vals, n_cols, i * n_cols + 1,
                          title="Expected Output Value")


    sub.axis("off")
    sub.imshow(true_pred.reshape(image_shape),
               cmap=plt.cm.gray,
               interpolation="nearest")

    for j, est in enumerate(sorted(ESTIMATORS)):

	# the predicted image from the estimators...
        predicted_final = y_test_predict[est][i]

        if i:
            sub = plt.subplot(n_vals, n_cols, i * n_cols + 2 + j)

        else:
            sub = plt.subplot(n_vals, n_cols, i * n_cols + 2 + j,
                              title=est)

        sub.axis("off")
        sub.imshow(predicted_final.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")

plt.show()

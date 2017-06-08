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
from process_images import fetch_images 

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# this method returns a .pkz file of images in 64x64 format
data = fetch_images()
targets = data.target
#pdb.set_trace()
#print(data)
#print(targets)

data = data.images.reshape((len(data.images), -1))
#print(data[0])
train = data[targets < 11] 
test = data[targets >= 11]  # Test on independent people

# Test on a subset of people
n_faces = 5
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces, ))
test = test[face_ids, :]

n_pixels = data.shape[1]
#pdb.set_trace()
# FIXME don't bother selecting halves
X_train = train[:, :np.ceil(0.5 * n_pixels)]  # Upper half of the faces
y_train = train[:, np.floor(0.5 * n_pixels):]  # Lower half of the faces
X_test = test[:, :np.ceil(0.5 * n_pixels)]
y_test = test[:, np.floor(0.5 * n_pixels):]
'''
X_train = train[:, :]  # Upper half of the faces
y_train = train[:, :]  # Lower half of the faces
X_test = test[:, :]
y_test = test[:, :]
'''
# Fit estimators
ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32,
                                       random_state=0),
    "K-nn": KNeighborsRegressor(),
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
}

y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)

# Plot the completed faces
image_shape = (64, 64)

n_cols = 1 + len(ESTIMATORS)
plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
plt.suptitle("Face completion with multi-output estimators", size=16)

for i in range(n_faces):
    pdb.set_trace()
    true_face = np.hstack((X_test[i], y_test[i]))

    if i:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
    else:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1,
                          title="sneha faces")


    sub.axis("off")
    #pdb.set_trace()
    sub.imshow(true_face.reshape(image_shape),
               cmap=plt.cm.gray,
               interpolation="nearest")

    for j, est in enumerate(sorted(ESTIMATORS)):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)

        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j,
                              title=est)

        sub.axis("off")
        sub.imshow(completed_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")

plt.show()

'''
Fashion Data

pca.fit(X) needs X to be "array-like, shape (n_samples, n_features)".
See here: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA.fit

You can use this code to bring the data into a format to be used with PCA. 
Note, that the images of men clothing or in a folder called 'man_200'.
Note also, that you would need to do this for two types of your choosing to 
perform classification.

It's a good idea to print out the arrays and shapes at each step to know what's
happening in the background!

There are probably a million ways to do this, don't feel constrained to use this snippet.
'''

from glob import glob
from PIL import Image
from resizeimage import resizeimage
import numpy as np

# create paths for all images
man_images = glob('Jerseys/*')

man_flattened = []
# for each image path
for path in man_images:
    # open it as a read file in binary mode
    with open(path, 'r+b') as f:
        # open it as an image
        with Image.open(f) as image:
            # resize the image to be more manageable
            cover = resizeimage.resize_cover(image, [20, 10])
            # flatten the matrix to an array and append it to all flattened images
            man_flattened.append((np.array(cover).flatten(), 0))
            

# Flatten it once more
man_flattened = np.asarray(man_flattened)

# Declare which are the X and Y inputs
X = man_flattened[:,0]
Y = man_flattened[:,1]

# Use np.stack to put the data into the right dimension
X = np.stack(i for i in X)
Y = np.stack(i for i in Y)

print("done")
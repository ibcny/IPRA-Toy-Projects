import numpy as np
from scipy import ndimage
import scipy.misc as misc
import time

imagePath = 'Bikesgray'

# window size
w_size = 9

#  Standard deviation for range distance. A larger value results in
#  averaging of pixels with larger spatial differences.
sigma_spatial = 5

# Standard deviation for grayvalue distance. A larger value results
#  in averaging of pixels with larger radiometric differences
sigma_color   = 0.1

# Middle point eg. if w_size = 11 => mOver2 = 5
mOver2 = int(np.floor(w_size / 2))

# Create Gaussian Kernel
x = np.arange(w_size)
g = np.exp(-0.5 * ((x - np.floor(w_size / 2.)) / sigma_spatial) ** 2)  # gaussian
distance_kern = np.outer(g, g)
distance_kern /= distance_kern.sum()

image = ndimage.imread(imagePath + ".jpg")

# Get image's width and height
imWidth = image.shape[0]
imHeight = image.shape[1]

# If it is an RGB image, take a single channel
if len(image.shape) == 3:
    image = image[:, :, 0]

image= image/255.  # normalize image
start = time.time()

# Initialize weight ind image as zero
out_image = np.zeros((imWidth, imHeight), dtype=np.float)
weights = np.zeros((imWidth, imHeight), dtype=np.float)

# Traverse every (i, j) eg. -5 to +5
for i in range(-mOver2, mOver2 + 1):
    for j in range(-mOver2, mOver2 + 1):
        # Shift the image by i,j to get f(x-i, y-j)
        rolledImage = np.roll(np.roll(image, i, axis=0), j, axis=1)

        # Find f(x-i, y-j) - f(x, y)
        intensity_dif = (rolledImage - image)

        # Multiplication of intensity difference weight and distance kernel
        partsTimeShifted = rolledImage * distance_kern[mOver2 + i, mOver2 + j]

        # G_{\rho}
        tmp = np.exp(-(np.multiply(intensity_dif, intensity_dif) / (2. * (sigma_color ** 2))))

        # A sum for i, j
        ij = np.multiply(tmp, partsTimeShifted)

        # do the sum operation
        out_image = out_image + ij

        # weight for this (i, j) pair. add it to all (i, j)s
        weights = weights + (tmp * distance_kern[mOver2 + i, mOver2 + j])

# Normalize every location
normalizedImage = np.divide(out_image, weights)

# This is the result
misc.imsave(imagePath + "_result_win_size_"+str(w_size)+"_sigma_spatial_"+str(sigma_spatial)+"_sigma_color_"+
            str(sigma_color)+".png", normalizedImage)

end = time.time()
print ("The operation took {0} milliseconds".format(end - start))

import numpy as np
import scipy.misc as misc
from scipy import ndimage

imageName = "clock"
r = 16

inputImage = misc.imread(imageName + ".jpg")

if len(inputImage.shape) == 3:
    inputImage = inputImage[:, :, 0]


def anamorphosis(input_image, radius):

    width, height = input_image.shape[0], input_image.shape[1]
    mid_x, mid_y = width/2, height/2

    mesh = np.meshgrid(range(width), range(height))
    # Calculate angle to center
    angle = np.mod(np.arctan2(mesh[0] - mid_x, mesh[1] - mid_y) + 2*np.pi, 2*np.pi)

    # Calculate distance to center and compare to max distance proportion excluding r
    distance_to_center = np.sqrt((mesh[0]-mid_x)**2+(mesh[1]-mid_y)**2)
    fraction_to_center = (distance_to_center-radius)/((mid_x-radius))

    # Higher the angle, from right most part get the y index
    y_index = (height-1) - (height-1) * angle/(2*np.pi)

    # Higher the fraction to center, get from upper x index
    x_index = (width-1) - (width-1)*fraction_to_center

    out_image = ndimage.map_coordinates(input_image, [x_index, y_index])
    out_image = np.rot90(out_image, -1) # Rotate clock wise, since we assumed 0, 0 to be left bottom
    return out_image


# Task a
wrapped = anamorphosis(input_image=inputImage, radius = 0)
misc.imsave("{0}_task_a.png".format(imageName), wrapped)

# Task b

r = 16
outputImage = anamorphosis(inputImage, r)
misc.imsave("{0}_r:{1}.png".format(imageName, r), outputImage)

r = 64
outputImage = anamorphosis(inputImage, r)
misc.imsave("{0}_r:{1}.png".format(imageName, r), outputImage)
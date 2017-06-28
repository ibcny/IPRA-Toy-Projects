import numpy as np

import scipy.misc as msc


def read_intensity_image(file):
    """
    Reads an image file and flattens the color layers into a single gray-scale layer.
    Parameters
    ----------
    file : str or file object
        The file name or file object to be read.
    Returns
    -------
    f : ndarray
        The 2D array obtained by reading the image and casting to float type.
    """
    f = msc.imread(file, flatten=True).astype('float')
    return f


def write_intensity_image(f, file):
    """
    Saves the image under the given file name or file object.
    Parameters
    ----------
    file : str
        The file name or file object in which the image array to be written.
   f : ndarray
        The 2D image array to be written to the given file.
    """
    msc.toimage(f, cmin=0, cmax=255).save(file)


def draw_ring(image, rMin, rMax):
    """
    Draws a ring/disk coloured in black around the center of the given 2-D grayscale image.
    Parameters
    ----------
    image : ndarray
        2-D grayscale image.
    rMin : scalar
        minimum distance of the ring/disk to the center of the image.
    rMax : scalar
        maxumum distance of the ring/disk to the center of the image.

    Returns
    -------
    image : ndarray
        2-D grayscale image with a ring around the center.
    """
    width, height = image.shape[0], image.shape[1]
    center = (width/2, height/2)
    y, x = np.ogrid[-center[1]:center[1], -center[0]:center[0]]  # open grid used to create the mask: -128:127
    dist = x ** 2 + y ** 2                                       # distance^2 matrix to the center
    mask = (dist <= rMax**2) & (dist >= rMin**2)                 # mask to be used as a boolean index
    image[mask] = 0.
    return image


if __name__ == '__main__':
    ff = read_intensity_image('clock.jpg')          # read image in grayscale
    ff = draw_ring(ff, 20, 80)                      # draw a ring in the center of the image
    write_intensity_image(ff, 'task_1.1_out.jpg')   # save the image under the given file name

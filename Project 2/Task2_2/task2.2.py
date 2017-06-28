import numpy as np 
import scipy.misc as msc
import scipy.ndimage as img
import math

# z-value corresponding to 0.01 error
z_value = 2.575

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


def smoothed_edge_detector(im, im_name, padding, w_size):
    """
    Apply gradient operator to a smoothed version of the input image
    Parameters
    ----------
    im: ndarray
        The 2D image array to be filtered
    padding: string
        border padding: "constant", "wrap", "reflect"
    w_size
        window size of the Gaussian Filter
    """
    sigma = (w_size-1.0) / (2.0*z_value)
    x = np.arange(w_size)
    g = np.exp(-0.5*((x - np.floor(w_size/2.))/sigma)**2)  # 1D gaussian kernel
    g /= g.sum()                                           # normalization
    msc.imsave('gauss1D.png', g.reshape(1, w_size))
    G = np.outer(g, g)                                     # 2D Gaussian Kernel
    G /= G.sum()                                           # Normalization
    msc.imsave('gauss2D.png', G)

    dy, dx = np.gradient(G)     #x,y gradient of the filter
    msc.imsave('smoothed_edge_filter_kernel_Y.png', dy)
    msc.imsave('smoothed_edge_filter_kernel_X.png', dx)

    # calculate smoothed edges
    grad_x = img.convolve(im, dx, mode=padding, cval=0.0)
    grad_y = img.convolve(im, dy, mode=padding, cval=0.0)

    msc.imsave(im_name+'_smoothed_edge_x.png', grad_x)
    msc.imsave(im_name+'_smoothed_edge_y.png', grad_y)

    # image emphasizing edges after smoothing with a Gaussian filter
    grad_xy = np.sqrt(grad_x ** 2 + grad_y ** 2)
    msc.imsave(im_name+"_smoothed_edge_xy.png", grad_xy)

if __name__ == '__main__':
    image = "bauckhage"
    ff = read_intensity_image(image+".jpg")      # read image in grayscale
    smoothed_edge_detector(ff, image, "constant", 7)      # task2: detected edges after smoothing with a Gauss. kernel

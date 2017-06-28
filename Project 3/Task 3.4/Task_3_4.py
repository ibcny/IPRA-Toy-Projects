import numpy as np
import scipy.misc as msc
from scipy import ndimage


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


def polar_transform(im, save):
    """
    Polar transformation from the cartesian coordinates of the given image
    Parameters
    ----------
    im: ndarray
        The 2D image array to be transformed
    save:
        Save intermediary output during the transformation process
    """
    height, width = im.shape[0], im.shape[1]
    # pole coincides with the center point of the image
    center = (height/2, width/2)
    r_min, r_max = 0., np.sqrt(center[0]**2 + center[1]**2)
    phi_min, phi_max = 0, 2*np.pi

    r, phi = np.meshgrid(np.linspace(r_min,   r_max,   width),
                         np.linspace(phi_min, phi_max, height))

    # map cartesian coordinates to polar coordinates
    im_polar = ndimage.map_coordinates(im, [center[0] + np.multiply(r, np.sin(phi)),
                                                      center[1] + np.multiply(r, np.cos(phi))])
    if save:
        write_intensity_image(im_polar, "polar_image.jpg")

    return im_polar


def cart_transform(im, w_size, sigma, padding, save):
    """
    Cartesian coordinates transformation from the polar coordinates of the given image
    Parameters
    ----------
    im: ndarray
        The 2D image array to be transformed
    w_size
        window size of the Gaussian Filter
    sigma:
        standart dev. of the Gaussian kernel
    padding: string
        border padding: "constant", "wrap", "reflect"
    save:
        If true, save intermediary output during the transformation process
    """
    height, width = im.shape[0], im.shape[1]
    center = (width/2, height/2)     # pole coincides with the center point of the image
    r_scale = np.sqrt(center[0]**2 + center[1]**2)
    phi_scale = 2*np.pi

    x, y = np.meshgrid(np.arange(-center[1], center[1]), np.arange(-center[0], center[0]))
    y = -y; x = -x

    # map polar coordinates to cartesian coordinates
    im_cart = ndimage.map_coordinates(im, [ (height-1)*(np.arctan2(y, x)+np.pi)/phi_scale ,
                                            (width-1)*np.sqrt(x**2 + y**2)/r_scale ])

    if save:
        write_intensity_image(im_cart, "cart_image_w_size_"+str(w_size) + "_sigma_" + str(sigma) +
                              '_padding_' + padding + '.png')

    return im_cart


def gauss_filter(im, padding, w_size, sigma, save):
    """
    Naive Gaussian Filter with 1D kernel on spatial domain given padding and window size of the filter
    Parameters
    ----------
    im: ndarray
        The 2D image array to be filtered
    padding: string
        border padding: "constant", "wrap", "reflect"
    w_size
        window size of the Gaussian Filter
    sigma:
        standart dev. of the Gaussian kernel
    save:
        If true, save intermediary output during the transformation process
    """
    x = np.arange(w_size)
    g = np.exp(-0.5*((x-np.floor(w_size/2.))/sigma)**2)  # gaussian
    g /= g.sum()

    im_padded = np.pad(im, (((w_size - 1)/2, (w_size - 1)/2), (0, 0)), mode=padding)

    # naive Gaussian filtering using column filter
    out = np.zeros(im.shape)
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            for j in range(w_size):
                out[x, y] += im_padded[x+j, y] * g[w_size-j-1]

    if save:
        msc.imsave('gauss1D_col.png', np.transpose( g.reshape(1, w_size)))
        write_intensity_image(out, 'polar_image_filtered_w_size_' + str(w_size) + "_sigma_" + str(sigma) +
                              '_padding_' + padding + '.png')
    return out


if __name__ == '__main__':
    w_size = 11
    padding = "wrap"
    sigma = [10, 20]
    image = read_intensity_image('bauckhage.jpg')       # read image in grayscale

    for i in range(len(sigma)):
        im_polar = polar_transform(image, True)              # polar transform
        im_filtered = gauss_filter(im_polar, padding, w_size, sigma[i], True) # smoothing with Gaussian filter
        cart_transform(im_filtered, w_size, sigma[i], padding, True)  # cartesian transformation

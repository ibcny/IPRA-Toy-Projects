import numpy.fft as fft
import numpy as np
import scipy.misc as msc
import matplotlib.pyplot as plt
import os

img = "clock"

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
    msc.toimage(f).save(file)


def band_pass_filter(image, rMin, rMax, saveImg):
    """
    Apply band pass filter to the given image in the Fourier domain.
    Parameters
    ----------
    image : ndarray
        2-D grayscale image.
    rMin : scalar
        lower bound of the frequencies to be passed.
    rMax : scalar
        upper bound of the frequencies to be passed.
    saveImg :
        if true, save transformed and filtered images.
    Returns
    -------
        None
    """
    G = fft.fft2(image)           # apply fourier transform
    beforeShiftG = G.copy();
    G = fft.fftshift(G)

    width = G.shape[0]
    height = G.shape[1]
    # We retain G, because we want to display it later.
    copyOfG = np.copy(G)

    # Mask the pixels from the transformed & shifted image which have greater distance than rMax and smaler
    # distance than rMin
    width, height = copyOfG.shape[0], copyOfG.shape[1]
    center = (width / 2, height / 2)
    y, x = np.ogrid[-center[1]:center[1], -center[0]:center[0]]  # open grid used to create the mask: -128:127
    dist = x ** 2 + y ** 2  # distance^2 matrix to the center
    mask = (dist > rMax ** 2) | (dist < rMin ** 2)  # mask to be used as a boolean index
    copyOfG[mask] = 0.

    # Perform inverse fourier transform on filtered image
    gPrime = fft.ifft2(fft.ifftshift(copyOfG))

    # save filtered image
    folderName = "{}, rmin = {}, rmax = {}".format(img, rMin, rMax)
    rMinMax = "rmin = {}, rmax = {}".format(rMin, rMax)
    if not os.path.exists(folderName):
        os.mkdir(folderName)

    # save images
    if saveImg:
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title(rMinMax)
        plt.imshow(np.log(np.abs(copyOfG)), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(np.abs(gPrime), cmap='gray')
        plt.savefig((folderName + "/1.3 output full.png"), transparent=True)

    write_intensity_image(np.log(np.abs(G)), img + " 1.3 output log(abs(G)).png")
    write_intensity_image(np.log(np.abs(beforeShiftG)), img + " 1.3 output before fftshift log(abs(G)).png")

if __name__ == '__main__':
    ff = read_intensity_image(img+".jpg")              # read image in grayscale
    band_pass_filter(ff, 20, 80, True)


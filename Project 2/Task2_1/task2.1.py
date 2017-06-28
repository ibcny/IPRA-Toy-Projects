import numpy as np
import scipy.misc as msc
import scipy.ndimage as img
import numpy.fft as fft
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib import pylab
import time

# z-value corresponding to 0.01 error
z_value = 2.575
test = False

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


def write_intensity_image(arr, file):
    """
    Saves the image under the given file name or file object.
    Parameters
    ----------
    file : str
        The file name or file object in which the image array to be written.
    f : ndarray
        The 2D image array to be written to the given file.
    """
    msc.imsave(file, arr)


def gauss_filter_spat(im, padding, w_size, save = False):
    """
    Gaussian Filter image on spatial domain given padding and window size of the filter
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
    g = np.exp(-0.5*((x - np.floor(w_size/2.))/sigma)**2)  # gaussian
    G = np.outer(g, g)
    G /= G.sum()
    if save:
        write_intensity_image(G, 'gauss2D.png' )
    im_padded = np.pad(im, (w_size - 1)/2, mode=padding)

    # border padding
    if save:
        write_intensity_image(im_padded, 'image_padded_'+padding+'.png')

    # naive Gaussian filtering
    out = np.zeros(im.shape)
    for x in range (im.shape[0]):
        for y in range(im.shape[1]):
            for i in range(w_size):
                for j in range(w_size):
                    out[x,y] += im_padded[x+i, y+j] * G[w_size-i-1, w_size-j-1]

    if save:
        write_intensity_image(out, 'image_2d_filtered_w_size_'+str(w_size)+'_padding_'+padding+'.png')

def gauss_filter_sep(im, padding, w_size, save = False):
    """
    Gaussian Filter with separated kernel on spatial domain given padding and window size of the filter
    Parameters
    ----------
    im: ndarray
        The 2D image array to be filtered
    padding: string
        border padding: "constant", "wrap", "reflect"
    w_size
        window size of the Gaussian Filter
    """
    sigma = (w_size - 1.0) / (2.0 * z_value)
    x = np.arange(w_size)
    g = np.exp(-0.5*((x-np.floor(w_size/2.))/sigma)**2)  # gaussian
    g /= g.sum()
    if save:
        write_intensity_image( g.reshape(1, w_size), 'gauss1D_row.png')
        write_intensity_image(np.transpose( g.reshape(1, w_size)), 'gauss1D_col.png')

    im_padded = np.pad(im, (w_size - 1)/2, mode=padding)

    # step 1: naive Gaussian filtering using row filter
    tmp_out = np.zeros((im.shape[0], im_padded.shape[1]))
    for x in range (im.shape[0]):
        for y in range(im_padded.shape[1]):
            for i in range(w_size):
                tmp_out[x,y] += im_padded[x+i, y] * g[w_size-i-1 ]

    # step 2: naive Gaussian filtering using column filter
    g_t = np.transpose(g)
    out = np.zeros(im.shape)
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            for j in range(w_size):
                out[x, y] += tmp_out[x, y+j] * g_t[w_size-j-1]

    if save:
        write_intensity_image(out, 'image_1d_filtered_w_size_' + str(w_size) + '_padding_' + padding + '.png')

def gauss_filter_fourr(im, w_size, save = False):
    """
    Gaussian Filter the image in frequency domain given padding and window size of the filter
    Parameters
    ----------
    im: ndarray
        The 2D image array to be filtered
    w_size
        window size of the Gaussian Filter
    """
    sigma = (w_size-1.0) / (2.0*z_value)
    padding = "constant"
    x = np.arange(w_size)
    g = np.exp(-0.5*((x - np.floor(w_size/2.))/sigma)**2)  # gaussian
    G = np.outer(g, g)
    G /= G.sum()

    m = im.shape[0]
    n = im.shape[1]
    # zero-pad the kernel up to (m+w_size-1)x(n+w_size-1)
    filter_padded = np.pad(G, (((int)(np.floor((m-1)/2.)), (int)(np.ceil((m-1)/2.))),
                               ((int)(np.floor((n-1)/2.)), (int)(np.ceil((n-1)/2.)) )), mode=padding)

    # zero-pad the image up to (m+w_size-1)x(n+w_size-1)
    im_padded = np.pad(im, (w_size - 1)/2, mode=padding)

    G_fft = fft.fft2(fft.ifftshift(filter_padded))           # apply fourier transform to padded kernel
    F_fft = fft.fft2(im_padded)                              # apply fourier transform to image
    convolved = np.multiply(F_fft, G_fft)
    filtered = fft.ifft2(convolved)
    #remove padding after inverse fourier transform
    filtered = filtered[w_size/2-1:filtered.shape[0]-w_size/2-1, w_size/2-1:filtered.shape[1]-w_size/2-1]

    if save:
        write_intensity_image(fft.ifftshift(filter_padded), 'gauss2D_padded.png')
        write_intensity_image((np.abs(fft.fftshift(G_fft))), 'fourier_gauss2D_padded_abs.png')
        write_intensity_image(np.log(np.abs(fft.fftshift(F_fft))), 'fourier_image2D_log_abs.png')
        write_intensity_image(np.log(np.abs(convolved)), 'fourier_image2D_convolved_log_abs.png')
        write_intensity_image(np.abs(filtered), 'fourier_image_2d_filtered_w_size_'+str(w_size)+'_padding_'+padding+'.png')

if __name__ == '__main__':
    ff = read_intensity_image('bauckhage.jpg')  # read image in grayscale
    gauss_filter_spat(ff, "constant", 11, True)   # task 1:  filter image on spatial domain
    gauss_filter_sep(ff, "wrap", 11, True)        # task 2: filter image on spatial domain using separated Gaussian filter
    gauss_filter_fourr(ff, 11, True)              # task 3:  filter image on frequency domain

    # task 4: test
    if test:
        # test with various filters
        filter_sizes = np.asarray([3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
        spatial_times = list()
        np.asarray(spatial_times)
        separated_times = list()
        fourier_times = list()
        
        for i in range(filter_sizes.shape[0]):
            filter_sz = filter_sizes[i]
            print "size: " + str(filter_sz)

            total_time_spatial = total_time_separated = total_time_fourier = 0
            # We test for 10 times
            for trial in range(10):
                spatial_start = time.time()
                # Test for Spatial naive convolution
                gauss_filter_spat(ff, 'reflect', filter_sz, False)
                total_time_spatial += time.time() - spatial_start
                separated_start = time.time()

                # Test for Separated convolution
                gauss_filter_sep(ff, 'reflect', filter_sz, False)
                total_time_separated += time.time() - separated_start
                fourier_start = time.time()

                # Test for Separated convolution
                gauss_filter_fourr(ff, filter_sz, False)
                total_time_fourier += time.time() - fourier_start

            spatial_times.append(total_time_spatial / 10.)
            separated_times.append(total_time_separated / 10.)
            fourier_times.append(total_time_fourier / 10.)

        # print spatial_times
        # print separated_times
        print fourier_times

        # spatial_times = np.asarray(spatial_times)
        # separated_times = np.asarray(separated_times)
        fourier_times = np.asarray(fourier_times)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        handle1, = ax.plot(filter_sizes, fourier_times, 'g^', label='Fourier')
        handle2, = ax.plot(filter_sizes, spatial_times, 'r--', label='Spatial')
        handle3, = ax.plot(filter_sizes, separated_times, 'bs', label='Separated')
        ax.set_yscale('log')
        ax.set_xlabel('kernel size')
        ax.set_ylabel('log(ms)')
        ax.legend([handle1, handle2, handle3], loc='upper left')
        fig.savefig('plot2.1.4.png')

        # spatial_times = [0.9056935787200928, 2.4088427305221556, 4.605582904815674, 7.472419619560242, 11.551127910614014, 15.665853643417359, 21.545386171340944, 30.06045446395874, 35.85298101902008, 39.69724643230438]
        # separated_times = [0.4493666887283325, 0.7143961191177368, 0.9700865268707275, 1.2502843141555786, 1.5750965833663941, 1.8585208177566528, 2.1971273899078367, 2.596834111213684, 2.772663974761963, 2.9016735553741455]
        # fourier_times = [0.025757765769958495, 0.01479649543762207, 0.0790729999542
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
import scipy.misc as misc
import time
from scipy.ndimage.filters import gaussian_filter

sigmas = [1, 3, 5]
for sigma in sigmas:

    imagePath = 'clock'
    f = ndimage.imread(imagePath + ".jpg")

    startingTime = time.time()

    # Set the parameters
    alpha1, alpha2 = 1.68, -0.6803
    beta1, beta2 = 3.7350, -0.2598
    gamma1, gamma2 = 1.7830, 1.7230
    omega1, omega2 = 0.6318, 1.9970

    # Calculate a^+, a^-, b^+ and b^-
    aplus = np.zeros(9, dtype=np.float)
    aplus[0] = alpha1 + alpha2

    aplus[1] = np.exp(-gamma2 / sigma) * (beta2 * np.sin(omega2 / sigma) - (alpha2 + 2. * alpha1) * np.cos(omega2 / sigma))
    aplus[1] += np.exp(-gamma1 / sigma) * (beta1 * np.sin(omega1 / sigma) - (2. * alpha2 + alpha1) * np.cos(omega1 / sigma))

    aplus[2] = (alpha1 + alpha2) * np.cos(omega2 / sigma) * np.cos(omega1 / sigma)
    aplus[2] -= np.cos(omega2 / sigma) * beta1 * np.sin(omega1 / sigma)
    aplus[2] -= np.cos(omega1 / sigma) * beta2 * np.sin(omega2 / sigma)
    aplus[2] *= 2. * np.exp((-gamma2 - gamma1) / sigma)
    aplus[2] += alpha2 * np.exp(-2. * gamma1 / sigma) + alpha1 * np.exp(-2 * gamma2 / sigma)

    aplus[3] = np.exp(-(gamma2 + 2. * gamma1) / sigma) * (beta2 * np.sin(omega2 / sigma) - alpha2 * np.cos(omega2 / sigma))
    aplus[3] += np.exp(-(gamma1 + 2. * gamma2) / sigma) * (beta1 * np.sin(omega1 / sigma) - alpha1 * np.cos(omega1 / sigma))

    bplus = np.zeros(9, dtype=np.float)

    bplus[0] = 0.

    bplus[1] = -2. * np.exp(-1. * gamma2 / sigma) * np.cos(omega2 / sigma)
    bplus[1] += -2. * np.exp(-1. * gamma1 / sigma) * np.cos(omega1 / sigma)

    bplus[2] = 4. * np.cos(omega2 / sigma) * np.cos(omega1 / sigma) * np.exp(-(gamma1 + 1. * gamma2) / sigma)
    bplus[2] += np.exp(-2. * gamma2 / sigma) + np.exp(-2. * gamma1 / sigma)

    bplus[3] = -2. * np.cos(omega1 / sigma) * np.exp(-(gamma1 + 2. * gamma2) / sigma)
    bplus[3] += -2. * np.cos(omega2 / sigma) * np.exp(-(gamma2 + gamma1 * 2.) / sigma)

    bplus[4] = np.exp(-(2. * gamma1 + 2. * gamma2) / sigma)

    aminus = np.zeros(9, dtype=np.float)

    aminus[1] = aplus[1] - bplus[1] * aplus[0]
    aminus[2] = aplus[2] - bplus[2] * aplus[0]
    aminus[3] = aplus[3] - bplus[3] * aplus[0]
    aminus[4] = - bplus[4] * aplus[0]

    # empty padding to be used in future for efficiency
    ze = np.zeros(4, dtype=np.float)


    def smoothSingleLine(line, width):
        # Finds y[n] for a single row

        # Create y^+ and y^- and Append zeros to avoid dimension errors so that - indices go to start of the array.
        yplus = np.zeros((width + 4), dtype=float)
        yminus = np.zeros((width + 4), dtype=float)

        # Append zeros to avoid dimension errors so that - indices go to start of an array.
        #line = np.append(line, ze)

        line = np.pad(line, (0, width*1), mode='reflect')

        # Apply the the functions y^+ and y^-
        for n in range(width):
            for m in range(0, 4):
                yplus[n] += aplus[m] * line[n - m]
            for m in range(1, 5):
                yplus[n] -= bplus[m] * yplus[n - m]

        for n in reversed(range(width)):
                yminus[n] += aminus[m] * line[n + m] - bplus[m] * yminus[n + m]

        # calculate  y from y^+ and y^-.
        return (yminus + yplus) / (sigma * np.sqrt(2 * np.pi))

    def smooth1D(image):

        # Finds y[n] for all the rows
        height, width = image.shape[0], image.shape[1]

        # Create smoothing, pad 0s to end of it to avoid errors
        smoothed = np.zeros((height, width + 4), dtype=np.float)

        # Apply smoothing to all lines
        for i in range(height):
            # Smooth a line
            smoothline = smoothSingleLine((image[i]), width)

            # Save the smoothed line
            smoothed[i, :] = smoothline

        # Return the 1D smoothed line, exclude 0 paddings.
        return smoothed[0:height, 0:width]

    # Apply transformation on one axis
    im = smooth1D(f)

    # Transpose so we can apply to the other axis
    im = np.transpose(im)
    im = smooth1D(im)

    # Correct the transposed image
    im = np.transpose(im)

    endTime = time.time()
    print "Sigma"+ str(sigma)+" took {0} milliseconds".format(endTime-startingTime)
    misc.imsave(imagePath + "_sigma_"+str(sigma)+"_filtered.png", im)
    m = 7

    start = time.time()
    c = np.ones((m,m),dtype=np.float)
    c[int(np.floor(m/2.))] = 1.

    mg = ndimage.gaussian_filter(f, sigma=(sigma), order=0)

    endTime = time.time()
    print "Sigma"+str(sigma)+", native convolution took {0} milliseconds".format(endTime - start)
    misc.imsave(imagePath+"_sigma_"+str(sigma)+"_nativefiltered.png", mg)

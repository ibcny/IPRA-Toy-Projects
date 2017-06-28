# author: Can Yuce
# e-mail: cany@uni-bonn.de

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from random import randint


def save_fourier_transform(offset=0., amplitude=1., frequency=1, phase=0., n=512):
    """
    Compute the one-dimensional discrete Fourier Transform of a sine wave given the parameters
    and save the plot of the function and its Fourier transform to the file.
    sine wave is:
            f = offset + amplitude * np.sin(frequency * x + phase)
    Parameters
    ----------
    offset : float
        vertical offset of the signal
    amplitude: float
        amplitude of the signal
    frequency: scalar
        number of angular oscillations per unit time.
    phase:
        phase angle of the signal in radians
    Returns
    -------
        None
    """
    x = np.linspace(0, 2*np.pi, n)
    f = offset + amplitude * np.sin(frequency * x + phase)   # evaluate function

    plt.subplot(2, 2, 1)
    plt.title("o={:.2f},a={:.2f},f={},p={:.2f}".format(offset, amplitude, frequency, np.rad2deg(phase)))
    plt.plot(x, f, 'k-')                                     # plot the function

    F = fft.fft(f)                                           # evaluate Fourier transform
    w = fft.fftfreq(n)
    plt.subplot(2, 2, 2)
    plt.title("F")
    plt.plot(w, F, "k-")                                     # plot frequencies

    plt.subplot(2, 2, 3)
    plt.title("abs(F)")
    plt.plot(w, np.abs(F), "k-")                             # plot modulus of the frequencies

    plt.subplot(2, 2, 4)
    plt.title("log(abs(F))")
    plt.plot(w, np.log(np.abs(F)), "k-")                     # plot log(modulus) of the frequencies

    # save the figure to the file.
    plt.savefig("Offset={:.2f}_Amplitude={:.2f}_Frequency={}_Phase={:.2f}.jpg".format(offset, amplitude, frequency,
                                                                                      np.rad2deg(phase)))
    plt.close()


if __name__ == '__main__':
    save_fourier_transform()                  # call with offset=0., amplitude=1., frequency=1,  phase=0., n=512
    save_fourier_transform(1., 2., 16, np.pi) # call with offset=1., amplitude=2., frequency=16, phase=np.pi., n=512

    _offset = [2., 4., 8.]
    _amplitude = [2.,-2.]
    _frequency = [1, 8, 64]
    _phase = [0., np.pi, np.pi * 2]

    # Calculate Fourier transforms of the sine waves corresponding to different offset values
    for o in _offset:
        save_fourier_transform( o,
                                _amplitude[randint(0, len(_amplitude)-1)], # randomize other parameters
                                _frequency[randint(0, len(_frequency)-1)],
                                    _phase[randint(0, len(_phase)    -1)],
                                )

    # Calculate Fourier transforms of the sine waves corresponding to different amplitude values
    for a in _amplitude:
        save_fourier_transform( _offset[randint(0, len(_offset)-1)],       # randomize other parameters
                                a,
                                _frequency[randint(0, len(_frequency)-1)],
                                    _phase[randint(0, len(_phase)    -1)],
                                )

    # Calculate Fourier transforms of the sine waves corresponding to different frequency values
    for f in _frequency:
        save_fourier_transform( _offset[randint(0, len(_offset)-1)],       # randomize other parameters
                                _amplitude[randint(0, len(_amplitude)-1)],
                                f,
                                _phase[randint(0, len(_phase)    -1)],
                                )

    # Calculate Fourier transform of the sine waves corresponding to different phase values
    for p in _phase:
        save_fourier_transform( _offset[randint(0, len(_offset) - 1)],      # randomize other parameters
                                _amplitude[randint(0, len(_amplitude) - 1)],
                                _frequency[randint(0, len(_frequency) - 1)],
                                p )
import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt

def write_intensity_image(f, filename):
    plt.imshow(f, cmap='gray')
    plt.savefig(filename)

g = plt.imread('bauckhage.jpg')
h = plt.imread('clock.jpg')

if len(g.shape) == 3:
    g = np.mean(g, axis=2)

if len(h.shape) == 3:
    h = np.mean(h, axis=2)

# Fourier transform of image g
G = fft.fft2(g)

# Fourier transform of image f
H = fft.fft2(h)

# sum the squares of real and imaginary parts, then, take their square root to get the modulus.
magnitudeG = np.abs(G)

# evaluate the phase of entries in other image
phaseH = np.arctan2(H.imag, H.real)

# Calculate the real values for every pixel
newReal = magnitudeG * np.cos(phaseH)

# Calculate the imaginary part for every pixel
newImaginary = magnitudeG * np.sin(phaseH) * 1j

# Put the real and imaginary part together, find K
K = newReal + newImaginary

# Inverse Fourier transform K
inverseK = fft.ifft2(K)

write_intensity_image(np.log(np.abs(fft.fftshift(magnitudeG))), 'magnitude of G.png')
write_intensity_image(np.log((np.abs(fft.fftshift(phaseH)))), 'phase of H.png')
write_intensity_image(np.log((np.abs(fft.fftshift(newReal)))), 'real part of F(result).png')
write_intensity_image(np.log((np.abs(fft.fftshift(newImaginary)))), 'imaginary of F(result).png')
write_intensity_image(np.log((np.abs(fft.fftshift(K)))), 'log(abs(Fourier(result))).png')
write_intensity_image((np.abs(inverseK)), 'inverse Fourier transform of result.png')

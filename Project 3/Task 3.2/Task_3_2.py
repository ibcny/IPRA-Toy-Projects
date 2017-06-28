import numpy as np
import scipy.misc as misc
from scipy import ndimage

print (range(1, 2))
imageName = "bauckhage"


def warp_image(img, frequency, amplitude, phase):
    width, height = img.shape[0], img.shape[1]

    mesh = np.meshgrid(range(width), range(height + 2 * amplitude))
    mesh_y = amplitude * np.sin(np.pi * frequency * (mesh[0]) + phase) + mesh[1] - amplitude
    out_img = ndimage.map_coordinates(img, [width - mesh[0], height-mesh_y])

    return out_img


inputImage = misc.imread(imageName + ".jpg")

# Wave parameters for axis x
f, amp, ph = .005, 0, -0.2  # Since amplitude is zero, basically ineffective

# Wave parameters for axis y
secondFreq, secondAmp, secondPh = 0.005, 64, -.02

# Warp a single axis first
transform = warp_image(inputImage, f, amp, ph)
# Warp the other axis too
transform = warp_image(transform, secondFreq, secondAmp, secondPh)

misc.imsave(imageName + "_transformed_example_1.png", transform)


# Another example
f, amp, ph = .008, 0, .4  # Difference from previous one is the frequency is higher, must be put on slides.
secondFreq, secondAmp, secondPh = 0.008, 64, .4  # Since amplitude is zero, basically ineffective

transform = warp_image(inputImage, f, amp, ph)
transform = warp_image(transform, secondFreq, secondAmp, secondPh)
misc.imsave(imageName + "_transformed_example_2.png", transform)


# Another example
f, amp, ph = .02, 16, .4  # Difference from previous one is the frequency is higher, must be put on slides.
secondFreq, secondAmp, secondPh = 0.02, 16, .4  # Since amplitude is zero, basically ineffective
transform = warp_image(inputImage, f, amp, ph)
transform = warp_image(transform, secondFreq, secondAmp, secondPh)
misc.imsave(imageName + "_transformed_example_3.png", transform)

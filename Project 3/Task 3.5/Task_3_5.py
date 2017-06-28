import numpy as np
import scipy.misc as misc
from scipy import ndimage

imageName = "isle"
imageToMapToName = "isle"

imageToMapTo = misc.imread(imageToMapToName + ".jpg")
image = misc.imread(imageName + ".jpg")
image = image.transpose()

# We need to make the input image into a square first
mx = max(image.shape)
image = misc.imresize(image, (mx, mx))

height, width = image.shape[0], image.shape[1]
isleHeight, isleWidth = imageToMapTo.shape[0], imageToMapTo.shape[1]


destinationPoints = np.array([

    [56, 215],
    [10, 365],
    [296, 364],
    [258, 218]
])

# x, y
dx, dy = destinationPoints[:, 1], destinationPoints[:, 0]

# X, Y
sx = [0,  width - 1, width - 1,0]
sy = [0, 0, height - 1, height - 1]

# sx, sy, dx, dy = sy, sx, dy, dx
# sx, sy, dx, dy = dx, dy, sx, sy

transformationMatrix = np.array([
    [dx[0], dy[0], 1, 0, 0, 0, - dx[0] * sx[0], - dy[0] * sx[0]],
    [dx[1], dy[1], 1, 0, 0, 0, - dx[1] * sx[1], - dy[1] * sx[1]],
    [dx[2], dy[2], 1, 0, 0, 0, - dx[2] * sx[2], - dy[2] * sx[2]],
    [dx[3], dy[2], 1, 0, 0, 0, - dx[3] * sx[3], - dy[3] * sx[3]],
    [0, 0, 0, dx[0], dy[0], 1, - dx[0] * sy[0], - dy[0] * sy[0]],
    [0, 0, 0, dx[1], dy[1], 1, - dx[1] * sy[1], - dy[1] * sy[1]],
    [0, 0, 0, dx[2], dy[2], 1, - dx[2] * sy[2], - dy[2] * sy[2]],
    [0, 0, 0, dx[3], dy[3], 1, - dx[3] * sy[3], - dy[3] * sy[3]],
])

# transformationMatrix = np.transpose(transformationMatrix)
print np.linalg.inv(transformationMatrix)

sourceLong = [
    [sx[0]],
    [sx[1]],
    [sx[2]],
    [sx[3]],
    [sy[0]],
    [sy[1]],
    [sy[2]],
    [sy[3]]
]

inverseTransformationmatrix = np.linalg.inv(transformationMatrix)
# inverseTransformationmatrix = np.transpose(inverseTransformationmatrix)

# Find coefficients
abcdefgh = np.dot(inverseTransformationmatrix, sourceLong)

[[a],
 [b],
 [c],
 [d],
 [e],
 [f],
 [g],
 [h]] = abcdefgh

mesh = np.meshgrid(range(isleWidth), range(isleHeight))

# Apply transformation parameters to every pixel in destination image
newx = (mesh[0] * a + mesh[1] * b + c)
newx = np.divide(newx, (mesh[0] * g + mesh[1] * h + 1))
newy = np.divide((mesh[0] * d + mesh[1] * e + f), (mesh[0] * g + mesh[1] * h + 1))

# Perform interpolation
out = ndimage.map_coordinates(image, [newx, newy])

# Paste the projected image onto new image
imageToMapTo[out>0] = 0

out += imageToMapTo

misc.imsave("mapped_with_{0}.png".format(imageName), out)


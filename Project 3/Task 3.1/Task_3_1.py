import numpy as np
import matplotlib.pyplot as plt

n = 20   # number of data points that will be used to estimate weights
m = 2000 # number of data points to be interpolated

x = np.arange(n) + np.random.randn(n) * 0.2
y = np.random.rand(n) * 2
xs = np.linspace(0, n, m)

def rb_interp(sigma):
    """
    Interpolate a function using radial basis functions
    Parameters
    ----------
    sigma :
        standard deviation of the radial basis function
    Returns
    -------
    f : ndarray
        array in which the interpolated function is kept
    """
    	
    x_j, x_i = np.meshgrid(x, x);
    phi = np.exp(-((x_i-x_j)**2)/(2*(sigma**2)))
    w = (np.linalg.inv(phi)).dot(y);

    xs_m = xs.reshape([m, 1]);
    xs_m = np.tile(xs_m, [1, n]);
    x_m  = x.reshape([1,n]);
    x_m  = np.tile(x_m, [m, 1]);

    phi = np.exp(-((xs_m-x_m)**2)/(2*sigma**2))
    f = np.dot(phi, w)
    return f

if __name__ == '__main__':
    sigma = [0.2,0.5,1,1.5,2,3,5]

    for i in range(len(sigma)):
        fig = plt.figure()
        f = rb_interp(sigma[i])
        plt.title("sigma={:.1f}".format(sigma[i]))

        handle1, = plt.plot(xs, f, 'r-', label='Interp. Func.')  # plot the function
        handle2, = plt.plot(x, y, 'g^', label='Samples')
        plt.legend([handle1, handle2], loc='upper left')

        # save the figure to the file.
        fig.savefig("sigma={:.1f}.jpg".format(sigma[i]))
        plt.close()



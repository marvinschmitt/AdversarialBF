import tensorflow as tf
import numpy as np
from functools import partial


def kl_latent_space(z, log_det_J):
    """ Computes the Kullback-Leibler divergence (Maximum Likelihood Loss) between true and approximate
    posterior using simulated data and parameters.
    """

    loss = tf.reduce_mean(0.5 * tf.square(tf.norm(z, axis=-1)) - log_det_J)
    return loss

def maximum_mean_discrepancy(source_samples, target_samples, minimum=0.):
    """ This Maximum Mean Discrepancy (MMD) loss is calculated with a number of different Gaussian kernels.

    """

    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(_gaussian_kernel_matrix, sigmas=sigmas)
    loss_value = _mmd_kernel(source_samples, target_samples, kernel=gaussian_kernel)
    loss_value = tf.maximum(minimum, loss_value) 
    return loss_value

def _gaussian_kernel_matrix(x, y, sigmas):
    """ Computes a Gaussian Radial Basis Kernel between the samples of x and y.

    We create a sum of multiple gaussian kernels each having a width :math:`\sigma_i`.

    Parameters
    ----------
    x :  tf.Tensor of shape (M, num_features)
    y :  tf.Tensor of shape (N, num_features)
    sigmas : list(float)
        List which denotes the widths of each of the gaussians in the kernel.

    Returns
    -------
    kernel: tf.Tensor
        RBF kernel of shape [num_samples{x}, num_samples{y}]
    """
    def norm(v):
        return tf.reduce_sum(tf.square(v), 1)
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    kernel = tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))
    return kernel

def _mmd_kernel(x, y, kernel=_gaussian_kernel_matrix):
    """ Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.

    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of the distributions of x and y.

    Parameters
    ----------
    x      : tf.Tensor of shape (num_samples, num_features)
    y      : tf.Tensor of shape (num_samples, num_features)
    kernel : callable, default: _gaussian_kernel_matrix
        A function which computes the kernel in MMD.

    Returns
    -------
    loss : tf.Tensor
        squared maximum mean discrepancy loss, shape (,)
    """

    loss = tf.reduce_mean(kernel(x, x))  # lint error: sigmas unfilled
    loss += tf.reduce_mean(kernel(y, y))  # lint error: sigmas unfilled
    loss -= 2 * tf.reduce_mean(kernel(x, y))  # lint error: sigmas unfilled
    return loss

def mmd_kl_loss(network, *args, mmd_weight=1.0):
    """KL loss in latent z space, MMD loss in summary space."""
    
    # Apply net and unpack 
    x_sum, out = network(*args, return_summary=True)
    z, log_det_J = out
    
    # Apply MMD loss to x_sum
    z_normal = tf.random.normal(x_sum.shape)
    mmd_loss = maximum_mean_discrepancy(x_sum, z_normal)
    
    # Apply KL loss for inference net
    kl_loss = kl_latent_space(z, log_det_J)
    
    # Sum and return losses
    return kl_loss + mmd_weight * mmd_loss






####

def mahalanobis_distance_2D_1D(data, ref, cov):
    n = data.shape[0]
    mahalanobis_distances = [scipy.spatial.distance.mahalanobis(data[i], ref, cov) for i in range(n)]
    return np.array(mahalanobis_distances)

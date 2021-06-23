
# Goal: Put together a basic test case to show everything works.

# Written by Joseph Kump, josek97@vt.edu
# Last modified 3/24/2021

import h5py
import numpy as np
import pywt
import time

from wavelet_xcorr.support_code.generate_all_weights_functions import generate_stacked_weight_matrices
from wavelet_xcorr.dense_code.calculate_nlevel_xcorrs          import calculate_nlevel_xcorrs
from wavelet_xcorr.sparse_code.calculate_nlevel_xcorrs_sparse  import calculate_nlevel_xcorrs_sparse

""" Now that we have a correct compute_all_diagonals sparse implementation, we'll try plugging it into our
	wavelet domain xcorr algorithm and evaluating.

	After that, we'll incorporate sparsity into the beginning and end terms. The easiest way to do that is
	probably by just using numpy searchsorted and indexing to correctly form the begin and end data matrices,
	and then just use normal dense operations. (We can try sparse matrix operations there if it's too slow).

"""

wavelet = "db3"
level   = 3
mode    = "periodic"

# Here we'll test our timing and accuracy on an autocorrelation:

x           = np.arange(0, 30000)
signal1     = np.zeros(30000)
frequencies = [2, 4, 6, 8, 30, 120]
amplitudes  = [200, 400, 400, 200, 1000]

for i,j in zip(frequencies, amplitudes):
    
    signal1 += j * np.sin(i * .004 * np.pi * x)
    
noise_mean = 0
noise_std  = 2000
signal1   += np.random.normal(noise_mean, noise_std, len(signal1))

signal2 = signal1[:15000]

coeffs1 = pywt.wavedec(signal1, wavelet, level=level, mode=mode)
coeffs2 = pywt.wavedec(signal2, wavelet, level=level, mode=mode)
lagmax  = 1000

# First, we need to threshold our wavelet functions and create their sparse representations.
# For now, we're thresholding each level separately.
threshold      = 90
thresh_coeffs1 = []
thresh_coeffs2 = []
sparse_coeffs1 = []
sparse_coeffs2 = []
for i in range(len(coeffs1)):
	thresh_coeffs1.append(pywt.threshold(coeffs1[i], np.percentile(np.abs(coeffs1[i]), threshold), mode='soft'))
	thresh_coeffs2.append(pywt.threshold(coeffs2[i], np.percentile(np.abs(coeffs1[i]), threshold), mode='soft'))

	sparse_coeffs1.append((np.nonzero(thresh_coeffs1[i])[0], thresh_coeffs1[i][np.nonzero(thresh_coeffs1[i])[0]], len(thresh_coeffs1[i])))
	sparse_coeffs2.append((np.nonzero(thresh_coeffs2[i])[0], thresh_coeffs2[i][np.nonzero(thresh_coeffs2[i])[0]], len(thresh_coeffs2[i])))


# Let's just check and make sure our weight matrices are reorganized properly:
stacked_weight_matrices, stacked_mixed_weight_matrices, mixed_endpoint_indices = generate_stacked_weight_matrices(wavelet, level)

dense_xcorrs  = calculate_nlevel_xcorrs(stacked_weight_matrices, stacked_mixed_weight_matrices,
                                        mixed_endpoint_indices, thresh_coeffs1, thresh_coeffs2, lagmax)
sparse_xcorrs = calculate_nlevel_xcorrs_sparse(stacked_weight_matrices, stacked_mixed_weight_matrices,
                                               mixed_endpoint_indices, sparse_coeffs1, sparse_coeffs2, lagmax)

relative_errors = np.abs(sparse_xcorrs - dense_xcorrs) / (np.abs(dense_xcorrs) + 1)


print(np.max(relative_errors))

print(np.mean(relative_errors), np.median(relative_errors))

#print(np.mean(sparse_xcorrs))

samples = 100
dense_times = np.zeros(100)
sparse_times = np.zeros(100)

for i in range(samples):
	t0 = time.time()
	dense_xcorrs = calculate_nlevel_xcorrs(stacked_weight_matrices, stacked_mixed_weight_matrices,
                                           mixed_endpoint_indices, thresh_coeffs1, thresh_coeffs2, lagmax)
	t1 = time.time()
	dense_times[i] = t1 - t0

for i in range(samples):
	t0 = time.time()
	sparse_xcorrs = calculate_nlevel_xcorrs_sparse(stacked_weight_matrices, stacked_mixed_weight_matrices,
                                                   mixed_endpoint_indices, sparse_coeffs1, sparse_coeffs2, lagmax)
	t1 = time.time()
	sparse_times[i] = t1 - t0

print(np.mean(dense_times), np.mean(sparse_times))


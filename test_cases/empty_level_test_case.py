# In this case, we're zeroing out an entire level of wavelet coefficients.

# Written by Joseph Kump, josek97@vt.edu
# Last modified 6/26/2021

import h5py
import numpy as np
import pywt
import time
import matplotlib.pyplot as plt

from wavelet_xcorr.support_code.generate_all_weights_functions import generate_stacked_weight_matrices
from wavelet_xcorr.support_code.thresholding_functions         import threshold_coeffs_one_channel, make_sparse_coeffs
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

signal2 = signal1[200:15208]
#signal1 = signal1[:3008]
#signal2 = signal1[200:1704]

coeffs1 = pywt.wavedec(signal1, wavelet, level=level, mode=mode)
coeffs2 = pywt.wavedec(signal2, wavelet, level=level, mode=mode)
lagmax  = 1008

# First, we need to compress our wavelet functions and create their sparse representations.
# We're going to zero out the detail 1 coefficients in both signals.
comp_coeffs1   = [coeffs1[0], coeffs1[1], coeffs1[2], np.zeros(coeffs1[3].shape)]
comp_coeffs2   = [coeffs2[0], coeffs2[1], np.zeros(coeffs2[2].shape), coeffs2[3]]
sparse_coeffs1 = make_sparse_coeffs(comp_coeffs1)
sparse_coeffs2 = make_sparse_coeffs(comp_coeffs2)

# For comparisons, we'll reconstruct the original signal using our compressed coefficients:
comp_signal1 = pywt.waverec(comp_coeffs1, wavelet, mode=mode)
comp_signal2 = pywt.waverec(comp_coeffs2, wavelet, mode=mode)

print([_[0].shape[0] == 0 for _ in sparse_coeffs1])


# Let's just check and make sure our weight matrices are reorganized properly:
stacked_weight_matrices, stacked_mixed_weight_matrices, mixed_endpoint_indices = generate_stacked_weight_matrices(wavelet, level)

time_xcorrs   = np.correlate(comp_signal1[:len(comp_signal2)+lagmax-1], comp_signal2, mode="valid")

dense_xcorrs  = calculate_nlevel_xcorrs(stacked_weight_matrices, stacked_mixed_weight_matrices,
                                        mixed_endpoint_indices, comp_coeffs1, comp_coeffs2, lagmax)

sparse_xcorrs = calculate_nlevel_xcorrs_sparse(stacked_weight_matrices, stacked_mixed_weight_matrices,
                                               mixed_endpoint_indices, sparse_coeffs1, sparse_coeffs2, lagmax)

relative_errors_dense = np.abs(dense_xcorrs - time_xcorrs) / (np.abs(time_xcorrs) + 1)
relative_errors_sparse = np.abs(sparse_xcorrs - time_xcorrs) / (np.abs(time_xcorrs) + 1)

#plt.figure(figsize=(16,6))
#plt.plot(time_xcorrs)
#plt.plot(sparse_xcorrs)
#plt.savefig("xcorrs.pdf", format="pdf")
#plt.show()

print("Dense errors:")
print(np.max(relative_errors_dense), np.mean(relative_errors_dense), np.median(relative_errors_dense))

print("Sparse errors:")
print(np.max(relative_errors_sparse), np.mean(relative_errors_sparse), np.median(relative_errors_sparse))

#print(np.mean(sparse_xcorrs))

samples = 100
dense_times = np.zeros(100)
sparse_times = np.zeros(100)

for i in range(samples):
    t0 = time.time()
    dense_xcorrs = calculate_nlevel_xcorrs(stacked_weight_matrices, stacked_mixed_weight_matrices,
                                           mixed_endpoint_indices, comp_coeffs1, comp_coeffs2, lagmax)
    t1 = time.time()
    dense_times[i] = t1 - t0

for i in range(samples):
    t0 = time.time()
    sparse_xcorrs = calculate_nlevel_xcorrs_sparse(stacked_weight_matrices, stacked_mixed_weight_matrices,
                                                   mixed_endpoint_indices, sparse_coeffs1, sparse_coeffs2, lagmax)
    t1 = time.time()
    sparse_times[i] = t1 - t0

print(np.mean(dense_times), np.mean(sparse_times))


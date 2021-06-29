
# Goal: Put together a basic test case to highlight the wavelet cross-correlation algorithm.

# Written by Joseph Kump, josek97@vt.edu
# Last modified 6/26/2021

import h5py
import numpy as np
import pywt
import time
import matplotlib.pyplot as plt

from scipy.signal import correlate

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
level   = 4
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

coeffs1 = pywt.wavedec(signal1, wavelet, level=level, mode=mode)
coeffs2 = pywt.wavedec(signal2, wavelet, level=level, mode=mode)
lagmax  = 1008

# First, we need to threshold our wavelet functions and create their sparse representations.
# For now, we're thresholding each level separately.
threshold      = 90
thresh_coeffs1 = threshold_coeffs_one_channel(coeffs1, threshold)
thresh_coeffs2 = threshold_coeffs_one_channel(coeffs2, threshold)
sparse_coeffs1 = make_sparse_coeffs(thresh_coeffs1)
sparse_coeffs2 = make_sparse_coeffs(thresh_coeffs2)


# Here we generate our cross-correlation matrices:
stacked_weight_matrices, stacked_mixed_weight_matrices, mixed_endpoint_indices = generate_stacked_weight_matrices(wavelet, level)

# Here we get the original time-domain cross-correlations, as well as thresholded wavelet-domain cross-correlations through the
# dense and sparse methods.
time_xcorrs   = correlate(signal1[:len(signal2)+lagmax-1], signal2, mode="valid")

dense_xcorrs  = calculate_nlevel_xcorrs(stacked_weight_matrices, stacked_mixed_weight_matrices,
                                        mixed_endpoint_indices, thresh_coeffs1, thresh_coeffs2, lagmax)

sparse_xcorrs = calculate_nlevel_xcorrs_sparse(stacked_weight_matrices, stacked_mixed_weight_matrices,
                                               mixed_endpoint_indices, sparse_coeffs1, sparse_coeffs2, lagmax)



dense_relative_errors  = np.abs(dense_xcorrs - time_xcorrs) / (np.abs(time_xcorrs) + 1)
sparse_relative_errors = np.abs(sparse_xcorrs - time_xcorrs) / (np.abs(time_xcorrs) + 1)

plt.figure(figsize=(16,6))
plt.plot(time_xcorrs)
plt.plot(sparse_xcorrs)
plt.title("Wavelet Cross-Correlations")
plt.xlabel("Time-Lag")
plt.ylabel("Amplitude")
plt.legend(["Time-Domain", "Thresholded Wavelet Domain (Sparse)"])
plt.savefig("xcorrs.pdf", format="pdf")
plt.show()


print("Relative errors of dense thresholded cross-correlation (should be somewhat high):")
print(np.max(dense_relative_errors), np.mean(dense_relative_errors), np.median(dense_relative_errors))

print("Relative errors of sparse thresholded cross-correlation (should be somewhat high):")
print(np.max(sparse_relative_errors), np.mean(sparse_relative_errors), np.median(sparse_relative_errors))


# Runtimes
samples = 100
original_times = np.zeros(samples)
dense_times    = np.zeros(samples)
sparse_times   = np.zeros(samples)

for i in range(samples):
    t0 = time.time()
    original_xcorrs = correlate(signal1[:len(signal2)+lagmax-1], signal2, mode="valid")
    t1 = time.time()
    original_times[i] = t1 - t0

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

print("Runtimes: time-domain, dense wavelet-domain, sparse wavelet-domain")
print(np.mean(original_times), np.mean(dense_times), np.mean(sparse_times))


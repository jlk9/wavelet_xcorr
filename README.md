# wavelet_xcorr
Cross-correlations in the wavelet domain.

## What does this library do?

This repository contains a library of functions for performing temporal cross-correlations on wavelet-domain data. Traditionally, cross-correlation analysis on wavelet-compressed data is done by reconstructing the original time-series data or an approximation of it (via an inverse wavelet transform), and then computing the cross-correlations using direct or FFT-based methods. This library eliminates the intermediate step, allowing us to compute the temporal cross-correlations directly using the wavelet coefficients. In addition to the usual benefits of wavelet compression, such as noise reduction and bandpass filtering, our approach reduces the storage requirements for cross-correlation analysis (since we no longer need to do store a time-domain representation of our data), and can improve the speed of analysis in some cases.

A detailed description of the underlying theory and method that motivates this implementation, as well as the benefits, is available in the electronic thesis document here: [https://vtechworks.lib.vt.edu/handle/10919/103864](https://vtechworks.lib.vt.edu/handle/10919/103864).

## Organization

The source code is all located within the `wavelet_xcorr` subdirectory. It is organized into three seperate folders:

1. `dense_code` contains functions for computing wavelet-domain cross-correlations on wavelet coefficients stored in a dense format (in particular, a list of NumPy arrays where each array corresponds to one level of wavelet coefficients, as used in the PyWavelet library). The main function is `calculate_nlevel_xcorrs`, located in the `.py` file of the same name. The rest of the functions are helpers.
2. `sparse_code` contains functions for computing wavelet_domain cross_correlations on wavelet coefficients stored in a sparse format. The dense NumPy arrays used for the `dense_code` implementation are replaced by tuples of (nonzero indices array, nonzero values array, total densely-represented array length). The main function is `calculate_nlevel_xcorrs_sparse`, located in the `.py` file of the same name. The rest of the functions are helpers.
3. `support_code` contains functions for generating the correlation matrices used for wavelet-domain computations in `dense_code` and `sparse_code`. It also contains functions for thresholding wavelet-domain data, converting wavelet-domain data into a sparse format for use with `sparse_code`, and storing these results in hdf5 files.

In a typical use case, you would use either `dense_code` or `sparse_code`, depending on the form of compression being utilized: `dense_code` is better for computing with uncompressed wavelet coefficients and bandpass filtering (where some levels are zeroed out, but other levels are fully preserved), while `sparse_code` is better for compression techniques that zero out large numbers of wavelet coefficients in every level, such as thresholding. Both methods require the use of a correlation matrix, which contains the precomputed values of cross-correlations of individual wavelet functions in a discrete wavelet transform - functions for generating correlation matrices are available in `support_code`.

## Using this repository

This repository is set up to be installed locally as a library using pip. To do so, type the following command into your command line interface:

`python -m pip install -e`

If your python installation is only on your user account, then add a `--user` tag before `-e`.

### Requirements

Obviously Python is required. In particular, this repository was written and tested with Python 3.8.3. Generally, an Anaconda installation of Python should be sufficient.

1. `dense_code` requires NumPy and SciPy.
2. `sparse_code` requires NumPy, SciPy, and ctypes. It also requires C (tested using Apple clang version 12.0.0), and for you to run the following line in the `sparse_code` directory: `cc -fPIC -shared -o ../../bin/diag_helper_sparse.so ./diag_helper_sparse.c`.
3. `support_code` requires NumPy, SciPy, PyWavelet, and h5py.

### Quickstart

Suppose you have two series of time-domain data, named `signal1` and `signal2`. You use PyWavelet to get the DWT coefficients of these series, called `coeffs1` and `coeffs2` respectively.

Before we can compute the temporal cross-correlations of `coeffs1` and `coeffs2`, we need to get the cross-correlation matrices for the level and wavelet family used in this DWT:

```
wavelet = 'db3' # Whatever wavelet family you used in the DWT to get coeffs1 and coeffs2
level   = 3     # Whatever level you used in the DWT to get coeffs1 and coeffs2

weight_matrices, mixed_weight_matrices, mixed_endpoint_indices = generate_stacked_weight_matrices(wavelet, level)
```

The arrays in `weight_matrices`, `mixed_weight_matrices`, and `mixed_endpoint_indices` contain the cross-correlations of different wavelets within the desired DWT (in this case the Daubechies 3 wavelet family with 3 levels). These objects can be reused for the cross-correlations of any wavelet coefficients corresponding to a DWT of the same wavelet family and level, serving as the "weights" of our computation. Once we get the cross-correlation matrices, we can then compute the cross-correlation of our two signals along a range of time-lags 0 to `lagmax` with:

```
dense_xcorrs  = calculate_nlevel_xcorrs(weight_matrices, mixed_weight_matrices, mixed_endpoint_indices, coeffs1, coeffs2, lagmax)
```

We can alternatively store sparse approximations of `coeffs1` and `coeffs2` (potentially attained by thresholding with `threshold_coeffs_one_channel` and `make_sparse_coeffs`), and use the sparse implementation in `sparse_code` instead:

```
sparse_xcorrs = calculate_nlevel_xcorrs_sparse(weight_matrices, mixed_weight_matrices, mixed_endpoint_indices, sparse_coeffs1, sparse_coeffs2, lagmax)
```

An example of creating weights and computing both sparse and dense wavelet-domain cross-correlations is available in `basic_test_case.py`, located in the `test_cases` repository. That script also shows how to import the required functions - `generate_stacked_weight_matrices`, `calculate_nlevel_xcorrs`, and `calculate_nlevel_xcorrs_sparse` - respectively.

### Limitations and Guidelines for Use

We've tested `wavelet_xcorr` on a variety of orthogonal wavelet families and levels. It provides reasonably accurate results, usually with a relative error < 1% (the relative error of cross-correlation on wavelet-compressed data is usually much higher than this to begin with). There are a few known limitations, however.

Using the terminology in Quickstart, suppose we have two time series `signal1` and `signal2` with respective wavelet coefficients stored in `coeffs1` and `coeffs2`, and wish to compute their cross-correlations ranging from time-lag 0 to `lagmax`. Then:

- `len(signal1)` and `len(signal2)` must both be divisible by 2^(level of DWT used). Use zero-padding to extend `signal1` and `signal2` so this is the case before applying the DWT.
- `lagmax` must be divisible by 2^(level of DWT used).
- For the dense implementation, `len(signal2) + lagmax` must be less than `len(signal1)`. This is *not* required for the sparse implementation.

In short, zero-pad the time series and the range of time-lags being tested so they're all divisible by the step size of wavelets used in the given DWT. This guarantees all functions within the wavelet family used for the DWT line up nicely with the two time series at every time-lag being evaluated. Failure to pad these components can increase the numerical error of the function, and potentially cause Python errors as well.

In general, the dense implementation is somewhat slower than time-domain cross-correlations computed via FFTs, but should scale similarly with larger problem sizes and can achieve excellent performance if wavelet compression is used as a bandpass filter (where only certain levels of wavelet coefficients are preserved). The sparse implementation is faster than FFT computations if the compression factor and the problem sizes are high enough - it works best if most of the wavelet coefficients within each level are zeroed out by compression.



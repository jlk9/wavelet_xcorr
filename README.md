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

In a typical use case, you would use either `dense_code` or `sparse_code`, depending on the form of compression being utilized: `dense_code` is better for computing with uncompressed wavelet coefficients and bandpass filtering (where some levels are zeroed out, but other levels are not changed at all), while `sparse_code` is better for compression techniques that zero out large numbers of wavelet coefficients in every level, such as thresholding. Both methods require the use of a correlation matrix, which contains the precomputed values of cross-correlations of individual wavelet functions in a discrete wavelet transform - functions for generating correlation matrices are available in `support_code`.


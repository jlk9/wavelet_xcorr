# wavelet_xcorr
Cross-correlations in the wavelet domain.

## What does this library do?

This repository contains a library of functions for performing temporal cross-correlations on wavelet-domain data. Traditionally, cross-correlation analysis on wavelet-compressed data is done by reconstructing the original time-series data or an approximation of it (via an inverse wavelet transform), and then computing the cross-correlations using direct or FFT-based methods. This library eliminates the intermediate step, allowing us to compute the temporal cross-correlations directly using the wavelet coefficients. In addition to the usual benefits of wavelet compression, such as noise reduction and bandpass filtering, our 'wavelet_xcorr' implementation reduces the storage requirements for cross-correlation analysis (since we no longer need to do store a time-domain representation of our data), and can improve the speed of analysis in some cases.

A more detailed description of the underlying theory and method that motivates this implementation, as well as the benefits, is available here: [title](https://vtechworks.lib.vt.edu/handle/10919/103864).


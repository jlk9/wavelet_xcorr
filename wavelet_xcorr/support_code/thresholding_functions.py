# Written by Joseph Kump, josek97@vt.edu
# Last modified 4/01/2021

import h5py
import numpy as np
import pywt

""" Given a coefficient vector and a percentile, this truncates all entries which are below
    the vector's percentile in magnitude.

Inputs:
coeffs          A set of wavelet coefficients derived from a DWT
percentile      The percentile of data to be preserved.

Returns:
thresheld_coeffs    A set of wavelet coefficients stored in the same format as coeffs, with all entries
                        smaller in magnitude than the percentile zeroed out.
"""
def threshold_coeffs_one_channel(coeffs, percentile):
    coeffs_array, coeffs_slices, coeffs_shapes = pywt.ravel_coeffs(coeffs, axes=None)

    thresheld_coeffs_array = pywt.threshold(coeffs_array, np.percentile(np.abs(coeffs_array), percentile),
                                             mode='hard', substitute=0)

    thresheld_coeffs = pywt.unravel_coeffs(thresheld_coeffs_array, coeffs_slices,
                                            coeffs_shapes, output_format='wavedecn')

    thresheld_coeffs = [thresheld_coeffs[0]] + [_['d'] for _ in thresheld_coeffs[1:]]
    
    return thresheld_coeffs


""" Given a set of coefficient vectors, this converts them into a sparse format.

Input:
coeffs          list of coefficient vectors for a signal in a DWT

Returns:
sparse_coeffs   coeffs in a sparse format - every dense array is replace by a triple of
                    (nonzero indices, nonzero values, orginal length of dense array)
"""
def make_sparse_coeffs(coeffs):
    
    sparse_coeffs = [(np.nonzero(coeff)[0], coeff[np.nonzero(coeff)[0]], len(coeff)) for coeff in coeffs]
    
    return sparse_coeffs



""" Creates HDF5 files containing wavelet-domain data instead of time-domain data. Stores these in
    a specified path. This preserves the file name (with wavelet_domain_ appended to the front) and
    the details of the transform, but not the metadata from the original file.

    Assumes the time-domain data of interest is in the DAS dataset, and it must be transposed.

Inputs:

Returns:
0. Also creates an hdf5 file containing the wavelet-transformed data, with datasets for each level
    of the DWT.

"""
def store_wavelet_hdf5(input_path, output_path, files, wavelet, level, percentile):

    for file in files:

        h5_file = h5py.File(input_path + file, 'r')
        DAS     = h5_file['DAS'][:]
        DAS     = DAS.astype(np.float64)
        
        h5_file.close()

        coeffs = pywt.wavedec(DAS.T, wavelet, level=level, mode="periodic")
        
        # TODO: threshold each channel here:
        if percentile != 0:
            for i in range(0, DAS.shape[1]):
                
                thresheld_coeffs = threshold_coeffs_one_channel([coeff[i] for coeff in coeffs], percentile)
                
                for j in range(len(coeffs)):
                    coeffs[j][i] = thresheld_coeffs[j]
        
        
        result = h5py.File(output_path + "wavelet_domain_" + str(percentile) + "_percentile_" + file, 'w')

        result.attrs["wavelet"] = wavelet
        result.attrs["level"]   = level

        result.create_dataset("approximation", data=coeffs[0])

        for i in range(1, level+1):
            result.create_dataset("detail_" + str(i), data=coeffs[-i])
            
        result.close()

    return 0









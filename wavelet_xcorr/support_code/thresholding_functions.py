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
def store_wavelet_hdf5(input_path, output_path, files, data_name, transposed, wavelet, level, percentile):

    for file in files:

        h5_file = h5py.File(input_path + file, 'r')
        data    = h5_file[data_name][:]
        data    = data.astype(np.float64)

        if transposed == "yes":
            data = data.T
        
        h5_file.close()

        coeffs = pywt.wavedec(data, wavelet, level=level, mode="periodic")
        
        # Threshold each channel here:
        if percentile != 0:
            for i in range(0, data.shape[0]):
                
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

""" Takes thresholded wavelets from store_wavelet_hdf5 and converts them into
    a sparse format.
    
Inputs:
input_path    the location of the dense thresholded wavelet-domain arrays
output_path   where to place the sparse array file
files         the list of thresholded wavelet-domain files to work with

Returns:
0, but also creates an hdf5 file of sparse wavelet coefficients for each file in input_path. These
files are located in output_path. Each file has three datasets:
    1. indices contains the nonzero indices at each row (one 2D array)
    2. values contains the nonzero values at each row (one 2D array)
    3. level_lengths contains the number of shift factors ("lengths") for each level
    4. level_starts shows the starting index for each level, since all levels are stored in one array

We also keep the wavelet and level attributes in our metadata.
"""
def make_wavelet_hdf5_sparse(input_path, output_path, files):
    
    for file in files:
        # First we need to extract the dense thresheld coefficients and concatenate them:
        dense_file            = h5py.File(input_path + file, 'r')
        wavelet               = dense_file.attrs["wavelet"]
        level                 = dense_file.attrs["level"]
        dense_file_all_levels = [dense_file["approximation"][:]] + [dense_file["detail_" + str(_)][:] for _ in range(level, 0, -1)]
        all_coeffs            = np.concatenate(dense_file_all_levels, axis=1)
        
        level_lengths = np.array([_.shape[1] for _ in dense_file_all_levels])
        level_starts  = np.array([np.sum(level_lengths[:i]) for i in range(len(level_lengths))])
        
        dense_file.close()
        
        # Now we get the nonzero indices for each row:
        indices = []
        values  = []
        for i in range(all_coeffs.shape[0]):
            indices.append(np.nonzero(all_coeffs[i])[0])
            values.append(all_coeffs[i,indices[i]])
        
        # Here we stack the indices and values into two arrays:
        # TODO: zero-pad to avoid issues with uneven lengths:
        index_array = np.stack(indices, axis=0)
        value_array = np.stack(values, axis=0)
        
        # And here we store this all into an hdf5 file:
        result = h5py.File(output_path + "sparse_" + file, 'w')

        result.attrs["wavelet"] = wavelet
        result.attrs["level"]   = level

        result.create_dataset("indices", data=index_array)
        result.create_dataset("values", data=value_array)
        result.create_dataset("level_starts", data=level_starts)
        result.create_dataset("level_lengths", data=level_lengths)
            
        result.close()
    
    return 0

""" Takes a sparse wavelet coefficients hdf5 file and breaks it up into a list of the wavelet coefficients,
    useful for our cross-correlation algorithm.
    
Inputs:
input_path    where the sparse wavelet coefficient files are located
files         the sparse wavelet coefficient file names

Outputs:
list of the wavelet coefficients for each file. Each file is represented by its own list of tuples for the sparse
coefficients at each channel and level. The list is sorted:

list of files -> list of channels -> list of levels -> each level is a tuple representinf a sparse vector

"""
def break_sparse_hdf5(input_path, files):
    
    all_file_coeffs = []
    
    for file in files:
        
        # The list of lists that make up our coefficients across the whole file:
        file_coeffs = []
        
        # We get the necessary information for this file:
        sparse_coeffs_file = h5py.File(input_path + file, 'r')
        indices            = sparse_coeffs_file["indices"][:]
        values             = sparse_coeffs_file["values"][:]
        level_lengths      = sparse_coeffs_file["level_lengths"][:]
        level_starts       = np.concatenate([sparse_coeffs_file["level_starts"][:], [sum(level_lengths)]])
        
        # Now we go through each channel:
        for i in range(indices.shape[0]):
            
            # Each channel has a different number of entries per level, so we need to find
            # which subsets of this channel correspond to each level:
            lv_bds = np.searchsorted(indices[i], level_starts)
            
            file_coeffs.append([(indices[i,lv_bds[_]:lv_bds[_+1]] - level_starts[_],
                                 values[i,lv_bds[_]:lv_bds[_+1]],
                                 level_lengths[_]) for _ in range(len(level_lengths))])
            
        all_file_coeffs.append(file_coeffs)
        
    
    return all_file_coeffs

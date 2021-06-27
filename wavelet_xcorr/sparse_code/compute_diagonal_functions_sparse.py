# Written by Joseph Kump, josek97@vt.edu
# Last modified 3/4/2021

import numpy as np
import math

from scipy.signal import correlate

# First, we need a few C related libraries:
from ctypes import c_void_p, c_double, c_int, cdll
from numpy.ctypeslib import ndpointer

# This loads the compiled C code and the function, getting its path relative to this module:
lib                   = cdll.LoadLibrary(__file__[:-62] + 'bin/diag_helper_sparse.so')
sparse_xcorr_sum_void = lib.sparse_xcorr_sum_void

# Now we need to load our function, which was already compiled with the following command:

# cc -fPIC -shared -o ../../bin/diag_helper_sparse.so ./diag_helper_sparse.c


""" Given two signals of the same level from coeffs1 and coeffs2, along with a set number of time lags and
    interior right and left entries, this will compute all of the required diagonals for our basic levelx -
    levelx component of our xcorr.
    
Inputs:
coeff1           the wavelet coefficients for the first (longer) signal
coeff2           the wavelet coefficients for the second (shorter) signal, same level as coeff1
right_length     the number of lags we need to compute pushing coeff2 forward, including the main diagonal,
                     for this weight matrix
left_length      the number of lags we need to compute pushing coeff2 backward for this weight matrix
lags             the number of time lags we need

Returns
left_diags       the left diagonals for each timelag
right_diags      the right diagonals for each timelag
"""
def compute_all_diags_sparse(sparse_coeff1, sparse_coeff2, left_length, right_length, offsets):
    
    left_diags  = np.zeros((left_length, offsets))
    right_diags = np.zeros((right_length, offsets))
    len_coeff2  = sparse_coeff2[2]

    # First we'll deal with the main diagonals, the first upper diagonals, and the last lower diagonals. These are the sliding
    # inner products of the two sets of wavelet coefficients against each other - we use C code to speed up the required for loops:
    # Note: might need to truncate the end of sparse_coeff1 here.
    diags = sparse_xcorr_calc_C_void((sparse_coeff1[0] + right_length-1, sparse_coeff1[1], sparse_coeff1[2]), sparse_coeff2, offsets + left_length + right_length-1)
    # The first few upper diagonals are the first sliding inner products:
    right_diags[::-1,0] = diags[:right_length]
    # Most of the sliding inner products make up the main diagonals:
    right_diags[0]      = diags[right_length-1:offsets+right_length-1]
    # The last few sliding inner products are the end lower diagonals:
    left_diags[:,offsets-1] = diags[-left_length:]
        
    # We can get the rest of the upper diagonals by slightly changing our main diagonals.
    # First, we need to determine which entries of the sparse vector must be added to each lower/upper diagonal.
    # These entries correspond to the end of what to remove from the uppers, then the begin/end of what to
    # remove from the lowers, respectively.

    # This determines what entries of coeff1 are needed to modify our upper and lower (right and left) diagonals:
    lower_upper_bounds = np.searchsorted(sparse_coeff1[0], [offsets-1, len_coeff2, len_coeff2+offsets-1], side='left', sorter=None)

    # This determines what entries of coeff2 modify our right diagonals:
    upper_begins = np.searchsorted(sparse_coeff2[0], list(range(right_length-1)), side='left', sorter=None)

    # This is the term from coeff1 we subtract from our upper diagonals:
    modify_diag = sparse_coeff1[1][:lower_upper_bounds[0]]
    # This gives us the indexing of the entries which are decremented by modify_diag:
    indexing = sparse_coeff1[0][:lower_upper_bounds[0]] + 1

    for i in range(1, right_length):

        # First, we know this upper diagonal almost equals the previous offset's previous upper diagonal:
        right_diags[i,1:] = right_diags[i-1,:offsets-1]

        # If our sparse vector contains the value to be removed, then we need to remove it to get the exact upper diagonal:
        if sparse_coeff2[0][upper_begins[i-1]] == i-1:
            right_diags[i, indexing] -= sparse_coeff2[1][upper_begins[i-1]] * modify_diag
        
    # Now we'll deal with the lower diagonals, first determining what part of coeff2 to remove:
    lower_ends = np.searchsorted(sparse_coeff2[0], list(range(len_coeff2-1, len_coeff2 - left_length-1, -1)), side='left', sorter=None)

    # This is the term from coeff1 we subtract from our lower diagonals:
    modify_diag = sparse_coeff1[1][lower_upper_bounds[1]:lower_upper_bounds[2]]
    # This gives us the indexing of the entries which are decremented by modify_diag:
    indexing    = sparse_coeff1[0][lower_upper_bounds[1]:lower_upper_bounds[2]] - len_coeff2

    # Here we'll establish the first lower subdiagonal:
    left_diags[0,:-1] = right_diags[0,1:]
    if (lower_ends[0] < len(sparse_coeff2[0])) and (sparse_coeff2[0][lower_ends[0]] == len_coeff2 - 1):
        left_diags[0, indexing] -= sparse_coeff2[1][lower_ends[0]] * modify_diag

    # And here we'll establish subsequent diagonals:
    for i in range(1, left_length):

        left_diags[i,:-1] = left_diags[i-1,1:]
        if (lower_ends[i] < len(sparse_coeff2[0])) and (sparse_coeff2[0][lower_ends[i]] == len_coeff2 - 1 - i):
            left_diags[i, indexing] -= sparse_coeff2[1][lower_ends[i]] * modify_diag
    
    return np.transpose(left_diags), np.transpose(right_diags)

""" Given two signals of the same level from coeffs1 and coeffs2, along with a set number of time lags and
    interior right and left entries, this will compute all of the required diagonals for our mixed level1 -
    level2 component of our xcorr.
    
Inputs:
coeff1           the wavelet coefficients for the first (longer) signal
coeff2           the wavelet coefficients for the second (shorter) signal, same level as coeff1
scale_diff
endpoint_ind
offsets
length_diag      the number of lags we need to compute pushing coeff2 forward for this weight matrix
len_coeff1       the length of coeff1

Returns
diags            the diagonals for each timelag
"""
def mixed_compute_all_diags_sparse(sparse_coeff1, sparse_coeff2, scale_diff, endpoint_ind, offsets, length_diag, len_coeff1):

    # We'll get the first column by padding some extra 0s to our timelags and applying redundancy rules:
    padding = math.ceil(length_diag / scale_diff) * scale_diff
    
    # Here we allocate the memory:
    diags = np.zeros((length_diag, offsets + padding))
    
    # The coeff2 endpoints are dependent on signal length, so we need to compute them here:
    coeff2_ends = endpoint_ind[2,:] + scale_diff * (len_coeff1 - endpoint_ind[1,:] - endpoint_ind[0,:])
    
    main_length = (offsets + padding) // scale_diff
    # We'll get the first diagonals here:

    # IDEA: break sparse_coeff2 up into parts, based on which part goes into which correlate call.
    # Then call sparse_xcorr_calc_C on those separate xcorr calculations

    # FIRST, call searchsorted out here to get the beginning and end indices of both coeffs for each i (begin will be coeff2[endpoint_ind[2,0]-i, end will be coeff2_ends[0]-i)
    coeff1_padded    = sparse_coeff1[0] + (padding // scale_diff)
    coeff1_endpoints = np.searchsorted(coeff1_padded, [endpoint_ind[0,0], len_coeff1-endpoint_ind[1,0]+main_length-1], side='left', sorter=None)
    coeff2_endpoints = np.searchsorted(sparse_coeff2[0], [endpoint_ind[2,0]-scale_diff+1, coeff2_ends[0]], side='left', sorter=None)

    # Here, we determine what portions of coeff1 and coeff2 we need to operate on:
    coeff1_to_compute = (coeff1_padded[coeff1_endpoints[0]:coeff1_endpoints[1]] - endpoint_ind[0,0], sparse_coeff1[1][coeff1_endpoints[0]:coeff1_endpoints[1]], sparse_coeff1[2])
    
    coeff2_to_compute_indices = sparse_coeff2[0][coeff2_endpoints[0]:coeff2_endpoints[1]]
    coeff2_to_compute_values  = sparse_coeff2[1][coeff2_endpoints[0]:coeff2_endpoints[1]]

    for i in range(scale_diff):
        # HERE, use np.where to find which entries of sparse_coeff2[0][begin for this i:end for this i] are divisible by scale_diff
        this_scale = ((coeff2_to_compute_indices % scale_diff) == (scale_diff - 1 - i))

        # LAST, call sparse_xcorr_calc_C here, with the sparse vectors filtered using work from above. The lagmax should be
        # main_length
        diags[0,i::scale_diff] = sparse_xcorr_calc_C_void(coeff1_to_compute, (coeff2_to_compute_indices[this_scale] // scale_diff, coeff2_to_compute_values[this_scale]), main_length)

    
    # Here we'll get subsequent rows based on the previous rows. Since we padded the front entries, we now use the redundancy rules to generate the
    # first column:
    for i in range(1, length_diag):
        # The basic rule is that the next diagonals is equal to the previous diagonal from the previous row:
        diags[i,1:] = diags[i-1,:-1]
        # TODO, for better accuracy:
        # Need to ADD element to front, if endpoint indices coeff1 start went down:
        # Need to REMOVE element from back, if endpoint indices coeff1 end went up:

    return diags[:,padding:]

""" In general, diagonals will go down one and to left because of how the signals slide across each other.
    Let's try that, make sure the overall error isn't too extreme, and test the savings:
    
    ASSUMPTION: the endpoint indices of our correlation matrix coeff1 will always either increment or decrement
    by 1 only. This affects how we fill in entries from the previous row of diagonals.

Inputs:
coeff1              the series of longer wavelets, from the shorter signal
coeff2              the series of shorter wavelets, from the longer signal
scale_diff          the difference between the scale of coeff1 and coeff2, 2 ^ (level 1 - level 2)
endpoint_ind        the endpoint coordinates for this mixed-wavelet xcorr
offsets             the number of diagonals we need, based on the strides of these wavelets and our
                        number of timelags
length_diag         the number of diagonals we need to compute
len_coeff1          the number of terms from coeff1 we use for 1 diagonal

Returns:
diags               the sliding xcorrs we need for interior points
"""
def mixed_compute_all_diags_case2_sparse(sparse_coeff1, sparse_coeff2, scale_diff, endpoint_ind, offsets, length_diag, len_coeff1):

    # We'll get the last column by padding some extra 0s to our timelags and applying redundancy rules:
    padding = math.ceil(length_diag / scale_diff) * scale_diff
    
    # Here we allocate the memory and get the coeff2 endpoints:
    diags       = np.zeros((length_diag, offsets + padding))
    coeff2_ends = endpoint_ind[2,:] + scale_diff * (len_coeff1 - endpoint_ind[1,:] - endpoint_ind[0,:])
    
    # FIRST, call searchsorted out here to get the beginning and end indices of both coeffs for each i (begin will be coeff2[endpoint_ind[2,0]-i, end will be coeff2_ends[0]-i)
    coeff1_endpoints = np.searchsorted(sparse_coeff1[0], [endpoint_ind[0,0], len_coeff1-endpoint_ind[1,0]], side='left', sorter=None)
    coeff2_endpoints = np.searchsorted(sparse_coeff2[0], [endpoint_ind[2,0], coeff2_ends[0]+offsets+padding-1], side='left', sorter=None)

    # Here, we determine what portions of coeff1 and coeff2 we need to operate on:
    coeff1_to_compute = (sparse_coeff1[0][coeff1_endpoints[0]:coeff1_endpoints[1]] - endpoint_ind[0,0], sparse_coeff1[1][coeff1_endpoints[0]:coeff1_endpoints[1]], sparse_coeff1[2])
    
    coeff2_to_compute_indices = sparse_coeff2[0][coeff2_endpoints[0]:coeff2_endpoints[1]] - endpoint_ind[2,0]
    coeff2_to_compute_values  = sparse_coeff2[1][coeff2_endpoints[0]:coeff2_endpoints[1]]
    for i in range(scale_diff):

        # HERE, use np.where to find which entries of sparse_coeff2[0][begin for this i:end for this i] are divisible by scale_diff
        this_scale             = ((coeff2_to_compute_indices % scale_diff) == i)
        diags[0,i::scale_diff] = sparse_xcorr_calc_C_void((coeff2_to_compute_indices[this_scale] // scale_diff, coeff2_to_compute_values[this_scale]), coeff1_to_compute,
                                                     len(diags[0,i::scale_diff]))

    # Fill in the correct entries for subsequent diagonals here. First, we need to determine what coeff 1 entries we need
    # to add and remove:
    coeff1_add_indices = np.searchsorted(sparse_coeff1[0], endpoint_ind[0,1:], side='left', sorter=None)
    coeff1_sub_indices = np.minimum(np.searchsorted(sparse_coeff1[0], len_coeff1 - endpoint_ind[1,1:], side='left', sorter=None), len(sparse_coeff1[0])-1)
    # We need to zero out the terms that are not the correct indices, or that repeat:
    coeff1_adds = sparse_coeff1[1][coeff1_add_indices] * (sparse_coeff1[0][coeff1_add_indices] == endpoint_ind[0,1:]) * (endpoint_ind[0,1:] != endpoint_ind[0,:-1])
    coeff1_subs = sparse_coeff1[1][coeff1_sub_indices] * (sparse_coeff1[0][coeff1_sub_indices] == len_coeff1 - endpoint_ind[1,1:]) * (endpoint_ind[1,1:] != endpoint_ind[1,:-1])

    coeff2_adds_start = np.searchsorted(sparse_coeff2[0], endpoint_ind[2,1:], side='left', sorter=None)
    coeff2_adds_end   = np.searchsorted(sparse_coeff2[0], endpoint_ind[2,1:] + offsets+padding-1, side='left', sorter=None)
    coeff2_subs_start = np.searchsorted(sparse_coeff2[0], coeff2_ends[1:], side='left', sorter=None)
    coeff2_subs_end   = np.searchsorted(sparse_coeff2[0], coeff2_ends[1:] + offsets+padding-1, side='left', sorter=None)

    for i in range(length_diag-1):
        # This is the basic rule, we need to decide what occurs in addition to this:
        diags[i+1,:-1] = diags[i,1:]
        # Need to ADD element to front, if endpoint indices coeff1 start went down:
        if coeff1_adds[i] != 0:
            diags[i+1, sparse_coeff2[0][coeff2_adds_start[i]:coeff2_adds_end[i]] - endpoint_ind[2,i+1]] +=  coeff1_adds[i] * sparse_coeff2[1][coeff2_adds_start[i]:coeff2_adds_end[i]]
        # Need to REMOVE element from back, if endpoint indices coeff1 went up:
        if coeff1_subs[i] != 0:
            diags[i+1, sparse_coeff2[0][coeff2_subs_start[i]:coeff2_subs_end[i]] - coeff2_ends[i+1]] -= coeff1_subs[i] * sparse_coeff2[1][coeff2_subs_start[i]:coeff2_subs_end[i]]
    
    return diags[:,:-padding]


""" Here we'll create our modified function, with the summation part done in C.
    Other things we need this function to do:
    1. Compute the first upper diagonals
    2. Compute the first lower diagonals
    3. Find the necessary entries of coeffs2 to modify values for subsequent diagonals
"""
def sparse_xcorr_calc_C_void(sparse_vector1, sparse_vector2, lagmax):

    # get array indices where vector1[0] is between curr_index and curr_index+1000:
    indices_left  = np.searchsorted(sparse_vector1[0], sparse_vector2[0], side='left', sorter=None)
    indices_right = np.searchsorted(sparse_vector1[0], sparse_vector2[0]+lagmax, side='left', sorter=None)
    length        = len(sparse_vector2[0])

    sparse_xcorrs = np.zeros((lagmax))
    
    sparse_xcorr_sum_void(c_void_p(indices_left.ctypes.data), c_void_p(indices_right.ctypes.data),
                          c_void_p(sparse_vector1[0].ctypes.data), c_void_p(sparse_vector2[0].ctypes.data),
                          c_void_p(sparse_vector1[1].ctypes.data), c_void_p(sparse_vector2[1].ctypes.data),
                          c_int(length), c_void_p(sparse_xcorrs.ctypes.data))
    
    return sparse_xcorrs


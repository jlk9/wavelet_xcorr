# Written by Joseph Kump, josek97@vt.edu
# Last modified 2/18/2021

# NOTE: this module has a dependency on compute_diagonal_functions and compute_vectorized_timelags

import numpy as np

from .compute_diagonal_functions  import compute_all_diags, mixed_compute_all_diags, mixed_compute_all_diags_case2
from .compute_vectorized_timelags import compute_vectorized_timelags, mixed_compute_vectorized_timelags

""" Our new vectorized implementation of xcorr calculations. This approach modularizes computation of the sliding
    inner products ("diagonals") needed for the interior entries of the weight matrix operations, allowing us to
    use redundancy to reduce compute time. It also aims to vectorize weight matrix / coeff vector computations whenever
    possible.

    TODO: modularize the begin and end computations as well, maybe write extra helper functions to reduce the length of
    this one.

    TODO: modify code so lags automatically line up with the stride sizes (for now, specify lags to be a multiple of all
            the strides in this DWT)

Inputs:
weight_matrices             1D list of weight matrices for each set of wavelets functions in this transformed
                                dataset, with timelags. Assumed to be in stacked format.
mixed_weight_matrices       2D list of weight matrices for each pair of wavelet functions in this transformed
                                dataset, with timelags. Assumed to be in stacked format.
mixed_endpoint_indices      2D list of endpoints for calculating the diagonals of mixed wavelet xcorr operations,
                                corresponding to the mixed_weight_matrices
coeffs1                     list of wavelet coefficients for the first signal, in the same order as for the
                                weight matrices
coeffs2                     list of wavelet coefficients for the second signal, in the same order as for the
                                weight matrices (length should be <= coeffs1)
lags                        integer, the largest time lag of the cross correlation: how far ahead start of
                                coeffs2 is from start of coeffs1 (assumed to be positive). Computes from 0
                                to lags.

Returns:
xcorr                   the overall cross correlations between the two transformed signals, from 0 to lags
"""
def calculate_nlevel_xcorrs(weight_matrices, mixed_weight_matrices, mixed_endpoint_indices, coeffs1, coeffs2, lags):

    # First we need to get some values necessary for calculating the time lag. Strides lets us know how big one
    # shift is at each level. Shifts lets us know how many steps of wavelets we need to move coeffs1 at each
    # level:
    levels      = np.array([len(weight_matrices) - 1] + list(range(len(weight_matrices) - 1, 0, -1)))
    strides     = np.array([thing[0].shape[0] for thing in weight_matrices])
    shifts      = lags // strides
    scale_diffs = 2 ** (levels[0]- levels[1:])
    steps       = [weight_matrices[_][0].shape[1] for _ in range(len(weight_matrices))]
    len_coeffs2 = [len(_) for _ in coeffs2]
    
    # These are used for the mixed wavelet terms, where we need quasi-symmetry:
    inverse_shifts = -1 * (-lags // strides)
    
    xcorrs = np.zeros((lags))
    
    # NEW STUFF FOR CREATING BEGIN END MATRICES HERE:
    coeff1_begins, coeff1_ends = form_coeff1_begin_end(coeffs1, steps, shifts, len_coeffs2)
    coeff2_begins, coeff2_ends = form_coeff2_begin_end(coeffs2[1:], steps[1:], scale_diffs)
    
    case2_coeff2_begins, case2_coeff2_ends = form_case2_coeff2_begin_end(coeffs1[1:], steps[1:],
                                                                         inverse_shifts[1:], len_coeffs2[1:])

    # Here we add the basic, non-mixed terms
    for i in range(len(weight_matrices)):
        
        length_left  = weight_matrices[i][1].shape[1]
        length_right = weight_matrices[i][2].shape[1]
        steps1       = steps[i]
        
        left_diags, right_diags = compute_all_diags(coeffs1[i], coeffs2[i], length_left,
                                                    length_right, shifts[i])

        # Here we loop over each stride of xcorrs for this level:
        xcorrs += compute_vectorized_timelags(weight_matrices[i], left_diags, right_diags, coeff1_begins[i],
                                              coeff1_ends[i], coeffs2[i][:steps1], coeffs2[i][-steps1:])

        # Here we proceed to the mixed terms:
        # We have to iterate through each mixed matrix, which only has matrices for smaller wavelets:
        for j in range(len(mixed_weight_matrices[i])):
            
            # This is the level of the smaller wavelet in this mixed computation:
            smaller_level = i+j+1
            
            # The stride, shift, and lag index we'll need are based on the smaller level:
            stride    = strides[smaller_level]
            shift     = shifts[smaller_level]
            steps2    = steps[i+j]
            
            # Here we'll deal with the larger coeffs1 term x the smaller coeffs2 term:
            length_diag = mixed_weight_matrices[i][j][1].shape[1]
            scale_diff  = 2**(levels[i]-levels[smaller_level])

            # CASE 1: we handle the coeffs1 term x the coeffs2 term for this level:
            
            # Here we deal with the interior terms:
            diags = mixed_compute_all_diags(coeffs1[i], coeffs2[smaller_level], scale_diff,
                                            mixed_endpoint_indices[i][j], shift, length_diag, len_coeffs2[i])
            
            xcorrs += mixed_compute_vectorized_timelags(mixed_weight_matrices[i][j], diags)
            
            # He we calculate the endpoint multiplications:
            begin_end  = [coeff1_begins[i] @ mixed_weight_matrices[i][j][0] @ coeff2_begins[i+j][_]
                        + coeff1_ends[i] @ mixed_weight_matrices[i][j][2] @ coeff2_ends[i+j][_]
                          for _ in range(-scale_diff, 0)]
            
            # We reformat the output's shape, and add it to xcorrs:
            xcorrs += np.concatenate(begin_end).flatten(order='F')

            # CASE 2: where we get the longer wavelet function from coeffs2 instead of coeffs1
            diags = mixed_compute_all_diags_case2(coeffs2[i], coeffs1[smaller_level], scale_diff, mixed_endpoint_indices[i][j],
                                                  inverse_shifts[smaller_level]+1, length_diag, len_coeffs2[i])
            
            # Here we flip our appropriate weight matrix:
            flipped_matrix = (mixed_weight_matrices[i][j][0], np.zeros(mixed_weight_matrices[i][j][1].shape),
                              mixed_weight_matrices[i][j][2])
            
            flipped_matrix[1][:] = np.flip(mixed_weight_matrices[i][j][1], axis=0)
            
            # The first diagonal entry is only used for timelag 0, so we will need to truncate the front end
            # of the resulting xcorrs:
            xcorrs += mixed_compute_vectorized_timelags(flipped_matrix, diags)[stride-1:-1]
            
            # Like for case 1, we add the beginning and end components to the xcorrs:
            begin_end = (coeffs2[i][:steps1]  @ np.flip(mixed_weight_matrices[i][j][0], axis=0) @ case2_coeff2_begins[i+j]
                       + coeffs2[i][-steps1:] @ np.flip(mixed_weight_matrices[i][j][2], axis=0) @ case2_coeff2_ends[i+j])
            
            xcorrs += begin_end.flatten(order='F')[stride-1:-1]
        
    return xcorrs

""" Helper, forms the beginning and end matrices for the first signal's coefficient vectors.
    Since they are all calculated here, we can reuse them as needed for calculating the endpoint
    components of xcorrs.

Inputs:
coeffs1
steps
shifts
len_coeffs2

Returns:
coeff1_begins
coeff1_ends
"""
def form_coeff1_begin_end(coeffs1, steps, shifts, len_coeffs2):
    
    count = range(len(steps))
    
    coeff1_begin_indexes = [np.arange(steps[_]) + np.array([np.arange(shifts[_])]).T for _ in count]
    
    coeff1_begins = [coeffs1[_][coeff1_begin_indexes[_]] for _ in count]
    coeff1_ends   = [coeffs1[_][coeff1_begin_indexes[_] + len_coeffs2[_] - steps[_]] for _ in count]
    
    return coeff1_begins, coeff1_ends

""" Helper, forms the beginning and end matrices for the second signal's coefficient vectors.
    Since they are all calculated here, we can reuse them as needed for calculating the endpoint
    components of xcorrs.

Inputs:
coeffs2
steps
scale_diffs

Returns:
coeff2_begins
coeff2_ends
"""
def form_coeff2_begin_end(coeffs2, steps, scale_diffs):
    
    count = range(len(steps))
    
    coeff2_indexes = [np.arange(-steps[_], 0) + np.array([np.arange(0, -scale_diffs[_], -1)]).T for _ in count]
    coeff2_begins  = []
    
    for i in count:
        coeff2_begin             = np.zeros((scale_diffs[i] * steps[i]))
        coeff2_begin[-steps[i]:] = coeffs2[i][:steps[i]]
        coeff2_begins.append(coeff2_begin[coeff2_indexes[i]])
        
    coeff2_ends = [coeffs2[_][coeff2_indexes[_]] for _ in count]
    
    return coeff2_begins, coeff2_ends

""" Helper, forms the beginning and end matrices for the first signal's coefficient vectors in the
    second mixed case. Since they are all calculated here, we can reuse them as needed for calculating
    the endpoint components of xcorrs.

Inputs:
coeffs1
steps
inverse_shifts
len_coeffs2

Returns:
case2_coeff2_begins
case2_coeff2_ends
"""
def form_case2_coeff2_begin_end(coeffs1, steps, inverse_shifts, len_coeffs2):
    
    count = range(len(steps))
    
    case2_coeff2_indexes = [np.arange(inverse_shifts[_] + 1) + np.array([np.arange(steps[_])]).T for _ in count]
    case2_coeff2_begins  = [coeffs1[_][case2_coeff2_indexes[_]] for _ in count]
    case2_coeff2_ends    = [coeffs1[_][case2_coeff2_indexes[_] + len_coeffs2[_]-steps[_]] for _ in count]
    
    return case2_coeff2_begins, case2_coeff2_ends


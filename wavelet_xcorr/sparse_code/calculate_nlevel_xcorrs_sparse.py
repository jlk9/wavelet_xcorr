# Written by Joseph Kump, josek97@vt.edu
# Last modified 3/4/2021

# NOTE: this module has a dependency on compute_diagonal_functions_sparse and compute_vectorized_timelags

import numpy as np

from .compute_diagonal_functions_sparse  import compute_all_diags_sparse, mixed_compute_all_diags_sparse, mixed_compute_all_diags_case2_sparse
from .compute_vectorized_timelags_sparse import compute_vectorized_timelags_sparse, mixed_compute_vectorized_timelags_sparse

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
sparse_coeffs1              list of wavelet coefficients for the first signal, in the same order as for the
                                weight matrices
sparse_coeffs2              list of wavelet coefficients for the second signal, in the same order as for the
                                weight matrices (length of signal should be <= coeffs1 signal length)
lags                        integer, the largest time lag of the cross correlation: how far ahead start of
                                coeffs2 is from start of coeffs1 (assumed to be positive). Computes from 0
                                to lags.

Returns:
xcorr       the overall cross correlations between the two transformed signals, from 0 to lags
"""
def calculate_nlevel_xcorrs_sparse(weight_matrices, mixed_weight_matrices, mixed_endpoint_indices, 
                                   sparse_coeffs1, sparse_coeffs2, lags):

    # First we need to get some values necessary for calculating the time lag. Strides lets us know how big one
    # shift is at each level. Shifts lets us know how many steps of wavelets we need to move coeffs1 at each
    # level:
    levels      = np.array([len(weight_matrices) - 1] + list(range(len(weight_matrices) - 1, 0, -1)))
    strides     = np.array([thing[0].shape[0] for thing in weight_matrices])
    shifts      = lags // strides
    scale_diffs = 2 ** (levels[0]- levels)
    steps       = np.array([weight_matrices[_][0].shape[1] for _ in range(len(weight_matrices))])
    len_coeffs2 = np.array([_[2] for _ in sparse_coeffs2])
    
    # These are used for the mixed wavelet terms, where we need quasi-symmetry:
    inverse_shifts = -1 * (-lags // strides)
    
    xcorrs = np.zeros((lags))
    
    # Here we generate the endpoint matrices we need. This assumes inverse_shift >= shifts:
    coeff1_begins, coeff1_ends = form_coeff1_begin_end(sparse_coeffs1, steps, inverse_shifts + 1, len_coeffs2)
    coeff2_begins, coeff2_ends = form_coeff2_begin_end(sparse_coeffs2, steps, scale_diffs, len_coeffs2)
    
    # Here we add the basic, non-mixed terms
    for i in range(len(weight_matrices)):

        length_left  = weight_matrices[i][1].shape[1]
        length_right = weight_matrices[i][2].shape[1]

        # We need to see if level i in either coeffs1 or coeffs2 is zero:
        if (sparse_coeffs1[i][0].shape[0] != 0) and (sparse_coeffs2[i][0].shape[0] != 0):
        
            left_diags, right_diags = compute_all_diags_sparse(sparse_coeffs1[i], sparse_coeffs2[i], length_left,
                                                               length_right, shifts[i])

            # Here we loop over each stride of xcorrs for this level:
            # NOTE: if we change the endpoints for coeffs2, this might need to change
            xcorrs += compute_vectorized_timelags_sparse(weight_matrices[i], left_diags, right_diags, coeff1_begins[i][:shifts[i]],
                                                         coeff1_ends[i][:shifts[i]], coeff2_begins[i][0], coeff2_ends[i][0])

        # Here we proceed to the mixed terms:
        # We have to iterate through each mixed matrix, which only has matrices for smaller wavelets:
        for j in range(len(mixed_weight_matrices[i])):
            
            # The stride and shift we'll need are based on the smaller level. The other terms help us compute
            # the diagonals properly:
            smaller_level = i+j+1
            stride        = strides[smaller_level]
            shift         = shifts[smaller_level]
            length_diag   = mixed_weight_matrices[i][j][1].shape[1]
            scale_diff    = 2**(levels[i]-levels[smaller_level])

            # CASE 1: we handle the coeffs1 term x the coeffs2 term for this level
            # First, we need to see if level i in coeffs1 or smaller_level in coeffs2 is zero:
            if (sparse_coeffs1[i][0].shape[0] != 0) and (sparse_coeffs2[smaller_level][0].shape[0] != 0):
            
                # Here we deal with the interior terms:
                diags = mixed_compute_all_diags_sparse(sparse_coeffs1[i], sparse_coeffs2[smaller_level], scale_diff,
                                                       mixed_endpoint_indices[i][j], shift, length_diag, len_coeffs2[i])
            
                xcorrs += mixed_compute_vectorized_timelags_sparse(mixed_weight_matrices[i][j], diags)
            
                # He we calculate the endpoint multiplications:
                begin_end  = [coeff1_begins[i][:shifts[i]] @ mixed_weight_matrices[i][j][0] @ coeff2_begins[smaller_level][_]
                            + coeff1_ends[i][:shifts[i]] @ mixed_weight_matrices[i][j][2] @ coeff2_ends[smaller_level][_]
                              for _ in range(-scale_diff, 0)]
            
                # We reformat the output's shape, and add it to xcorrs:
                xcorrs += np.concatenate(begin_end).flatten(order='F')


            # CASE 2: where we get the longer wavelet function from coeffs2 instead of coeffs1
            # First, we need to see if smaller_level in coeffs1 or i in coeffs2 is zero:
            if (sparse_coeffs1[smaller_level][0].shape[0] != 0) and (sparse_coeffs2[i][0].shape[0] != 0):

                diags = mixed_compute_all_diags_case2_sparse(sparse_coeffs2[i], sparse_coeffs1[smaller_level], scale_diff,
                                                             mixed_endpoint_indices[i][j], inverse_shifts[smaller_level]+1,
                                                             length_diag, len_coeffs2[i])
            
                # Here we flip our appropriate weight matrix:
                flipped_matrix = (mixed_weight_matrices[i][j][0], np.zeros(mixed_weight_matrices[i][j][1].shape),
                                  mixed_weight_matrices[i][j][2])
            
                flipped_matrix[1][:] = np.flip(mixed_weight_matrices[i][j][1], axis=0)
            
                # The first diagonal entry is only used for timelag 0, so we will need to truncate the front end
                # of the resulting xcorrs:
                xcorrs += mixed_compute_vectorized_timelags_sparse(flipped_matrix, diags)[stride-1:-1]
            
                # Like for case 1, we add the beginning and end components to the xcorrs:
                # NOTE: if we change the endpoints for coeffs2, this might need to change
                begin_end = (coeff2_begins[i][0] @ np.flip(mixed_weight_matrices[i][j][0], axis=0) @ coeff1_begins[smaller_level].T
                           + coeff2_ends[i][0]   @ np.flip(mixed_weight_matrices[i][j][2], axis=0) @ coeff1_ends[smaller_level].T)
            
                xcorrs += begin_end.flatten(order='F')[stride-1:-1]
        
    return xcorrs


""" Helper, forms the beginning and end matrices for the first signal's coefficient vectors.
    Since they are all calculated here, we can reuse them as needed for calculating the endpoint
    components of xcorrs.

Inputs:
sparse_coeffs1
steps
shifts
len_coeffs2

Returns:
coeff1_begins
coeff1_ends
"""
def form_coeff1_begin_end(sparse_coeffs1, steps, shifts, len_coeffs2):

    count = range(len(steps))

    coeff1_begin_values = (shifts + steps) - 1
    coeff1_end_values   = len_coeffs2 - steps
    coeff1_values       = np.stack((coeff1_begin_values, coeff1_end_values, (coeff1_begin_values + coeff1_end_values))).T

    # For the sparse implementation, we need to determine which entries of begin and end are nonzero:
    coeff1_endpoints = [np.searchsorted(sparse_coeffs1[_][0], coeff1_values[_], side='left', sorter=None) for _ in count]
    coeff1_begins    = [np.zeros((shifts[_] + steps[_] - 1)) for _ in count]
    coeff1_ends      = [np.zeros((shifts[_] + steps[_] - 1)) for _ in count]

    for i in count:
        coeff1_begins[i][sparse_coeffs1[i][0][:coeff1_endpoints[i][0]]] = sparse_coeffs1[i][1][:coeff1_endpoints[i][0]]
        coeff1_ends[i][sparse_coeffs1[i][0][coeff1_endpoints[i][1]:coeff1_endpoints[i][2]] - len_coeffs2[i] + steps[i]] = sparse_coeffs1[i][1][coeff1_endpoints[i][1]:coeff1_endpoints[i][2]]

    
    coeff1_indexes = [np.arange(steps[_]) + np.array([np.arange(shifts[_])]).T for _ in count]
    coeff1_begins  = [coeff1_begins[_][coeff1_indexes[_]] for _ in count]
    coeff1_ends    = [coeff1_ends[_][coeff1_indexes[_]] for _ in count]
    
    return coeff1_begins, coeff1_ends

""" Helper, forms the beginning and end matrices for the second signal's coefficient vectors.
    Since they are all calculated here, we can reuse them as needed for calculating the endpoint
    components of xcorrs.

Inputs:
sparse_coeffs2
steps
scale_diffs

Returns:
coeff2_begins
coeff2_ends
"""
def form_coeff2_begin_end(sparse_coeffs2, steps, scale_diffs, len_coeffs2):
    
    count = range(len(steps))

    # Here we determine what entries of the sparse coefficients we need:
    coeff2_values    = np.stack((steps, len_coeffs2 - scale_diffs - steps)).T
    coeff2_endpoints = [np.searchsorted(sparse_coeffs2[_][0], coeff2_values[_], side='left', sorter=None) for _ in count]
    coeff2_begins    = [np.zeros((scale_diffs[_] + steps[_])) for _ in count]
    coeff2_ends      = [np.zeros((scale_diffs[_] + steps[_])) for _ in count]
    
    for i in count:
        coeff2_begins[i][scale_diffs[i] + sparse_coeffs2[i][0][:coeff2_endpoints[i][0]]] = sparse_coeffs2[i][1][:coeff2_endpoints[i][0]]
        coeff2_ends[i][sparse_coeffs2[i][0][coeff2_endpoints[i][1]:] - coeff2_values[i,1]] = sparse_coeffs2[i][1][coeff2_endpoints[i][1]:]

    coeff2_indexes = [np.arange(-steps[_], 0) + np.array([np.arange(0, -scale_diffs[_], -1)]).T for _ in count]
    coeff2_begins  = [coeff2_begins[_][coeff2_indexes[_]] for _ in count] 
    coeff2_ends    = [coeff2_ends[_][coeff2_indexes[_]] for _ in count]
    
    return coeff2_begins, coeff2_ends




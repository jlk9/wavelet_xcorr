# Written by Joseph Kump, josek97@vt.edu
# Last modified 2/18/2021

import numpy as np

""" Here we compute the xcorr components for one levelx X levelx correlation. We need to compute
    the diagonals with compute_all_diagonals first.

Inputs:
xcorr_matrices      tuple of stacked weight tensors for this level
left_diags          diagonals below the main from compute_all_diags
right_diags         diagonals above and including the main from compute_all_diags
coeff1_begins       2D array of the beginning portions of the longer signal (coeffs1) in each sliding
                        window
coeff1_ends         2D array of the end portions of the longer signal (coeffs1) in each sliding
                        window
coeff2_begin        1D array of the first portion of the shorter coeffs2 signal. Since we're sliding it
                        along coeffs1, it is constant
coeff2_end          1D array of the end portion of the shorter coeffs2 signal

Returns:
The xcorr results for this level, arranged in a 1D array across timelags from 0 to lagmax.
The number of timelags we're calculating is given by the number of offsets in coeff1 begins
and ends, plus left and right diags. Thus we don't need to explicitly give this function
lagmax as a parameter.

This handles the weighted inner product operations of the starts and ends of the signals
with the beginning and end submatrices, and the interior entries of the signal with the
interior values of the correlation matrix. The sliding inner products between the signal
interiors are precomputed in advance using the compute_all_diagonals function.

The xcorr_matrices tuple used here is a stacked tensor, so it computes a range of timelags
concurrently.

NOTE: this is identical to compute_vectorized_timelags_sparse in the file of the same name
    in sparse_code. Eventually that function may be implemented differently to take advantage
    of sparsity, hence the separation between it and the dense one here. If you make changes to
    one function, you should change both.
"""
def compute_vectorized_timelags(xcorr_matrices, left_diags, right_diags, coeff1_begins,
                                coeff1_ends, coeff2_begin, coeff2_end):
    
    # Let's get our submatrices and interior:
    begin, interior_left, interior_right, end = xcorr_matrices
    
    # Here we'll add up the components of the xcorr. First the beginning submatrix:
    # Then the interior, both left and right of the main diagonal:
    # Then the end submatrix:
    
    interiors =  left_diags @ np.transpose(interior_left) + right_diags @ np.transpose(interior_right)
    
    # TODO: streamline this reshaping:
    begins    = np.transpose(coeff1_begins @ begin @ coeff2_begin).flatten()
    ends      = np.transpose(coeff1_ends   @ end   @ coeff2_end).flatten()
        
    return interiors.flatten() + begins + ends

""" This computes the weight matrix X the diagonals for the mixed case. Eventually we might incorporate the
    endpoints here as well for modularity.

Inputs:
xcorr_matrices      tuple, the mixed weight matrix for this level xcorr, in a stacked format
diags               the diagonals for this mixed xcorr, from either case

Returns:
interiors           the interior components of xcorrs at this level.

NOTE: this is identical to mixed_compute_vectorized_timelags_sparse in the file compute_vectorized_timelags_sparse
    in sparse_code. Eventually that function may be implemented differently to take advantage
    of sparsity, hence the separation between it and the dense one here. If you make changes to
    one function, you should change both.
"""
def mixed_compute_vectorized_timelags(xcorr_matrices, diags):
    
    begin, interior, end = xcorr_matrices
    
    interiors = np.transpose(interior @ diags)
    
    return interiors.flatten()


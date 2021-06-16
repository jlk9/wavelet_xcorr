# Written by Joseph Kump, josek97@vt.edu
# Last modified 3/9/2021

import numpy as np

from .form_weight_matrices_functions import form_weights_opt, form_weights_timelag_opt, form_mixed_weights_timelag_opt, mixed_diagonal_endpoint_indices

from .get_wavelet_functions import get_wavelet_functions

""" This generates all the correlation matrices ("weight matrices") for the provided wavelet functions,
    from get_wavelet_functions. It stores them in the inefficient old format, as lists of tuples.

Inputs:
wavelet_functions       list of wavelet functions from get_wavelet_functions, used to calculate the
                            xcorrs in these weight matrices
stride_size             the size of a single shift for a level 1 wavelet in the family that generates
                            these wavelet functions (seems to almost always be 2)

Returns:
weight_matrices_timelag         2D list, contains weight matrices for basic wavelet xcorrs, at every
                                    level provided and for every necessary time lag
mixed_weight_matrices_timelag   3D list, contains weight matrices for mixed wavelet xcorrs, at every
                                    level provided and for every necessary time lag. An extra
                                    dimension is needed since we store it by [level of first wavelet,
                                    level of second wavelet]
endpoint_indices                2D list of 2D arrays, contains the necessary endpoints for mixed-wavelet
                                    computations, since those are tricky but can be precomputed in advance
"""
def generate_all_weights_timelag(wavelet_functions, stride_size):
    
    levels = [len(wavelet_functions) - 1] + list(range(len(wavelet_functions) - 1, 0, -1))
    weight_matrices_timelag = []
    
    # First, we'll form the the basic (same-level) weight matrices. We store these in a 2D list, ordered
    # first by scaling factor (scaling functions, then decreasing levels of wavelet functions), then by
    # the timelag involved.
    for i in range(len(wavelet_functions)):

        # Here we calculate the 0-timelag weight matrix:
        weights_one_level = [form_weights_opt(wavelet_functions[i], levels[i], stride_size)]
        stride = stride_size * (2 ** (levels[i]-1))
        
        # Here we iterate through each of the positive timelag weight matrices and add them into our list:
        for j in range(1, stride):
            weights_one_level.append(form_weights_timelag_opt(wavelet_functions[i], levels[i], stride_size, j))
        
        weight_matrices_timelag.append(weights_one_level)

    # Next, we form the mixed-level weight matrices, iterating through every level and generating the mixed
    # weight matrices for all smaller levels. Since the resulting data is a 3D list, we ultimately need a triple
    # for loop, going from level of first wavelet -> level of second wavelet -> time lag:
    mixed_weight_matrices_timelag = []
    for i in range(len(wavelet_functions)-1):
        mixed_weight_matrices_i = []
        
        for j in range(i+1, len(wavelet_functions)):
            mixed_weight_matrices_ij = [form_mixed_weights_timelag_opt(wavelet_functions[i], wavelet_functions[j], levels[i], levels[j], stride_size, 0)]
            stride = stride_size * (2 ** (levels[j]-1))
            
            for k in range(1, stride):
                mixed_weight_matrices_ij.append(form_mixed_weights_timelag_opt(wavelet_functions[i], wavelet_functions[j], levels[i], levels[j], stride_size, k))
                
            mixed_weight_matrices_i.append(mixed_weight_matrices_ij)
            
        mixed_weight_matrices_timelag.append(mixed_weight_matrices_i)

    # Lastly, we precompute the endpoint indices that will help use use the mixed-level weight matrices for xcorr operations:
    endpoint_indices = []
    for i in range(len(mixed_weight_matrices_timelag)):
        endpoint_indices_i = []
        for j in range(len(mixed_weight_matrices_timelag[i])):

            steps1 = mixed_weight_matrices_timelag[i][j][0][0].shape[0]
            steps2 = mixed_weight_matrices_timelag[i][j][0][0].shape[1]
            scale_diff = 2**(levels[i]-levels[i+j+1])
            signal_length = max([len(mixed_weight_matrices_timelag[i][j][k][1]) for k in range(len(mixed_weight_matrices_timelag[i][j]))])

            endpoint_indices_i.append(mixed_diagonal_endpoint_indices(steps1, steps2, scale_diff, signal_length))

        endpoint_indices.append(endpoint_indices_i)
    
    return weight_matrices_timelag, mixed_weight_matrices_timelag, endpoint_indices


""" Takes the weight matrices structures from generate_all_weights_timelag and stackes them into
    multidimensional arrays: each level is now represented by one tuple instead of a list of tuples,
    with a begin matrix of 3 dimensions, and interior(s) of 2, and and end matrix of 3 dimensions.

Inputs:
weight_matrices         2D list of basic weight matrix tuples as made in generate_all_weights_timelag
mixed_weight_matrices   3D list of mixed weight matrix tuples as made in generate_all_weights_timelag

Returns:
stacked_weight_matrices         1D list of new, stacked basic weight matrix tuples (each component has
                                an extra dimension for time lag)
stacked_mixed_weight_matrices   2D list of new, stacked mixed weight matrix tuples (each component has
                                an extra dimension for time lag)
"""
def stack_weight_matrices(weight_matrices, mixed_weight_matrices):
    
    stacked_weight_matrices       = []
    stacked_mixed_weight_matrices = []
    
    # First we'll stack the basic weight matrices:
    for i in range(len(weight_matrices)):
        
        # The number of timelags (i.e. the stride size) for this level:
        stride = len(weight_matrices[i])
        
        # Here we allocate the memory for the beginning and end submatrices (both 3D)
        begin_submatrix = np.stack([weight_matrices[i][_][0] for _ in range(stride)])
        end_submatrix   = np.stack([weight_matrices[i][_][3] for _ in range(stride)])
        
        # Since interior left and interior right have different lengths, we cannot use np.stack, so
        # we'll allocate the necessay arrays here and fill them in with a loop (since performance doesn't
        # matter)
        interior_left_max  = max([len(weight_matrices[i][_][1]) for _ in range(stride)])
        interior_right_max = max([len(weight_matrices[i][_][2]) for _ in range(stride)])
        interior_left      = np.zeros((stride, interior_left_max))
        interior_right     = np.zeros((stride, interior_right_max))
        
        for j in range(stride):
            interior_left[j,:len(weight_matrices[i][j][1])]  = weight_matrices[i][j][1]
            interior_right[j,:len(weight_matrices[i][j][2])] = weight_matrices[i][j][2]
            
        stacked_weight_matrices.append((begin_submatrix, interior_left, interior_right, end_submatrix))
        
    
    # Here we'll stack the mixed weight matrices:
    for i in range(len(mixed_weight_matrices)):
        stacked_mixed_weight_matrices_i = []
        
        for j in range(len(mixed_weight_matrices[i])):
            
            stride = len(mixed_weight_matrices[i][j])
            
            # Here we allocate the memory for the beginning and end submatrices (both 3D)
            begin_submatrix = np.stack([mixed_weight_matrices[i][j][_][0] for _ in range(stride)])
            end_submatrix   = np.stack([mixed_weight_matrices[i][j][_][2] for _ in range(stride)])
            
            # Since are interiors already have a constant length, we can just call np.stack here,
            # unlike the basic case:
            interior = np.stack([mixed_weight_matrices[i][j][_][1] for _ in range(stride)])
            
            # Add this stacked matrix to this set of matrices for larger wavelet i
            stacked_mixed_weight_matrices_i.append((begin_submatrix, interior, end_submatrix))
            
        # Add this list of weight matrices to our 2D list of stacked mixed-wavelet matrices:
        stacked_mixed_weight_matrices.append(stacked_mixed_weight_matrices_i)

    # Here we append one last blank mixed-weight matrix list for the final detail wavelet:
    stacked_mixed_weight_matrices.append([])
    
    return stacked_weight_matrices, stacked_mixed_weight_matrices


""" Wrapper that combines get_wavelet_functions, generate_all_weights_timelag, and stack_weight_matrices
    to produce the necessary weight matrices for a DWT given its wavelet family and level

Inputs:
wavelet_name   string, the type of wavelet used for this particular wavelet transform
level          positive integer, the level of the wavelet transform (the number of scaling factors we'll compute)

Returns:
stacked_weight_matrices         1D list of new, stacked basic weight matrix tuples (each component has
                                an extra dimension for time lag)
stacked_mixed_weight_matrices   2D list of new, stacked mixed weight matrix tuples (each component has
                                an extra dimension for time lag)
endpoint_indices                2D list of 2D arrays, contains the necessary endpoints for mixed-wavelet
                                    computations, since those are tricky but can be precomputed in advance
"""
def generate_stacked_weight_matrices(wavelet_name, level):

    sample_wavelets, sample_stride                                 = get_wavelet_functions(wavelet_name, level)
    weight_matrices, mixed_weight_matrices, mixed_endpoint_indices = generate_all_weights_timelag(sample_wavelets,
                                                                                                  sample_stride)
    stacked_weight_matrices, stacked_mixed_weight_matrices         = stack_weight_matrices(weight_matrices,
                                                                                           mixed_weight_matrices)

    # Needed for a design choice in calculate_xcorrs
    stacked_mixed_weight_matrices.append([])

    return stacked_weight_matrices, stacked_mixed_weight_matrices, mixed_endpoint_indices


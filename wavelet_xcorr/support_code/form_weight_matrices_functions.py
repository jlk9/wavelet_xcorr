# Written by Joseph Kump, josek97@vt.edu
# Last modified 2/2/2021

import numpy as np

from scipy.signal import correlate

""" This creates a weight matrix object for a particular wavelet function of level 'level' at time lag 0,
    ehich can then be used for computation in wavelet-domain xcorr calculations.
    Because of the mathematical properties of wavelets, we can represent this weight matrix as a quadruple,
    with four specific entries: a beginning submatrix handling xcorrs of wavelets at the beginning of
    the signal, interior-left and interior-right arrays that stores the repeating diagonal entries that
    make up the bulk of the weight matrix's nonzero values, and an end submatrix that stores the xcorrs
    of the wavelets at the end of the signal.
    
    The beginning and end submatrices are of size steps X steps, where steps is the length of the wavelet
    function's support / the size of its shifts. This is because all of the wavelet functions that are not
    fully "on" the signal will be in either the first 'steps' or last 'steps'-count of the signal.

    The interior-right entries, which include the main diagonal and all diagonals above the main, are also
    of length 'steps' - the bandendess of the correlation matrix proves any wavelets functions of a difference
    in shift facter greater than steps have no overlapping support. Meanwhile, the interior left entries handles
    all subdiagonals below the main diagonal, so it is of length steps - 1. In the zero timelag case we have
    symmetry in the correlation matrix, so its values equal the interior right.

    In addition to the wavelet function itself, we also need to provide the level (the scaling factor) of this
    particular function, and the size of a single shift for the level-1 case of this wavelet family.

Inputs:
wave_fun        array of floats, the support of a wavelet function as it appears in the signal
                    (i.e. when you reconstruct it with waverec)
level           int, the level of this wavelets function, its scaling factor is 2 ^ (level-1)
stride_size     int, the number of entries this wavelet function jumps with each shift factor

Returns:
weight_matrix   tuple that represents the weight coefficients of this wavelet domain xcorr. The components are:

wavelet_beginning       2D array, xcorr weights for endpoint wavelets at the beginning of the signal (these
                            are not fully represented on the signal)
wavelet_interior_left   1D array, xcorr weights for wavelets that are fully on the signal, making up the lower
                            subdiagonals
wavelet_interior_right  1D array, xcorr weights for wavelets that are fully on the signal, making up the upper
                            subdiagonals and main diagonal
wavelet_end             2D array, xcorr weights for endpoint wavelets at the end of the signal (these are not
                            fully represented on the signal)
"""
def form_weights_opt(wave_fun, level, stride_size):
    
    # First we need our stride size and number of steps
    stride = stride_size * (2 ** (level-1))
    steps  = len(wave_fun) // stride
    
    # Our wavelet beginning and end will both be 2D lists of length steps, with decreasing length for each entry
    # Wavelet interior will be of length <= steps
    
    # First let's deal with interior right and left. Main diag is the index of the main diagonal:
    wavelet_interior_right = np.zeros((steps))
    wavelet_interior_left = np.zeros((steps-1))
    
    wavelet_interior_right[0] = np.inner(wave_fun, wave_fun)
    for i in range(1, steps):
        wavelet_interior_right[i]  = np.inner(wave_fun[i*stride:], wave_fun[:-i*stride])
        wavelet_interior_left[i-1] = wavelet_interior_right[i]


    # If steps = 1, then there are no wavelets partially represented on the signal (they're either fully
    # represented or not at all), so we can disregard the beginning and end wavelets. We'll set beginning
    # and interior equal to 0 in this case:
    if steps == 0:
        return (np.array([0]), wavelet_interior, np.array([0]))
    
    
    # Now let's deal with the beginning. This requires us to operate with the chopped off ends
    # of wavelets, which is a little confusing:
    wavelet_begin = np.zeros((steps, steps))
    for i in range(steps):
        wave_begin          = wave_fun[-(i+1)*stride:]
        values              = list(correlate(wave_begin, wave_fun, mode='valid'))
        wavelet_begin[i,i:] = values[::stride] - wavelet_interior_right[:steps-i]
        
    # Now let's deal with the ending:
    wavelet_end = np.zeros((steps, steps))
    for i in range(steps):
        wave_end                              = wave_fun[:(i+1)*stride]
        values                                = list(correlate(wave_fun, wave_end, mode='valid'))
        wavelet_end[steps-1-i, steps-1-i::-1] = values[::stride] - wavelet_interior_right[:steps-i]
        
    # Since weights for basic 0-lag matrix are symmetric, we can fill in the remaining entries here:
    for i in range(1, steps):
        for j in range(i):
            wavelet_begin[i,j] = wavelet_begin[j,i]
            wavelet_end[j,i] = wavelet_end[i,j]
    
    return (wavelet_begin, wavelet_interior_left, wavelet_interior_right, wavelet_end)


""" This creates a weight matrix object for a particular wavelet function of level 'level' at a time lag
    between 1 and stride size - 1, which can then be used for computation in wavelet-domain xcorr calculations.

    Because of the mathematical properties of wavelets, we can represent this weight matrix as a quadruple,
    with four specific entries: a beginning submatrix handling xcorrs of wavelets at the beginning of
    the signal, interior-left and interior-right arrays that stores the repeating diagonal entries that
    make up the bulk of the weight matrix's nonzero values, and an end submatrix that stores the xcorrs
    of the wavelets at the end of the signal.
    
    The beginning and end submatrices are of size steps X steps, where steps is the length of the wavelet
    function's support / the size of its shifts. This is because all of the wavelet functions that are not
    fully "on" the signal will be in either the first 'steps' or last 'steps'-count of the signal.

    The interior-right entries, which include the main diagonal and all diagonals above the main, are also
    of length 'steps' - the bandendess of the correlation matrix proves any wavelets functions of a difference
    in shift facter greater than steps have no overlapping support. Meanwhile, the interior left entries handles
    all subdiagonals below the main diagonal, so it is of length steps - 1.

    In addition to the wavelet function itself, we also need to provide the level (the scaling factor) of this
    particular function, and the size of a single shift for the level-1 case of this wavelet family.

    This differs from form_weights_opt because it handles timelags between 1 and the level's stride size - 1.

Inputs:
wave_fun        array of floats, the support of a wavelet function as it appears in the signal
                    (i.e. when you reconstruct it with waverec)
level           int, the level of this wavelets function, its scaling factor is 2 ^ (level-1)
stride_size     int, the number of entries this wavelet function jumps with each shift factor
lag             int, the time lag between the two sets of wavelet functions involved, between 1 and
                    stride_size * 2 ^ (level-1)

Returns:
weight_matrix   tuple that represents the weight coefficients of this wavelet domain xcorr. The components are:

wavelet_beginning       2D array, xcorr weights for endpoint wavelets at the beginning of the signal (these
                            are not fully represented on the signal)
wavelet_interior_left   1D array, xcorr weights for wavelets that are fully on the signal, making up the lower
                            subdiagonals
wavelet_interior_right  1D array, xcorr weights for wavelets that are fully on the signal, making up the upper
                            subdiagonals and main diagonal
wavelet_end             2D array, xcorr weights for endpoint wavelets at the end of the signal (these are not
                            fully represented on the signal)
"""
def form_weights_timelag_opt(wave_fun, level, stride_size, lag):
    
    # First we need our stride size and number of steps
    stride = stride_size * (2 ** (level-1))
    steps  = len(wave_fun) // stride
    
    # First we'll deal with the interior:
    interior       = np.correlate(wave_fun, wave_fun, mode='full')
    interior_right = interior[len(wave_fun)-1+lag::stride]
    interior_left  = interior[len(wave_fun)-1-(stride-lag)::-stride]
    
    
    # If either of our step sizes are 1, then we don't need to represent beginning and end terms:
    if steps == 1:
        return (np.array([0]), interior_left, interior_right, np.array([0]))
    
    # We'll deal with the beginning:
    begin = np.zeros((steps, steps))
    
    for i in range(steps):
        chunk1 = wave_fun[-(i+1)*stride:]
        
        for j in range(steps):
            # Need to append the lag onto the front of the wavelet:
            chunk2     = np.append(np.zeros(lag), wave_fun[-(j+1)*stride:])
            min_length = min(len(chunk1), len(chunk2))
            begin[i,j] = np.inner(chunk1[:min_length], chunk2[:min_length])
            
            
    # We'll deal with the ending:
    end = np.zeros((steps, steps))
    
    # This is used to get the proper, timelagged portions of wavelets:
    wavefun_lag = np.append(np.zeros(lag), wave_fun)
    
    for i in range(steps):
        chunk1 = wave_fun[:(i+1)*stride]
        
        for j in range(steps):
            chunk2     = wavefun_lag[:(j+1)*stride]
            min_length = min(len(chunk1), len(chunk2))
            end[steps-1-i,steps-1-j] = np.inner(chunk1[-min_length:], chunk2[-min_length:])
            
    # Subtract interior values from begin and end:
    interior = np.append(np.flip(interior_left), interior_right)
    for i in range(steps):
        # These are the interior values we'd "expect" for these diagonals:
        begin[i,:]       -= interior[len(interior_left)-i:len(interior_left)-i+steps]
        end[steps-1-i,:] -= interior[len(interior_left)-steps+1+i:len(interior_left)+1+i]
            
    return begin, interior_left, interior_right, end

""" Given two wavelet functions from one DWT, as they are represented in get_wavelet_functions, this will form a
    quasi-banded, quasi-toeplitz matrix representing the weights of their mixed cross correlations at a nonnegative
    time lag.

    Unlike the basic, same-level case, in the mixed case we place all the interior values into a single array. This
    is because there is no longer a main diagonal in the mixed-weight matrix.

Inputs:
wave_fun1        array of floats, the first wavelet function (should be the longer function)
wave_fun2        array of floats, the second wavelet function
level1           positive integer, level of wave_fun1
level2           positive integer, level of wave_fun2
stride_size      positive integer, length of stride for a level 1 detail wavelet in this DWT
lag              positive integer, the time lag between the two sets of coefficients

Returns:
begin_mixed_terms       2D array of weights for the beginning wavelets in this signal (ones that are not fully
                            represented in the signal)
interior_mixed_terms    1D array of weights for the interior wavelets (i.e. wavelets that are fully in the signal),
                            note this is toeplitz but not necessarily symmetric
end_mixed_terms         2D array of weights for the end wavelets in this signal
"""
def form_mixed_weights_timelag_opt(wave_fun1, wave_fun2, level1, level2, stride_size, lag):
    
    # First we need our stride size and number of steps for each:
    stride1 = stride_size * (2 ** (level1-1))
    stride2 = stride_size * (2 ** (level2-1))
    steps1  = len(wave_fun1) // stride1
    steps2  = len(wave_fun2) // stride2
    
    # Here are the interior points. This will be a row that increments along stride2. Imagine wave_fun1
    # is the wavelet that is "constant", while wave_fun2 is sliding along (hence the order of parameters in
    # correlate):
    correlations = np.correlate(wave_fun1, wave_fun2, mode='full')
    # We only need entries sliding along a stride size of the second wavelet function. The starting point is when
    # the overlap is lag, which happens at index lag-1:
    interior_mixed_terms = correlations[lag-1::stride2]
    
    if lag == 0:
        interior_mixed_terms = np.append(0, correlations[stride2-1::stride2])
    
    # If either of our step sizes are 1, then we don't need to represent beginning and end terms:
    if (steps1 == 1) or (steps2 == 1):
        return (np.array([0]), interior_mixed_terms, np.array([0]))

    # The number of outputs we'll have will be (steps1)*(steps2), so let's start with that:
    begin_mixed_terms = np.zeros((steps1, steps2))
    
    # Now we'll use a loop to go through each of the steps1-1 configurations for the first signal:
    for i in range(steps1):
        chunk1 = wave_fun1[-(i+1)*stride1:]
        
        for j in range(steps2):
            chunk2 = wave_fun2[-(j+1)*stride2:]
            # here we account for the lag in chunk2:
            chunk2 = np.append(np.zeros(lag), chunk2)
            min_length = min(len(chunk1), len(chunk2))
            begin_mixed_terms[i,j] = np.inner(chunk1[:min_length], chunk2[:min_length])
            
    
    # TODO: deal with end submatrix, make it steps1 x steps2
    # Here we'll deal with the ending:
    end_mixed_terms = np.zeros((steps1, steps2))
    
    # This is used to get the proper, timelagged portions of wavelets:
    wavefun2_lag = np.append(np.zeros(lag), wave_fun2)
    
    # Now we'll use a loop to go through each of the steps1-1 configurations for the first signal:
    for i in range(steps1):
        chunk1 = wave_fun1[:(i+1)*stride1]
        
        for j in range(steps2):
            chunk2 = wavefun2_lag[:(j+1)*stride2]
            min_length = min(len(chunk1), len(chunk2))
            end_mixed_terms[steps1 - 1 - i,steps2 - 1 - j] = np.inner(chunk1[-min_length:], chunk2[-min_length:])
            
    return (begin_mixed_terms, interior_mixed_terms, end_mixed_terms)


""" The endpoint coordinates for mixed-level wavelet xcorrs are tricky to determine, so this helper function
    generates them in advance. These coordinates are relative to the beginning and end of the signal, so they can
    be used for any signal length, for the two specific levels they handle.

Inputs:
steps1          number of steps in first wavelet function
steps2          number of steps in second wavelet function
scale_diff      the difference in scaling factor between the two levels (multiplicative: scale 1 is X times
                    longer than scale 2)
interior_length the number of nonzero weights in the interior of this weight matrix  

Returns:
endpoint_indices   the endpoints to base our diagonals off of. The rows are the beginning of the longer wavelet,
                        the end of the longer wavelet, and the beginning of the shorter wavelet respectively 
                        (end of shorter wavelet is determined during the xcorr calculation)
"""
def mixed_diagonal_endpoint_indices(steps1, steps2, scale_diff, interior_length):

    endpoint_indices = np.zeros((3, interior_length), dtype=int)

    for i in range(steps2 - scale_diff + 1):

        endpoint_indices[0,i] = steps1
        endpoint_indices[1,i] = min((i // scale_diff), steps1)
        endpoint_indices[2,i] = i + scale_diff - 1

    for i in range(steps2 - scale_diff + 1, interior_length):

        endpoint_indices[0,i] = steps1 - (i + scale_diff - 1 - steps2) // scale_diff
        endpoint_indices[1,i] = min((i // scale_diff), steps1)
        endpoint_indices[2,i] = steps2 + ((i + scale_diff - 1 - steps2) % scale_diff)

    return endpoint_indices

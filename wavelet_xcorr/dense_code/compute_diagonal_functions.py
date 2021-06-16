# Written by Joseph Kump, josek97@vt.edu
# Last modified 2/18/2021

import numpy as np

from scipy.signal import correlate

""" Given two signals of the same level from coeffs1 and coeffs2, along with a set number of time lags and
    interior right and left entries, this will compute all of the required diagonals for our basic levelx -
    levelx component of our xcorr.
    
Inputs:
coeff1           the wavelet coefficients for the first (longer) signal
coeff2           the wavelet coefficients for the second (shorter) signal, same level as coeff1
right_length     the number of steps we need to compute pushing coeff2 forward, including the main diagonal,
                     for this weight matrix
left_length      the number of steps we need to compute pushing coeff2 backward for this weight matrix
offsets          the number of time lags we need for this level

NOTE: right_length and left_length are based directly on the lengths of interior_left and interior_right
    in the weight matrix used for computing the xcorr between these coefficients.

Returns
left_diags       the left diagonals for each timelag
right_diags      the right diagonals for each timelag

NOTE: the sizes of left_diags and right_diags are determined by the number of interior entries we need
    to compute at each time lag X the number of time lags.

"""
def compute_all_diags(coeff1, coeff2, left_length, right_length, offsets):
    
    left_diags  = np.zeros((left_length, offsets))
    right_diags = np.zeros((right_length, offsets))
    
    len_coeff2 = len(coeff2)
    
    # First we'll deal with the main diagonals:
    #for i in range(offsets):
    #    right_diags[0,i] = np.inner(coeff1[i:len_coeff2+i], coeff2)
    # We'll do this using FFTs:
    right_diags[0] = correlate(coeff1[:len_coeff2+offsets-1], coeff2, mode='valid', method='fft')
        
    # Now we'll deal with the first upper diagonals, by filling in from the main diagonal.
    # The first upper diagonals at offset 0 do not have a relation to any of our main diagonals, so they
    # must be computed separately:
    right_diags[1:,0]  = np.array([np.inner(coeff1[:len_coeff2-_], coeff2[_:]) for _ in range(1, right_length)])
    
    # We can get the rest of the upper diagonals by slightly changing our main diagonals (and previous uppers)
    for i in range(1, right_length):
        right_diags[i,1:] = right_diags[i-1,:offsets-1] - coeff2[i-1] * coeff1[:offsets-1]
        
    # Now we'll deal with the lower diagonals, first the last lower diagonals at the final offset:
    left_diags[:,offsets-1] = np.array([np.inner(coeff1[offsets+_:len_coeff2+offsets-1], coeff2[:-_-1])
                                        for _ in range(left_length)])
    
    # Here we'll establish the first lower diagonal:
    left_diags[0,:-1] = right_diags[0,1:] - coeff2[-1] * coeff1[len_coeff2:len_coeff2+offsets-1]
    
    # And here we'll establish subsequent diagonals:
    for i in range(1, left_length):
        left_diags[i,:-1] = left_diags[i-1,1:] - coeff2[-i-1] * coeff1[len_coeff2:len_coeff2+offsets-1]
    
    return np.transpose(left_diags), np.transpose(right_diags)


""" Computes the diagonals for the mixed wavelet xcorr in Case 1, where the longer wavelets are coming
    from the longer signal. Getting the indices of coeff1 and coeff2 right for this is very intricate -
    it's likely the main source of error.
    Note: this one is meant specifically for the first case, coeff1 is larger wavelets, coeff2 is smaller.
    This computes all the necessary diagonals for the interior of the mixed-wavelet computation.

    In this version of the function, we take advantage of the redundancy in values between diagonals at
    different offsets. We still calculate the first diagonals as we did previously, but now we use the
    values of the first diagonals to fill in subsequent diagonals, based on rules of sliding inner products
    between vectors.

Inputs:
coeff1              array of floats, the series of longer wavelets from the longer signal
coeff2              array of floats, the series of shorter wavelets from the shorter signal
scale_diff          int, the difference between the scale of coeff1 and coeff2, calculated as 2 ^ (level 1 - level 2)
endpoint_ind        the endpoint coordinates for this mixed-wavelet xcorr
offsets             int, the number of diagonals we need, based on the strides of these wavelets and our
                        number of timelags
length_diag         int, the number of diagonals we need to compute, based on the number of interior entries
                        in the corresponding weight matrix
len_coeff1          int, the number of terms from coeff1 we use for 1 diagonal

Returns:
diags               the sliding xcorrs we need for interior points, a 2D array of floats of size offsets x
                        length_diag
"""
def mixed_compute_all_diags(coeff1, coeff2, scale_diff, endpoint_ind, offsets, length_diag, len_coeff1):
    
    # Here we allocate the memory:
    diags = np.zeros((length_diag, offsets))
    
    # The coeff2 endpoints are dependent on signal length, so we need to compute them here:
    coeff2_ends = endpoint_ind[2,:] + scale_diff * (len_coeff1 - endpoint_ind[1,:] - endpoint_ind[0,:])
    
    main_length = offsets // scale_diff
    # We'll get the first diagonals here:
    for i in range(scale_diff):
        diags[0,i::scale_diff] = correlate(coeff1[endpoint_ind[0,0]:len_coeff1-endpoint_ind[1,0]+main_length-1],
                                           coeff2[endpoint_ind[2,0]-i:coeff2_ends[0]-i:scale_diff],
                                           mode='valid', method='fft')
    
    # Here we'll calculate the first column, since we can't use redundancy rules for it:
    diags[1:,0] = [np.inner(coeff1[endpoint_ind[0,i]:len_coeff1-endpoint_ind[1,i]],
                            coeff2[endpoint_ind[2,i]:coeff2_ends[i]:scale_diff])for i in range(1, length_diag)]
    
    
    # Here we'll get subsequent rows based on the previous rows:
    for i in range(1, length_diag):
        # The basic rule is that the next diagonals is equal to the previous diagonal from the previous row:
        diags[i,1:] = diags[i-1,:-1]
        # TODO, for better accuracy:
        # Need to ADD element to front, if endpoint indices coeff1 start went down:
        # Need to REMOVE element from back, if endpoint indices coeff1 end went up:

    return diags


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
def mixed_compute_all_diags_case2(coeff1, coeff2, scale_diff, endpoint_ind, offsets, length_diag, len_coeff1):
    
    # Here we allocate the memory and get the coeff2 endpoints:
    diags       = np.zeros((length_diag, offsets))
    coeff2_ends = endpoint_ind[2,:] + scale_diff * (len_coeff1 - endpoint_ind[1,:] - endpoint_ind[0,:])
    
    # Each row will have a cyclic pattern for its entries related to endpoint_ind, so it may be best to
    # base this off of that.
    # Can use the current calculate_nlevel_xcorrs_vec diag fill-in for error checking
    for i in range(scale_diff):
        # Fix the need for [:len(diags[i,j::scale_diff])]
        diags[0,i::scale_diff] = correlate(coeff2[i+endpoint_ind[2,0]:coeff2_ends[0]+offsets-1:scale_diff],
                                           coeff1[endpoint_ind[0,0]:len_coeff1-endpoint_ind[1,0]],
                                           mode='valid', method='fft')[:len(diags[0,i::scale_diff])]
        
    # Since the rightmost entries don't have a main diagonal to base off of, we'll get them here:
    diags[1:,-1] = [np.inner(coeff1[endpoint_ind[0,i]:len_coeff1-endpoint_ind[1,i]],
                             coeff2[offsets-1+endpoint_ind[2,i]:offsets-1+coeff2_ends[i]:scale_diff])
                             for i in range(1, length_diag)]
    
    # Fill in the correct entries for subsequent diagonals here:
    for i in range(1, length_diag):
        # This is the basic rule, we need to decide what occurs in addition to this:
        diags[i,:-1] = diags[i-1,1:]
        # Need to ADD element to front, if endpoint indices coeff1 start went down:
        if endpoint_ind[0,i] < endpoint_ind[0,i-1]:
            diags[i,:-1] += coeff1[endpoint_ind[0,i]] * coeff2[endpoint_ind[2,i]:endpoint_ind[2,i]+offsets-1]
        # Need to REMOVE element from back, if endpoint indices coeff1 went up:
        if endpoint_ind[1,i] > endpoint_ind[1,i-1]:
            diags[i,:-1] -= coeff1[len_coeff1-endpoint_ind[1,i]] * coeff2[coeff2_ends[i]:coeff2_ends[i]+offsets-1]
    
    return diags


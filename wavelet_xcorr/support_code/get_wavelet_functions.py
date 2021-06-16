# Written by Joseph Kump, josek97@vt.edu
# Last modified 3/9/2021

import numpy as np

import pywt
import pywt.data

""" Given a particular type of wavelet function and level, we will return all wavelet functions
    and the scaling function required for a wavelet transform of that type and level. These wavelet
    functions will be as they appear in the first fully represented instance on the signal, and have
    a length divided by their stride size.

Inputs:
wavelet_name   string, the type of wavelet used for this particular wavelet transform
level          positive integer, the level of the wavelet transform (the number of scaling factors we'll compute)

Returns:
original_functions   tuple of numpy arrays, representing the wavelet functions "as they appear", with
                         order (approximation, detail n, ..., detail 1)
stride_size          size of stride for a level 1 wavelet function in this family (subsequent levels have a size of
                         2^(level-1) * base_stride)
                         
TODO: Can add helper function for zero padding at end to reduce total lines of code (OPTIONAL)
"""
def get_wavelet_functions(wavelet_name, level):
    
    wavelet = pywt.Wavelet(wavelet_name)
    
    original_functions = []
    
    # First we'll get the original scaling function
    phi, _, _ = wavelet.wavefun(level=level)
    phi       = np.trim_zeros(phi, trim='fb')
    phi       = phi / np.linalg.norm(phi)
    
    original_functions.append(phi)
    
    # Then we'll add the original wavelet functions:
    for i in range(level, 0, -1):
        
        _, psi, _ = wavelet.wavefun(level=i)
        # Often these functions contain excess 0's, which we need to remove:
        psi = np.trim_zeros(psi, trim='fb')
        psi = psi / np.linalg.norm(psi)
        original_functions.append(psi)
        
    # Now we have each of the wavelet functions we need, but we're not done yet. We also need to
    # get the stride size of the level 1 wavelet, and make sure the wavelets are 0-padded the way
    # we expect.
    
    # Stride size:
    blank_coefficients             = np.zeros((len(original_functions[-1]) * 10))
    detail1                        = np.zeros((len(original_functions[-1]) * 10))
    detail2                        = np.zeros((len(original_functions[-1]) * 10))
    detail1[len(detail1) // 2]     = 1.0
    detail2[len(detail1) // 2 + 1] = 1.0
    
    signal1 = np.abs(pywt.waverec((blank_coefficients, detail1), wavelet_name))
    signal2 = np.abs(pywt.waverec((blank_coefficients, detail2), wavelet_name))
    
    # We get the stride size by seeing how much of an offset makes the two signals equal:
    stride_size = -1
    
    for i in range(len(original_functions[0])):
        if (np.array_equal(signal1[:-i], signal2[i:])):
            stride_size = i
            break
    
    if stride_size == -1:
        print("Error: proper stride size cannot be found.")
        return 1
    
    # With the proper stride size found, we now pad the wavelet functions with 0's so the first entry to fully
    # appear on the signal lines up with the start of the signal, and it is divisible by its stride size.
    # First construct a dummy signal large enough to include the level i wavelet:
    blank_signal = np.zeros(len(original_functions[0]) * 2)
    blank_coeffs = pywt.wavedec(blank_signal, wavelet, level=level)

    # Note original_functions[-i] corresponds to the level-i wavelet
    for i in range(1, level+1):
        i_stride_size = stride_size * (2 ** (i-1))
        
        fully_represented_wavelet_index = (len(original_functions[-i]) // i_stride_size)
        blank_coeffs[-i][fully_represented_wavelet_index] = 1.0
        
        # This gives us the reconstructed wavelet as it appears through the wavelet transform:
        reconstructed_wavelet = pywt.waverec(blank_coeffs, wavelet)
        # It is possible this reconstructed wavelet may have an extra stride_size 0's in front, since we
        # chose fully_represented_wavelet_index to be sufficiently large to guarantee the wavelet appears
        # fully. We'll remove this if so:
        if (np.array_equal(reconstructed_wavelet[:i_stride_size], np.zeros(i_stride_size))):
            reconstructed_wavelet = reconstructed_wavelet[i_stride_size:]
            
        original_functions[-i] = reconstructed_wavelet[:(fully_represented_wavelet_index+1)*i_stride_size]
        # Once again, we added an extra entry to be safe. If the last stride size is all 0's, we can remove it:
        if (np.array_equal(original_functions[-i][-i_stride_size:], np.zeros(i_stride_size))):
            original_functions[-i] = original_functions[-i][:-i_stride_size]
        
        blank_coeffs[-i][fully_represented_wavelet_index] = 0
        
    # Repeat, with the scaling function:
    
    approx_stride_size = stride_size * (2 ** (level-1))
    fully_represented_scaling_index = (len(original_functions[0]) // approx_stride_size)
    blank_coeffs[0][fully_represented_scaling_index] = 1.0
    
    reconstructed_scaling = pywt.waverec(blank_coeffs, wavelet)
    
    if (np.array_equal(reconstructed_scaling[:approx_stride_size], np.zeros(approx_stride_size))):
        reconstructed_scaling = reconstructed_scaling[approx_stride_size:]
    
    original_functions[0] = reconstructed_scaling[:(fully_represented_scaling_index+1)*approx_stride_size]
    # Once again, we added an extra entry to be safe. If the last stride size is all 0's, we can remove it:
    if (np.array_equal(original_functions[0][-approx_stride_size:], np.zeros(approx_stride_size))):
        original_functions[0] = original_functions[0][:-approx_stride_size]
    
        
    return original_functions, stride_size


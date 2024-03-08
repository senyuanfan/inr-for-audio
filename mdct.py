"""
- mdct.py -- Computes reasonably fast MDCT/IMDCT using numpy FFT/IFFT
- from https://github.com/senyuanfan/TransientDetection/blob/main/mdct.py
"""


import numpy as np
from window import *
from scipy.fft import fft, ifft

sin = np.sin
cos = np.cos
exp = np.exp
pi = np.pi

def MDCT(data, a, b, isInverse=False):
    """
    Fast MDCT algorithm for window length a+b following pp. 141-143 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    (and where 2/N factor is included in forward transform instead of inverse)
    a: left half-window length
    b: right half-window length
    """

    N = a + b
    n0 = (b+1)/2
    n = np.arange(N)
    k = np.arange(N//2)
    if isInverse:
        factor = 2
        pre_twid = exp(1j*2*pi*k*n0/N)
        post_twid = exp(1j*pi*(n+n0)/N)
        twid_data = data * pre_twid
        fft_out = ifft(twid_data,N) * N
    else:
        factor = 2/N
        pre_twid = exp(-1j*pi*n/N)
        post_twid = exp(-1j*2*pi*n0*(k+0.5)/N)
        twid_data = data * pre_twid
        fft_out = fft(twid_data)[0:int(N//2)]

    return factor*(fft_out*post_twid).real



def IMDCT(data,a,b):

    return MDCT(data, a, b, True)

# perform MDCT on a long signal with framing and windowing
def STMDCT(data, N = 1024):
    # data should be an 1-D array of floats, bounded between 1 and -1, with length larger than N
    # define a and b for MDCT should both equal to half of N, and N should be 1024 or 2048
    halfN = N // 2
    pad_length = halfN - (len(data) % halfN)
    padded_data = np.pad(data, (0, pad_length), 'constant', constant_values=(0, 0))
    num_frames = len(data) // halfN

    mdct_coefficients = np.zeros((halfN, num_frames))

    for i in range(num_frames):
        # Frame extraction with windowing
        start_idx = i * halfN
        end_idx = start_idx + N
        frame = KBDWindow(padded_data[start_idx:end_idx])
        
        # MDCT
        mdct_coefficients[:, i] = MDCT(frame, halfN, halfN)
    
    # mdct_coefficients = mdct_coefficients.T

    return mdct_coefficients

def ISTMDCT(mdct_coefficients, N=1024):
    """
    Recover the original data from MDCT coefficients using the Overlap-Add method
    and applying a window function to each frame before overlap-adding.
    
    Parameters:
        mdct_coefficients (np.array): 2-D array of MDCT coefficients.
        N (int): Frame length, typically 1024 or 2048.
        window_func (function): A function that takes the frame length N and returns a window of the same length.
    
    Returns:
        np.array: Recovered original data.
    """

    # mdct_coefficients = mdct_coefficients.T

    halfN = N // 2
    num_frames = mdct_coefficients.shape[1]
    
    # Initialize the recovered signal with zeros
    recovered_length = halfN * num_frames + halfN
    recovered_data = np.zeros(recovered_length)
    
    
    for i in range(num_frames):
        # IMDCT
        frame = IMDCT(mdct_coefficients[:, i], halfN, halfN)
        
        # Apply window to the frame
        windowed_frame = KBDWindow(frame)
        
        # Overlap-Add
        start_idx = i * halfN
        end_idx = start_idx + N
        recovered_data[start_idx:end_idx] += windowed_frame
    
    # Trim the padding added during the forward transform
    recovered_data = recovered_data[:recovered_length - halfN]
    
    return recovered_data
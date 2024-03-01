"""
window.py -- Defines functions to window an array of data samples
"""

import numpy as np
from scipy.special import i0
# from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
# from mdct import MDCT, IMDCT

sin = np.sin
cos = np.cos
exp = np.exp
pi = np.pi

def SineWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray sine-windowed
    Sine window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    ### YOUR CODE STARTS HERE ###
    N = len(dataSampleArray)
    n = np.arange(N) # Vector from 0 to N-1, of length N
    return (sin(pi*(n+0.5)/N))*dataSampleArray


def HanningWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray Hanning-windowed
    Hann window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    ### YOUR CODE STARTS HERE ###
    N = len(dataSampleArray)
    n = np.arange(N)
    return 0.5*(1-cos(2*pi*(n+0.5)/N))*dataSampleArray


def KBDWindow(dataSampleArray,alpha=4.):
    """
    Returns a copy of the dataSampleArray KBD-windowed
    KBD window is defined following the KDB Window handout in the 
	Canvas Files/Assignments/HW3 folder
    """

    ### YOUR CODE STARTS HERE ###
    N = len(dataSampleArray)
    n = np.arange(N)
    n_a = n[0:N//2+1]
    n_b = n[N//2:N]
    KBW = i0(pi*alpha*np.sqrt(1-(((2*n_a+1)/(N/2+1))-1)**2))/i0(pi*alpha)
    KBW_sum = np.sum(KBW)
    n_a = n[0:N//2]
    KBW = i0(pi*alpha*np.sqrt(1-(((2*n_a+1)/(N/2+1))-1)**2))/i0(pi*alpha)
    KBDW_a = np.sqrt(np.cumsum(KBW)/KBW_sum)
    KBDW_b = KBDW_a[N-n_b-1]
    KBDW = np.concatenate([KBDW_a,KBDW_b])
    return KBDW*dataSampleArray 
    ### YOUR CODE ENDS HERE ###
def RECTWindow(dataSampleArray):
    return np.ones(len(dataSampleArray))*dataSampleArray*0.2
#-----------------------------------------------------------------------------

# N_BLOCK = 1024
# N_LONG = N_BLOCK
# N_SHORT = int(N_BLOCK/4) 
# N_TRANSITION = int((N_LONG + N_SHORT)/2)

# def getLongWindow(dataSampleArray): # use KBD type windows as long windows for faster dropoff
#     # long window: N_LONG = N_orig = 1024 point long, alpha = 4
#     # a0 = N_LONG/2, b0 = N_LONG/2
#     # n0 = b0/2 + 1/2 = (256 + 1)/2 = 257/2
#     a0 = N_LONG/2
#     b0 = N_LONG/2
#     n0 = b0/2 + 1/2
#     return KBDWindow(np.ones(N_LONG),alpha=4.) * dataSampleArray

# def getShortWindow(dataSampleArray): # use Sine type windows as short for better freq localization
#     # short window: N_SHORT = N_orig/4 = N_long/4 = 1024/4 = 256 point long, alpha = 6
#     # n0 = b0/2 + 1/2 = (64 + 1)/2 = 65/2
#     a0 = N_SHORT/2
#     b0 = N_SHORT/2 # for MDCT function
#     n0 = b0/2 + 1/2
#     return SineWindow(np.ones(N_SHORT)) * dataSampleArray

# def getTranStartWindow(dataSampleArray):
#     # transition start window: (long to short) asymmetric
#     # N_TRANSITION = 640 point long = N_LONG/2 + N_SHORT/2
#     # n0 = b0/2 + 1/2 = (64 + 1)/2 = 65/2
#     a0 = int(N_LONG/2) # 1024/2 = 512
#     b0 = int(N_SHORT/2) # 256/2 = 128
#     n0 = int(b0/2 + 1/2)
#     # left side Long
#     left = KBDWindow(np.ones(N_LONG),alpha=4.)[0:a0] * dataSampleArray[0:a0] 
#     # right side Short
#     right = SineWindow(np.ones(N_SHORT))[int(N_SHORT/2):] * dataSampleArray[a0:int(a0+b0)] 
#     #right = SineWindow(np.ones(N_SHORT))[a0:int(a0+b0)] * dataSampleArray[a0:int(a0+b0)] #512
#     return np.concatenate((left, right), axis=None)

# def getTranStopWindow(dataSampleArray):
#     # transition stop window: (short to long) asymmetric
#     # N_TRANSITION = 640 point long = N_LONG/2 + N_SHORT/2
#     a0 = int(N_SHORT/2) # 128
#     b0 = int(N_LONG/2) # 512
#     n0 = int(b0/2 + 1/2) #= (256 + 1)/2 = 257/2
#     # left side Short
#     left = SineWindow(np.ones(N_SHORT))[0:a0] * dataSampleArray[0:a0]
#     # right side Long
#     right = KBDWindow(np.ones(N_LONG), alpha=4.)[int(N_LONG/2):] * dataSampleArray[a0:int(a0+b0)] 
#     return np.concatenate((left, right), axis=None)
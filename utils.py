import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

import numpy as np
import matplotlib.pyplot as plt


from collections import OrderedDict
import matplotlib
import numpy.fft as fft
import scipy.stats as stats

import scipy.io.wavfile as wavfile
from scipy.signal import butter, filtfilt, decimate
import librosa

import mdct

from torchsummary import summary

def plotspec(signal, fs, title):
    # print('original signal shape', signal.shape)
    plt.specgram(signal, NFFT=2048, noverlap=512, Fs=fs, mode='magnitude', scale='dB')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')

def visualizer(data2d, savename, cmap='viridis'):
    '''
    STMDCT Visualizer, could also be used for magnitude spectrum
    '''
    stft_magnitude = np.abs(data2d)

    # Prepare the plot
    plt.figure(figsize=(10, 6))
    plt.imshow(stft_magnitude, origin='lower', aspect='auto', cmap=cmap)

    plt.colorbar(label='Magnitude')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(savename)

def hpfilter(data, cutoff, fs):
    order = 5
    b, a = butter(order, cutoff, btype='highpass', fs = fs)
    return filtfilt(b, a, data)

def lpfilter(data, cutoff, fs):
    order = 5
    b, a = butter(order, cutoff, btype='lowpass', fs = fs)
    return filtfilt(b, a, data)

def Thresh(f):
    """
    Returns the threshold in quiet measured in SPL at frequency f (in Hz)
    """
    f_new = f.clip(20,None)
    Af_db = ( 3.64 * ((f_new/1000)) ** (-0.8) ) - 6.5 * np.exp( -0.6 * ((f_new/1000) - 3.3) ** 2 ) + (10 ** (-3)) * ((f_new / 1000) ** 4)
    return Af_db

def Intensity(spl):
    """
    Returns the intensity  for SPL
    """
    # original
    # return 10 ** ((spl-96) / 10) # TO REPLACE WITH YOUR CODE

    # for MDCT magnitude
    return 10 ** ((spl-96) / 20)

def calculate_snr(original_signal, noisy_signal):
    # Ensure the signals are numpy arrays
    original_signal = np.asarray(original_signal)
    noisy_signal = np.asarray(noisy_signal)
    
    # Calculate the noise
    noise = noisy_signal - original_signal
    
    # Calculate the power of the original signal
    signal_power = np.mean(original_signal ** 2)
    
    # Calculate the power of the noise
    noise_power = np.mean(noise ** 2)
    
    # Calculate the SNR
    snr = signal_power / noise_power
    
    # Convert SNR to decibels
    snr_db = 10 * np.log10(snr)
    
    return snr_db

def get_coord(sidelen, dim=2, scale=1):
    """
    Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int
    """
    tensors = tuple(dim * [torch.linspace(-1*scale, 1*scale, steps=sidelen)])
    coord = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1) # added indexing='ij' to eliminate warning
    coord = coord.reshape(-1, dim)
    # print("mgrid shape:", mgrid.shape)
    return coord

class WaveformFitting(Dataset):
    def __init__(self, filename, duration, decimation=1):
        self.sample_rate, self.data = wavfile.read(filename)
        if(len(self.data.shape) > 1):
            self.data = self.data[:, 0]
        self.data = self.data.astype(np.float32)[0 : duration * self.sample_rate]

        self.original_sample_rate = self.sample_rate
        # cutoff = 8000 # in Hz
        if decimation > 1:
            q = int(decimation)
            # self.data = lpfilter(self.data, cutoff, self.sample_rate)
            self.data = decimate(self.data, q=q)
            self.sample_rate = self.sample_rate // q
        # else:
        #     # self.data = self.data
        #     self.data = hpfilter(self.data, cutoff, self.sample_rate)

        self.height = len(self.data)
        self.width = 1

        self.coord = get_coord(len(self.data), 1)

        print('Original sample rate: ', self.original_sample_rate)
        print('Training sample rate:', self.sample_rate)
        print('Training coord shape: ', self.coord.shape)

    def get_num_samples(self):
        return self.coord.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        amplitude = self.data
        scale = np.max(np.abs(amplitude))
        amplitude = (amplitude / scale)
        amplitude = torch.Tensor(amplitude).view(-1, 1)
        return self.coord, amplitude

class WaveformFittingExp(Dataset):
    def __init__(self, input_signal, input_fs, decimation=1, scale=1):
        '''
        Takes 1D input signal, perform normalization and calculate corresponding time coordinates (ranging from -1 to 1) for INR training
        '''
        self.original_sample_rate = input_fs
        if decimation > 1:
            q = int(decimation)
            # self.data = lpfilter(self.data, cutoff, self.sample_rate)
            self.data = decimate(input_signal, q=q)
            self.sample_rate = input_fs // q
        else:
            self.data = input_signal
            self.sample_rate = input_fs

        self.scale = np.max(np.abs(self.data))
        self.data = torch.Tensor(self.data/self.scale).view(-1, 1)

        self.height = len(self.data)
        self.width = 1

        self.coord = get_coord(self.height, self.width, scale=100)

        print('Training sample rate:', self.sample_rate)
        print('Training coord shape: ', self.coord.shape)

    def get_num_samples(self):
        return self.coord.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coord, self.data
    
class MultiWaveformFitting(Dataset):
    def __init__(self, filename, duration, num_channels, lp=False):
        self.sample_rate, self.data = wavfile.read(filename)
        self.data = self.data.astype(np.float32)[: duration*self.sample_rate, :num_channels]

        self.original_sample_rate = self.sample_rate

        if lp:
            q = 2
            decimated_channels = []
            for i in range(num_channels):
                # the decimate function will automatically apply low-pass anti-aliasing filter
                decimated_channel = decimate(self.data[:, i], q, ftype='fir', zero_phase=True)
                decimated_channels.append(decimated_channel)

            self.data = np.column_stack(decimated_channels)
            self.sample_rate = self.sample_rate // q
        
        self.height, self.width = self.data.shape

        print("height: ", self.height) # number of samples
        print("width: ", self.width) # number of channels

        print('waveform fitting sample rate:', self.sample_rate)
        
        height_norm = torch.linspace(-1, 1, steps = self.height)
        if num_channels == 1:
            width_norm = torch.linspace(0, 0, steps = self.width)
        else:
            width_norm = torch.linspace(-1, 1, steps = self.width)

        h_grid, w_grid = torch.meshgrid(height_norm, width_norm, indexing='ij')
        combined_grid = torch.stack((h_grid, w_grid), dim=-1)

        self.coords = combined_grid.reshape(self.height * self.width, -1)
        self.samples = self.data.reshape(-1, 1)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # amplitude = self.data
        # scale = np.max(np.abs(amplitude))
        # amplitude = (amplitude / scale)
        # amplitude = torch.Tensor(amplitude).view(-1, 1)
        return self.coords, self.samples
    
class FFTFitting(Dataset):
    def __init__(self, filename, duration, n_fft=1024, highpass=False):
        super().__init__()
        # Load the audio file
        self.sample_rate, self.data = wavfile.read(filename)

        if len(self.data.shape) > 1:
            self.data = self.data[:, 1]
        
        if highpass:
            self.data = hpfilter(self.data, 100, self.sample_rate)

        self.data = torch.from_numpy(self.data.astype(np.float32)[:duration * self.sample_rate]/np.max(np.abs(self.data))) # scaling and normalization is important

        # Generate the spectrogram
        # transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=2)
        # self.spectrogram = transform(torch.tensor(self.data))
        self.window = torch.hann_window(n_fft)
        self.stft_complex = torch.stft(self.data, n_fft = n_fft, window=self.window, return_complex = True)
        
        # self.stft_real = torch.view_as_real(self.stft_complex)
        self.stft_real = np.abs(self.stft_complex)

        # Convert the spectrogram to dB scale - this compresses the range of the data
        # self.spectrogram = torchaudio.transforms.AmplitudeToDB()(self.spectrogram)

        # Normalize the spectrogram to -1 to 1 range
        self.scale = self.stft_real.max()
        self.stft_real = self.stft_real / self.scale

        print("sample rate: ", self.sample_rate)
        print("max spectrogram: ", self.stft_real.max())
        print("min spectrogram: ", self.stft_real.min())

        # The shape of the spectrogram defines the sidelength
        # (height, width, dim) = self.stft_real.shape
        height, width = self.stft_real.shape
        self.dimensions = self.stft_real.shape

        print("height: ", height)
        print("width: ", width)
        # print("dim: ", dim)

        height_norm = torch.linspace(-1, 1, steps = height)
        width_norm = torch.linspace(-1, 1, steps = width)
        dim_norm = torch.tensor([-1, 1], dtype=torch.float32)

        # h_grid, w_grid, d_grid = torch.meshgrid(height_norm, width_norm, dim_norm, indexing='ij')

        # combined_grid = torch.stack((h_grid, w_grid, d_grid), dim=-1)

        # h_grid, w_grid, d_grid = torch.meshgrid(height_norm, width_norm, dim_norm, indexing='ij')
        h_grid, w_grid = torch.meshgrid(height_norm, width_norm, indexing='ij')
        
        # To achieve a shape of [513, 1723, 3], you can reshape or combine these grids appropriately.
        # Assuming you want each grid point to have a coordinate (h, w, d), where d is either -1 or 1:

        # Flatten the grids and stack them to form the desired shape
        combined_grid = torch.stack((h_grid, w_grid), dim=-1)

        # print("Final grid shape: ", combined_grid.shape)

        self.coords = combined_grid.reshape(height * width, -1)

        # the last dimension need to be changed for magnitude or complex representations
        self.pixels = self.stft_real.reshape(-1, 1)
        # self.pixels = self.stft_real

        print("coords shape: ", self.coords.shape)
        print("specs shape: ", self.pixels.shape)

    def __len__(self):
        return 1  # Only one item in this dataset

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
        return self.coords, self.pixels

class MDCTFitting(Dataset):
    def __init__(self, filename, duration, N=1024, highpass=False, takelog=False):
        super().__init__()
        # Load the audio file
        self.sample_rate, self.data = wavfile.read(filename)
        self.original_sample_rate = self.sample_rate
        if len(self.data.shape) > 1:
            self.data = self.data[:, 1]

        if highpass:
            self.data = hpfilter(self.data, 150, self.sample_rate)

        self.data = torch.from_numpy(self.data.astype(np.float32)[:duration * self.sample_rate]/np.max(np.abs(self.data))) # scaling and normalization is important

        # Generate the spectrogram
        # transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=2)
        # self.spectrogram = transform(torch.tensor(self.data)
        self.mdct = mdct.STMDCT(self.data, N=N).astype(np.float32)

        print("max mdct (w/o normalization): ", np.max(self.mdct))
        print("min mdct (w/o normalization): ", np.min(self.mdct))

        # Convert the spectrogram to dB scale - this compresses the range of the data
        # self.spectrogram = torchaudio.transforms.AmplitudeToDB()(self.spectrogram)

        self.shift = 0.0
        if takelog:
            aMinValue = 1e-8
            self.shift = np.abs(np.min(self.mdct)) + aMinValue
            self.mdct = self.mdct + self.shift

            print("max mdct (w/ shift): ", np.max(self.mdct))
            print("min mdct (w/ shift): ", np.min(self.mdct))
            # Turn output into log
            self.mdct = np.log(self.mdct)

        # Normalize the spectrogram to -1 to 1 range using mean and standard deviation
        self.mean = np.mean(self.mdct)
        # self.std = np.std(self.mdct)
        self.mdct = self.mdct - self.mean
        self.scale = np.max(np.abs(self.mdct))
        self.mdct = self.mdct / self.scale
        
        print("sample rate: ", self.sample_rate)
        print("mdct scale: ", self.scale)
        print("mdct mean: ", self.mean)
        print("max mdct (w/ normalization): ", np.max(self.mdct))
        print("min mdct (w/ normalization): ", np.min(self.mdct))

        # The shape of the spectrogram defines the sidelength
        # (height, width, dim) = self.stft_real.shape
        self.height, self.width = self.mdct.shape

        # calcualte hearing threshold mask, for attenuating loss function later
        # freqs =  np.arange(N)[0:N//2] * self.sample_rate/2/((N//2)-1)+1
        # threshold = Thresh(freqs)
        # threshold = threshold - min(threshold)
        # threshold = threshold.clip(None, 10)

        # reduction = (100 - threshold)/100 * 0.2 + 0.8
        # self.mask = np.tile(reduction, (self.width, 1)).T
        # self.mask = self.mask.reshape(1, -1, 1)
        # print("mask shape: ", self.mask.shape)
        # print("max mask: ", self.mask.max())
        # print("min mask: ", self.mask.min())

        print("height: ", self.height)
        print("width: ", self.width)
        # print("dim: ", dim)

        height_norm = torch.linspace(-1, 1, steps = self.height)
        width_norm = torch.linspace(-1, 1, steps = self.width)

        # h_grid, w_grid, d_grid = torch.meshgrid(height_norm, width_norm, dim_norm, indexing='ij')

        # combined_grid = torch.stack((h_grid, w_grid, d_grid), dim=-1)

        # h_grid, w_grid, d_grid = torch.meshgrid(height_norm, width_norm, dim_norm, indexing='ij')
        h_grid, w_grid = torch.meshgrid(height_norm, width_norm, indexing='ij')
        
        # To achieve a shape of [513, 1723, 3], you can reshape or combine these grids appropriately.
        # Assuming you want each grid point to have a coordinate (h, w, d), where d is either -1 or 1:

        # Flatten the grids and stack them to form the desired shape
        combined_grid = torch.stack((h_grid, w_grid), dim=-1)

        # print("Final grid shape: ", combined_grid.shape)

        self.coords = combined_grid.reshape(self.height * self.width, -1)

        # the last dimension need to be changed for magnitude or complex representations
        self.pixels = self.mdct.reshape(-1, 1)

        print("coords shape: ", self.coords.shape)
        print("specs shape: ", self.pixels.shape)

    def __len__(self):
        return 1  # Only one item in this dataset

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
        return self.coords, self.pixels

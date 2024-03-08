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
from scipy.signal import butter, filtfilt

import auraloss
import mdct

from torchsummary import summary


class ReLU(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()
        
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features, nn.LeakyReLU(0.01)))

        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features, nn.LeakyReLU(0.01)))

        self.net.append(nn.Linear(hidden_features, out_features)) 

        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords      
    
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))
                

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1) # added indexing='ij' to eliminate warning
    mgrid = mgrid.reshape(-1, dim)
    print("mgrid shape:", mgrid.shape)
    return mgrid
    
class WaveformFitting(Dataset):
    def __init__(self, filename, duration, highpass = False):
        self.sample_rate, self.data = wavfile.read(filename)
        if(len(self.data.shape) > 1):
            self.data = self.data[:, 1]
        self.data = self.data.astype(np.float32)[0 : duration * self.sample_rate]
        if highpass:
            self.data = hpfilter(self.data, 100, self.sample_rate)
        self.timepoints = get_mgrid(len(self.data), 1)
        print("timepoints shape: ", self.timepoints.shape)

    def get_num_samples(self):
        return self.timepoints.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        amplitude = self.data
        scale = np.max(np.abs(amplitude))
        amplitude = (amplitude / scale)
        amplitude = torch.Tensor(amplitude).view(-1, 1)
        return self.timepoints, amplitude

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
    def __init__(self, filename, duration, n_fft=1024, highpass = False):
        super().__init__()
        # Load the audio file
        self.sample_rate, self.data = wavfile.read(filename)

        if len(self.data.shape) > 1:
            self.data = self.data[:, 1]

        if highpass:
            self.data = hpfilter(self.data, 150, self.sample_rate)

        self.data = torch.from_numpy(self.data.astype(np.float32)[:duration * self.sample_rate]/np.max(np.abs(self.data))) # scaling and normalization is important

        # Generate the spectrogram
        # transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=2)
        # self.spectrogram = transform(torch.tensor(self.data)
        self.mdct = mdct.STMDCT(self.data, N=n_fft).astype(np.float32)

        # Convert the spectrogram to dB scale - this compresses the range of the data
        # self.spectrogram = torchaudio.transforms.AmplitudeToDB()(self.spectrogram)

        # Normalize the spectrogram to -1 to 1 range
        self.scale = np.max(np.abs(self.mdct))
        self.scale = 1.0
        self.mdct = self.mdct / self.scale

        print("sample rate: ", self.sample_rate)
        print("max spectrogram: ", self.mdct.max())
        print("min spectrogram: ", self.mdct.min())

        # The shape of the spectrogram defines the sidelength
        # (height, width, dim) = self.stft_real.shape
        height, width = self.mdct.shape
        self.dimensions = self.mdct.shape

        print("height: ", height)
        print("width: ", width)
        # print("dim: ", dim)

        height_norm = torch.linspace(-1, 1, steps = height)
        width_norm = torch.linspace(-1, 1, steps = width)

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
        self.pixels = self.mdct.reshape(-1, 1)

        print("coords shape: ", self.coords.shape)
        print("specs shape: ", self.pixels.shape)

    def __len__(self):
        return 1  # Only one item in this dataset

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
        return self.coords, self.pixels

def visualizer(data2d, savename, cmap='viridis'):

    print("Data: ", data2d.shape)

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
    order = 7
    b, a = butter(order, cutoff, btype='highpass', fs = fs)
    return filtfilt(b, a, data)

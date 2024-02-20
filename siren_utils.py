import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time

from collections import OrderedDict
import matplotlib
import numpy.fft as fft
import scipy.stats as stats

import scipy.io.wavfile as wavfile
import io
from IPython.display import Audio

import auraloss

from torchsummary import summary

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

def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())        
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)
        
        print("pixels shape: ", self.pixels.shape)
        print("coords shape: ", self.coords.shape)
        

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels
    
class AudioFile(Dataset):
    def __init__(self, filename, duration):
        self.rate, self.data = wavfile.read(filename)
        if(len(self.data.shape) > 1):
            self.data = self.data[:, 1]
        self.data = self.data.astype(np.float32)[0 : duration * self.rate]
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

class SpectrogramFitting(Dataset):
    def __init__(self, filename, duration, n_fft=1024):
        super().__init__()
        # Load the audio file
        self.rate, self.data = wavfile.read(filename)
        if len(self.data.shape) > 1:
            self.data = self.data.mean(axis=1)  # Convert to mono if stereo
        self.data = torch.from_numpy(self.data.astype(np.float32)[:duration * self.rate])

        # Generate the spectrogram
        # transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=2)
        # self.spectrogram = transform(torch.tensor(self.data))
        stft = torch.stft(self.data, n_fft = n_fft, window=torch.hann_window(n_fft), return_complex = True)
        self.spectrogram = torch.view_as_real(stft)

        # Convert the spectrogram to dB scale - this compresses the range of the data
        # self.spectrogram = torchaudio.transforms.AmplitudeToDB()(self.spectrogram)

        # Normalize the spectrogram to -1 to 1 range
        self.spectrogram = self.spectrogram / self.spectrogram.max()
        self.scalar = self.spectrogram.max()

        print("spectrogram dimension: ", self.spectrogram.shape)
        print("max spectrogram: ", self.spectrogram.max())
        print("min spectrogram: ", self.spectrogram.min())


        # The shape of the spectrogram defines the sidelength
        (height, width, dim) = self.spectrogram.shape

        print("height: ", height)
        print("width: ", width)
        print("dim: ", dim)

        height_norm = torch.linspace(-1, 1, steps = height)
        width_norm = torch.linspace(-1, 1, steps = width)
        dim_norm = torch.tensor([-1, 1], dtype=torch.float32)

        # h_grid, w_grid, d_grid = torch.meshgrid(height_norm, width_norm, dim_norm, indexing='ij')

        # combined_grid = torch.stack((h_grid, w_grid, d_grid), dim=-1)

        h_grid, w_grid, d_grid = torch.meshgrid(height_norm, width_norm, dim_norm, indexing='ij')

        # To achieve a shape of [513, 1723, 3], you can reshape or combine these grids appropriately.
        # Assuming you want each grid point to have a coordinate (h, w, d), where d is either -1 or 1:

        # Flatten the grids and stack them to form the desired shape
        combined_grid = torch.stack((h_grid, w_grid, d_grid), dim=-1)


        print("Final grid shape: ", combined_grid.shape)

        self.coords = combined_grid.reshape(height * width * dim, -1)
        self.pixels = self.spectrogram.reshape(-1, 1)

        print("coords shape: ", self.coords.shape)
        print("specs shape: ", self.pixels.shape)

        # # Prepare coordinates
        # self.coords = get_mgrid(height * width, 2).reshape(height, width, 2)
        # self.coords = self.coords.reshape(-1, 2)  # Flatten the grid for input into the model
        
        # # Prepare pixel values
        # self.pixels = self.spectrogram.view(-1, 1)  # Flatten the spectrogram for model input

    def __len__(self):
        return 1  # Only one item in this dataset

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
        return self.coords, self.pixels

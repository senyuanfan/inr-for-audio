import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np

from collections import OrderedDict

import scipy.io.wavfile as wavfile


class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''

    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True, num_frequencies=None, scale=2):
        super().__init__()

        self.in_features = in_features
        self.scale = scale
        self.sidelength = sidelength
        if num_frequencies == None:
            if self.in_features == 3:
                self.num_frequencies = 10
            elif self.in_features == 2:
                assert sidelength is not None
                if isinstance(sidelength, int):
                    sidelength = (sidelength, sidelength)
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
            elif self.in_features == 1:
                assert fn_samples is not None
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
        else:
            self.num_frequencies = num_frequencies
        # self.frequencies_per_axis = (num_frequencies * np.array(sidelength)) // max(sidelength)
        self.out_dim = in_features + in_features * 2 * self.num_frequencies  # (sum(self.frequencies_per_axis))

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(np.floor(np.log2(nyquist_rate)))

    def forward(self, coords):
        coords_pos_enc = coords
        for i in range(self.num_frequencies):

            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((self.scale ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((self.scale ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc
    
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



class ScaledSineLayer(nn.Module):
    # A customized sine layer that sets the omega_0 liearly scaled for each neuron
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.out_features = out_features
        
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

    ## naive implementation
    # def forward(self, input):
    #     if self.is_first:
    #         # print('input shape:', input.shape)
    #         linear_output = self.linear(input)
    #         # print('linear output shape:', linear_output.shape)
    #         output = torch.zeros_like(linear_output)
    #         for i in range(self.out_features):
    #             omega_scale = i / self.out_features * self.omega_0
    #             # print('activation shape', activation.shape)
    #             output[:, :, i] = torch.sin(omega_scale * linear_output[:, :, i])
    #     else:
    #         output = torch.sin(self.omega_0 * self.linear(input))
        
    #     return output

    ## vectorized implementation
    def forward(self, input):
        if self.is_first:
            linear_output = self.linear(input)
            # Create a tensor of omega_scale values
            omega_scales = torch.linspace(0, self.omega_0, steps=self.out_features, device=input.device, dtype=input.dtype)
            omega_scales = omega_scales * (1 / self.out_features)
            # Expand omega_scales to match the shape of linear_output for broadcasting
            omega_scales = omega_scales.view(1, 1, self.out_features)
            # Vectorized operation
            output = torch.sin(omega_scales * linear_output)
        else:
            output = torch.sin(self.omega_0 * self.linear(input))
        
        return output
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
class Snake(nn.Module):
    '''         
    Implementation of the serpentine-like sine-based periodic activation function:
    .. math::
         Snake_a := x + \frac{1}{a} sin^2(ax) = x - \frac{1}{2a}cos{2ax} + \frac{1}{2a}
    This activation function is able to better extrapolate to previously unseen data,
    especially in the case of learning periodic functions

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
        
    Parameters:
        - a - trainable parameter
    
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
        
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, a=None, trainable=True):
        '''
        Initialization.
        Args:
            in_features: shape of the input
            a: trainable parameter
            trainable: sets `a` as a trainable parameter
            
            `a` is initialized to 1 by default, higher values = higher-frequency, 
            5-50 is a good starting point if you already think your data is periodic, 
            consider starting lower e.g. 0.5 if you think not, but don't worry, 
            `a` will be trained along with the rest of your model
        '''
        super(Snake,self).__init__()
        self.in_features = in_features if isinstance(in_features, list) else [in_features]

        # Initialize `a`
        if a is not None:
            self.a = nn.Parameter(torch.ones(self.in_features) * a) # create a tensor out of alpha
        else:            
            m = torch.distributions.exponential.Exponential(torch.tensor([0.1]))
            self.a = nn.Parameter((m.rsample(self.in_features)).squeeze()) # random init = mix of frequencies

        self.a.requiresGrad = trainable # set the training of `a` to true

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a* sin^2 (xa)
        '''
        return  x + (1.0/self.a) * torch.pow(torch.sin(x * self.a), 2)
    
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
        # return output, coords   
        return output     

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


class SirenWithSnakeTanh(nn.Module):
    '''
    MLP with Snake and Tanh activations
    '''
    def __init__(self, in_features, out_features, hidden_features, num_sine, num_snake, num_tanh, first_linear=False, last_linear=True, 
                 first_omega_0=30, hidden_omega_0=30., a_initial=50, num_freq=32, scale=2.0):
        super().__init__()
        
        self.net = []


        '''
        Positional encoding
        '''
        if num_freq is not None:
            pos_out_dim = 2 * num_freq + 1
            is_first = False
            self.positional_encoding = PosEncodingNeRF(in_features=in_features,
                                                        sidelength=None,
                                                        fn_samples=None,
                                                        use_nyquist=True,
                                                        num_frequencies=num_freq,
                                                        scale=scale)
        else:
            self.positional_encoding = nn.Identity()
            pos_out_dim = in_features
            is_first = True
        '''
        First layer need to be sine for waveform
        '''
        if first_linear:
            fc = nn.Linear(pos_out_dim, hidden_features)
            snake = Snake(hidden_features, a=50)
            self.net.append(fc)
            self.net.append(snake)
        else:
            self.net.append(SineLayer(pos_out_dim, hidden_features, is_first=True, omega_0=first_omega_0))
       

               
        for i in range(num_sine):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))
                
        for i in range(num_snake):
            fc = nn.Linear(hidden_features, hidden_features)
            snake = Snake(hidden_features, a=a_initial)
            # fc.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
            #                                   np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(fc)
            self.net.append(snake)
      
        for i in range(num_tanh):
            fc = nn.Linear(hidden_features, hidden_features)
            tanh = nn.Tanh()
            # fc.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
            #                                   np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(fc)
            self.net.append(tanh)

        if last_linear:
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
        pos_coords = self.positional_encoding(coords)
        output = self.net(pos_coords)
        # return output, coords   
        return output     

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
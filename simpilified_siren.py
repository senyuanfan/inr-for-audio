import torch
import torchaudio
import torch.nn as nn
import skimage
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchsummary import summary

class SineLayer(nn.Module):

    def __init__(self, w0):
        super(SineLayer, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class Siren(nn.Module):
    def __init__(self, w0=30, in_dim=2, hidden_dim=256, out_dim=1):
        super(Siren, self).__init__()

        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), SineLayer(w0),
                                 nn.Linear(hidden_dim, hidden_dim), SineLayer(w0),
                                 nn.Linear(hidden_dim, hidden_dim), SineLayer(w0),
                                 nn.Linear(hidden_dim, hidden_dim), SineLayer(w0),
                                 nn.Linear(hidden_dim, out_dim))

        # Init weights
        with torch.no_grad():
            self.net[0].weight.uniform_(-1. / in_dim, 1. / in_dim)
            self.net[2].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                        np.sqrt(6. / hidden_dim) / w0)
            self.net[4].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                        np.sqrt(6. / hidden_dim) / w0)
            self.net[6].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                        np.sqrt(6. / hidden_dim) / w0)
            self.net[8].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                        np.sqrt(6. / hidden_dim) / w0)

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=256, out_dim=1):
        super(MLP, self).__init__()

        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.net(x)



def load_audio_waveform(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    # get the left channel of the waveform, and only the first 5 seconds
    waveform = waveform[0,0:sample_rate*10]
    # Normalize waveform to [-1, 1]
    waveform = waveform / waveform.abs().max()
    return waveform, sample_rate

def generate_sample_coordinates(waveform_length, sample_rate):
    timestamps = torch.linspace(0, waveform_length / sample_rate, steps=waveform_length)
    # print("timestamps size", timestamps.shape)
    return timestamps.reshape(-1, 1)


def train(model, optimizer, audio_waveform, sample_coordinates, nb_epochs=15000):
    psnr = []  # Initialize a list to store PSNR values for each epoch

    for _ in tqdm(range(nb_epochs)):
        model_output = model(sample_coordinates)  # Forward pass
        loss = ((model_output - audio_waveform) ** 2).mean()  # Compute mean squared error loss

        # Ensure max_pixel is a tensor for consistent operation with torch.log10
        max_pixel = torch.tensor(1.0).to(audio_waveform.device)  
        # Adjust the PSNR calculation to use tensor-based operation
        psnr_value = 20 * torch.log10(max_pixel) - 10 * torch.log10(loss)
        psnr.append(psnr_value.item())

        optimizer.zero_grad()  # Zero gradients
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters

    return psnr, model_output


if __name__ == "__main__":
    device = 'cuda'

    siren = Siren(w0=30, in_dim=1, hidden_dim=256, out_dim=1).to(device)
    mlp = MLP(in_dim=1, hidden_dim=256, out_dim=1).to(device)
    summary(siren)
    # Load and preprocess audio
    audio_waveform, sample_rate = load_audio_waveform('castanets.wav')
    audio_waveform = audio_waveform.to(device)

    # Flatten the waveform for training
    audio_waveform = audio_waveform.reshape(-1, 1)

    print("audio waveform shape", audio_waveform.shape)
    
    # Generate sample coordinates
    sample_coordinates = generate_sample_coordinates(audio_waveform.shape[0], sample_rate).to(device)

    print("sample coordinates shape", sample_coordinates.shape)
    print(sample_coordinates[0])
    print(sample_coordinates[1])

    # Training
    for model in [siren]:
        optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
        psnr, model_output = train(model, optim, audio_waveform, sample_coordinates, nb_epochs=15000)

        # Visualization or evaluation logic here
        # Note: PSNR plotting might be replaced with a more suitable metric for audio
    
    # Reshape the model_output to [1, samples] for saving as mono audio
    print("model output shape", model_output.shape)
    model_output_reshaped = model_output.detach().cpu().reshape(1, -1)

    # Save the final model output to a .wav file
    output_file_path = 'model_output.wav'
    torchaudio.save(output_file_path, model_output_reshaped, sample_rate)


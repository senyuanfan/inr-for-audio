import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchaudio
from torchaudio import transforms
from scipy.io import wavfile
import matplotlib.pyplot as plt
from siren_utils import Siren, AudioFile # Assuming these are defined in siren_utils or relevant modules
import auraloss
from torchsummary import summary

def train(inst, num_hidden_features=256, num_hidden_layers=4, omega=5000, total_steps=100, learning_rate=1e-4, lr_decay_step=50, lr_decay_gamma=0.5):
    filename = f'data/{inst}.wav'
    sample_rate, _ = wavfile.read(filename)
    input_audio = AudioFile(filename, duration=10) # Hardcoded input length as 10 seconds

    dataloader = DataLoader(input_audio, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

    model = Siren(in_features=1, out_features=1, hidden_features=num_hidden_features, 
                  hidden_layers=num_hidden_layers, first_omega_0=omega, outermost_linear=True)
    model.cuda()
    summary(model)

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=lr_decay_step, gamma=lr_decay_gamma)
    loss_function = nn.MSELoss()

    losses = []
    for step in range(total_steps):
        model_input, ground_truth = next(iter(dataloader))
        model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

        model_output, coords = model(model_input)    
        loss = loss_function(model_output, ground_truth)
        losses.append(loss.item())

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()

    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    savename = f'results/{inst}-{num_hidden_features}-{num_hidden_layers}-{omega}.png'
    plt.savefig(savename)
    
    final_model_output, _ = model(model_input)
    torchaudio.save(f'{savename[:-4]}.wav', final_model_output.cpu().detach().reshape(1, -1), sample_rate)

if __name__ == "__main__":
    configurations = [
        {'inst': 'castanets', 'num_hidden_features': 512, 'num_hidden_layers': 5, 'omega': 10000, 'total_steps': 5000},
        {'inst': 'quartet', 'num_hidden_features': 512, 'num_hidden_layers': 5, 'omega': 10000, 'total_steps': 5000},
        {'inst': 'violin', 'num_hidden_features': 512, 'num_hidden_layers': 5, 'omega': 10000, 'total_steps': 5000},
        {'inst': 'oboe', 'num_hidden_features': 512, 'num_hidden_layers': 5, 'omega': 10000, 'total_steps': 5000}
    ]

    for config in configurations:
        train(**config)

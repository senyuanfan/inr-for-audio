import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from scipy.io import wavfile
import matplotlib.pyplot as plt
from siren_utils import * # Assuming these are defined in siren_utils or relevant modules
import auraloss
from torchsummary import summary
import time
from tqdm import tqdm
import numpy as np

def train_wave(inst, num_hidden_features=256, num_hidden_layers=5, omega=22000, total_steps=10000, learning_rate=1e-4, alpha=0.0):
    start_time = time.time()
    method = 'wave'

    filename = f'data/{inst}.wav'
    # sample_rate, _ = wavfile.read(filename)
    input_audio = WaveformFitting(filename, duration=10) # Hardcoded input length as 10 seconds

    dataloader = DataLoader(input_audio, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

    model = Siren(in_features=1, out_features=1, hidden_features=num_hidden_features, 
                  hidden_layers=num_hidden_layers, first_omega_0=omega, outermost_linear=True)
    model.cuda()
    summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    mae = nn.L1Loss()
    mse = nn.MSELoss()
    mrstft = auraloss.freq.MultiResolutionSTFTLoss()

    losses = []
    lrs = []

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
    for step in tqdm(range(total_steps), desc = "Training Progress"):
        
        model_output, coords = model(model_input)
        mae_loss = mae(model_output, ground_truth)
        mse_loss = mse(model_output, ground_truth)
        # mrstft_loss = mrstft(model_output.view(1, 1, -1), ground_truth.view(1, 1, -1))
        loss = (1 - alpha) * mse_loss + alpha * mae_loss
        losses.append(10 * np.log10(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        current_lr = scheduler.get_last_lr()
        lrs.append(10 * np.log10(current_lr))

    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss (dB)")
    plt.xlim([0, total_steps])
    savename = f'results/loss-{inst}-{method}-{num_hidden_features}-{num_hidden_layers}-{omega}-{total_steps}'
    plt.savefig(savename + '.png')

    plt.figure()
    plt.plot(lrs)
    plt.title("Learning Rate")
    plt.xlabel("Step")
    plt.xlim([0, total_steps])
    plt.ylabel("Learning Rate (dB)")
    savename = f'results/lr-{inst}-{method}-{num_hidden_features}-{num_hidden_layers}-{omega}-{total_steps}'
    plt.savefig(savename + '.png')
        
    savename = f'results/{inst}-{num_hidden_features}-{num_hidden_layers}-{omega}-{total_steps}'
    final_model_output, _ = model(model_input)

    signal_recovered = final_model_output.cpu().detach()
    print("signal recovered shape: ", signal_recovered.shape)
    print("signal recovered dtype: ", signal_recovered.dtype)
    print("signal max: ", np.max(signal_recovered.numpy()))
    print("signal min: ", np.min(signal_recovered.numpy()))
    torchaudio.save(savename + '.wav', signal_recovered.reshape(1, -1), input_audio.sample_rate)

    end_time = time.time()
    print("Time Elapsed: ", end_time-start_time)

if __name__ == "__main__":
    configurations = [
        # {'inst': 'castanets', 'num_hidden_features': 256, 'num_hidden_layers': 6, 'omega': 22000, 'total_steps': 10000, 'alpha':0.0},
        # {'inst': 'violin', 'num_hidden_features': 256, 'num_hidden_layers': 6, 'omega': 22000, 'total_steps': 10000, 'alpha':0.0},

        # {'inst': 'castanets', 'num_hidden_features': 256, 'num_hidden_layers': 7, 'omega': 22000, 'total_steps': 10000, 'alpha':0.0},
        # {'inst': 'violin', 'num_hidden_features': 256, 'num_hidden_layers': 7, 'omega': 22000, 'total_steps': 10000, 'alpha':0.0},

        # {'inst': 'castanets', 'num_hidden_features': 512, 'num_hidden_layers': 6, 'omega': 22000, 'total_steps': 10000, 'alpha':0.0},
        # {'inst': 'violin', 'num_hidden_features': 512, 'num_hidden_layers': 6, 'omega': 22000, 'total_steps': 10000, 'alpha':0.0},

        {'inst': 'castanets', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 44100, 'total_steps': 5000, 'alpha':0.0},
        # {'inst': 'violin', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 22000, 'total_steps': 20000, 'alpha':0.0},
        # {'inst': 'oboe', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 22000, 'total_steps': 20000, 'alpha':0.0},
        # {'inst': 'quartet', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 22000, 'total_steps': 20000, 'alpha':0.0},
        # {'inst': 'glockenspiel', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 22000, 'total_steps': 20000, 'alpha':0.0},
        # {'inst': 'harpsichord', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 22000, 'total_steps': 20000, 'alpha':0.0},
        # {'inst': 'oboe', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 22000, 'total_steps': 20000, 'alpha':0.0},
        # {'inst': 'spgm', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 22000, 'total_steps': 20000, 'alpha':0.0},
        
    ]

    for config in configurations:
        train_wave(**config)

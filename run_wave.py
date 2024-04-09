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


import copy
import loss_landscapes
import loss_landscapes.metrics

def train_wave(inst:str, tag:str, num_hidden_features=256, num_hidden_layers=5, omega=22000, total_steps=10000, learning_rate=1e-4, alpha=0.0, load_checkpoint=False, save_checkpoint=False, visualization=False):
    method = 'wave'
    

    filename = f'data/{inst}.wav'
    # sample_rate, _ = wavfile.read(filename)
    input_audio = WaveformFitting(filename, duration=20, highpass=False) # Hardcoded input length as 10 seconds

    dataloader = DataLoader(input_audio, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

    model = Siren(in_features=1, out_features=1, hidden_features=num_hidden_features, 
                  hidden_layers=num_hidden_layers, first_omega_0=omega, outermost_linear=True)
    model.cuda()
    # summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=100, min_lr=1e-8)
    mae = nn.L1Loss()
    mse = nn.MSELoss()
    mrstft = auraloss.freq.MultiResolutionSTFTLoss()

    losses = []
    lrs = []

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    start_time = time.time()
    best_loss = 10
    for step in tqdm(range(total_steps), desc = "Training Progress"):
        
        model_output = model(model_input)
        mae_loss = mae(model_output, ground_truth)
        mse_loss = mse(model_output, ground_truth)
        # mrstft_loss = mrstft(model_output.view(1, 1, -1), ground_truth.view(1, 1, -1))
        loss = (1 - alpha) * mse_loss + alpha * mae_loss
        if loss.item() < best_loss:
            best_loss = loss.item()
        losses.append(10 * np.log10(loss.item()))
        best_model = model
        best_iter = step

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        current_lr = scheduler.get_last_lr()
        lrs.append(10 * np.log10(current_lr))
    
    ### plot the latest loss landscape
    if visualization:
        STEPS = 50
        metric = loss_landscapes.metrics.Loss(mse, model_input.cpu(), ground_truth.cpu())

        model_final = copy.deepcopy(best_model)
        loss_data_fin = loss_landscapes.random_plane(model_final.cpu(), metric, distance=5, steps=STEPS, normalization='filter', deepcopy_model=True)

        plt.figure()
        ax = plt.axes(projection='3d')
        X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
        Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
        ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title('Surface Plot of Loss Landscape')
        savename = f'results/[scape]{inst}-{method}-{tag}-{total_steps}'
        plt.savefig(savename + '.png')

    end_time = time.time()

    ### save model
    if save_checkpoint:
        savename = f'results/[model]{inst}-{method}-{tag}-{total_steps}'
        torch.save(best_model.state_dict(), savename+'.pt')
    ### plot loss and learning rate history

    plt.figure()
    plt.plot(losses)
    plt.title(f'Training Loss, Best Iteration: {best_iter}, Total time: {(end_time-start_time)/60:.2f} min')
    plt.xlabel("Step")
    plt.ylabel("Loss (dB)")
    plt.xlim([0, total_steps])
    savename = f'results/[loss]{inst}-{method}-{tag}-{total_steps}'
    plt.savefig(savename + '.png')

    plt.figure()
    plt.plot(lrs)
    plt.title("Learning Rate")
    plt.xlabel("Step")
    plt.xlim([0, total_steps])
    plt.ylabel("Learning Rate (dB)")
    savename = f'results/[lr]{inst}-{method}-{tag}-{total_steps}'
    plt.savefig(savename + '.png')
        
    final_model_output = best_model(model_input)

    signal_recovered = final_model_output.cpu().detach()
    print("signal recovered shape: ", signal_recovered.shape)
    print("signal recovered dtype: ", signal_recovered.dtype)
    print("signal max: ", np.max(signal_recovered.numpy()))
    print("signal min: ", np.min(signal_recovered.numpy()))
    savename = f'results/[audio]{inst}-{method}-{tag}-{total_steps}'
    # wavfile.write(savename+'.wav', input_spec.sample_rate, signal_recovered)
    torchaudio.save(savename+'.wav', signal_recovered.reshape(1, -1), input_audio.sample_rate)

    end_time = time.time()
    print("Time Elapsed: ", end_time-start_time)

if __name__ == "__main__":
    
    tag = 'rlrop-e408100'
    configurations = [
        # {'inst': 'castanets', 'num_hidden_features': 256, 'num_hidden_layers': 6, 'omega': 22000, 'total_steps': 10000, 'alpha':0.0},
        # {'inst': 'violin', 'num_hidden_features': 256, 'num_hidden_layers': 6, 'omega': 22000, 'total_steps': 10000, 'alpha':0.0},

        # {'inst': 'castanets', 'num_hidden_features': 256, 'num_hidden_layers': 7, 'omega': 22000, 'total_steps': 10000, 'alpha':0.0},
        # {'inst': 'violin', 'num_hidden_features': 256, 'num_hidden_layers': 7, 'omega': 22000, 'total_steps': 10000, 'alpha':0.0},

        # {'inst': 'castanets', 'num_hidden_features': 512, 'num_hidden_layers': 6, 'omega': 22000, 'total_steps': 10000, 'alpha':0.0},
        # {'inst': 'violin', 'num_hidden_features': 512, 'num_hidden_layers': 6, 'omega': 22000, 'total_steps': 10000, 'alpha':0.0},

        # {'inst': 'violin', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 22000, 'total_steps': 20000, 'alpha':0.0},
        # {'inst': 'oboe', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 22000, 'total_steps': 20000, 'alpha':0.0},
        # {'inst': 'quartet', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 22000, 'total_steps': 20000, 'alpha':0.0},
        # {'inst': 'glockenspiel', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 22000, 'total_steps': 20000, 'alpha':0.0},
        # {'inst': 'harpsichord', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 22000, 'total_steps': 20000, 'alpha':0.0},
        # {'inst': 'oboe', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 22000, 'total_steps': 20000, 'alpha':0.0},
        # {'inst': 'spgm', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 22000, 'total_steps': 20000, 'alpha':0.0},
        {'inst': 'violin', 'tag': tag+'-scale', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 22000, 'total_steps': 100, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':True},
        # {'inst': 'castanets', 'tag': tag+'4l', 'num_hidden_features': 256, 'num_hidden_layers': 4, 'omega': 22000, 'total_steps': 40000, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':True},

        # {'inst': 'violin', 'tag': tag+'3l', 'num_hidden_features': 256, 'num_hidden_layers': 3, 'omega': 22000, 'total_steps': 40000, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':True},
        # {'inst': 'castanets', 'tag': tag+'3l', 'num_hidden_features': 256, 'num_hidden_layers': 3, 'omega': 22000, 'total_steps': 40000, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':True},
        # {'inst': 'oboe', 'tag': tag, 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 22000, 'total_steps': 40000, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':False},
        # {'inst': 'glockenspiel', 'tag': tag, 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 22000, 'total_steps': 40000, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':False}
    ]

    for config in configurations:
        train_wave(**config)

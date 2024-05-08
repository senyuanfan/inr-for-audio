import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from scipy.io import wavfile
import matplotlib.pyplot as plt
from utils import * # Assuming these are defined in siren_utils or relevant modules
from models import *
import auraloss
from torchsummary import summary
import time
from tqdm import tqdm
import numpy as np

import copy
import loss_landscapes
import loss_landscapes.metrics

def train(inst:str, tag:str, num_hidden_features=256, num_hidden_layers=5, omega=300, total_steps=10000, learning_rate=1e-6, alpha=0.0, load_checkpoint=False, save_checkpoint=False, visualization=False, mask=False):
    method = "mdct"


    filename = f'data/{inst}.wav'
    input_spec = MDCTFitting(filename, duration=10, highpass=False)
    height, width = input_spec.dimensions

    dataloader = DataLoader(input_spec, shuffle=True, batch_size=1, pin_memory=True, num_workers=4)

    model_path = 'model_checkpoint.pth'
    optimizer_path = 'optimizer_checkpoint.pth'
    if load_checkpoint:
        model = torch.load(model_path)
        # optimizer = torch.load(optimizer_path)
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    else:
        model = Siren(in_features=2, out_features=1, hidden_features=num_hidden_features, 
                 hidden_layers=num_hidden_layers, first_omega_0=omega, outermost_linear=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2000, min_lr=1e-6)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma = 0.1)
    # model = ReLU(in_features=2, out_features=2, hidden_features=num_hidden_features, hidden_layers=num_hidden_layers)
    
    model.cuda()
    # summary(model)
   
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    huber = nn.HuberLoss(reduction='mean', delta=1.0)
    mrstft = auraloss.freq.MultiResolutionSTFTLoss()

    losses = []
    lrs = []

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
    
    start_time = time.time()

    best_loss = 10

    loss_mask = torch.from_numpy(input_spec.mask).float().cuda()
    for step in tqdm(range(total_steps), desc = "Training Progress"):
        
        model_output = model(model_input)
        
        # print("Model Ooutput Shape: ", model_output.shape)
        if mask:
            ground_truth = ground_truth * loss_mask
            model_output = model_output * loss_mask

        mse_loss = mse(model_output, ground_truth)
        # mae_loss = mae(model_output, ground_truth)
        huber_loss = huber(model_output, ground_truth)
        # mrstft_loss = mrstft(model_output.view(1, 1, -1), ground_truth.view(1, 1, -1))
        loss = (1 - alpha) * mse_loss + alpha * huber_loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            
        losses.append(10 * np.log10(loss.item()))
        best_model = model
        best_iter = step

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step() # for StepLR learning rate scheduler,we don't need to pass loss into the step
        scheduler.step(loss) 

        current_lr = scheduler.get_last_lr() 
        lrs.append(10 * np.log10(current_lr))


    ### plot the latest loss landscape
    if visualization:
        STEPS = 50
        metric = loss_landscapes.metrics.Loss(mse, model_input.cpu(), ground_truth.cpu())

        model_final = copy.deepcopy(best_model)
        loss_data_fin = loss_landscapes.random_plane(model_final.cpu(), metric, distance=10, steps=STEPS, normalization='filter', deepcopy_model=True)

        plt.figure()
        ax = plt.axes(projection='3d')
        X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
        Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
        ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title('Surface Plot of Loss Landscape')
        savename = f'results/[scape]{inst}-{method}-{tag}-{total_steps}'
        plt.savefig(savename + '.png')

    ### record training and visualization time
    end_time = time.time()
    
    ### plot loss and learning rate history

    plt.figure()
    plt.plot(losses)
    plt.title(f'Training Loss, Best Iteration: {best_iter}, Total time: {end_time-start_time:.2f}')
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
    # perform scaling 
    print("spectrogram scaling factor: ", input_spec.scale)
    spec_recovered = final_model_output.reshape(height, width) * input_spec.scale
    spec_recovered = spec_recovered.cpu().detach().numpy()

    # visualizer(input_spec.stft_complex / input_spec.scale, "original_stft.png")
    visualizer(input_spec.mdct, f'results/[spec]{inst}-{method}-{tag}-original.png')
    visualizer(spec_recovered, f"results/[spec]{inst}_{method}-{tag}-fitted.png") # move to cpu, detach from gradients, and convert to numpy
    
    # fucking annoy type converion, tensor to numpy conversion, torchaudio format conversion, etc.
    print(np.max(spec_recovered))
    signal_recovered = mdct.ISTMDCT(spec_recovered).reshape(-1, 1).astype(np.float32)

    savename = f'results/[audio]{inst}-{method}-{tag}-{total_steps}'
    # torchaudio.save(savename + '.wav', torch.from_numpy(signal_recovered.reshape(-1, 1)), input_spec.sample_rate, format = "wav")
    wavfile.write(savename+'.wav', input_spec.sample_rate, signal_recovered)

    if save_checkpoint:
        torch.save(best_model, model_path)
        torch.save(optimizer, optimizer_path)

if __name__ == "__main__":

    tag = 'fixed-e6' # initial learning rate = e-06, decay = 0.8, patience = 200, long input signal
    configurations = [
        # {'inst': 'castanets', 'num_hidden_features': 256, 'num_hidden_layers': 6, 'omega': 22000, 'total_steps': 10000, 'alpha':0.0},
        # {'inst': 'violin', 'num_hidden_features': 256, 'num_hidden_layers': 6, 'omega': 22000, 'total_steps': 10000, 'alpha':0.0},

        # {'inst': 'castanets', 'num_hidden_features': 256, 'num_hidden_layers': 7, 'omega': 22000, 'total_steps': 10000, 'alpha':0.0},
        # {'inst': 'violin', 'num_hidden_features': 256, 'num_hidden_layers': 7, 'omega': 22000, 'total_steps': 10000, 'alpha':0.0},

        # {'inst': 'castanets', 'num_hidden_features': 512, 'num_hidden_layers': 6, 'omega': 22000, 'total_steps': 10000, 'alpha':0.0},
        # {'inst': 'violin', 'num_hidden_features': 512, 'num_hidden_layers': 6, 'omega': 22000, 'total_steps': 10000, 'alpha':0.0},

      
        # {'inst': 'castanets', 'tag': tag, 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 1000, 'total_steps': 15000, 'alpha':0, 'hp':False},
        # {'inst': 'oboe', 'tag': tag, 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 1000, 'total_steps': 15000, 'alpha':0, 'hp':False},
        # {'inst': 'glockenspiel', 'tag': tag, 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 1000, 'total_steps': 15000, 'alpha':0, 'hp':False},

        {'inst': 'violin', 'tag': tag + '-mask', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 500, 'total_steps': 20000, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':False, 'mask':True},
        # {'inst': 'violin', 'tag': tag, 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 500, 'total_steps': 20000, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':False, 'mask':False},
        {'inst': 'castanets', 'tag': tag + '-mask', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 500, 'total_steps': 20000, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':False, 'mask':True},
        # {'inst': 'castanets', 'tag': tag, 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 500, 'total_steps': 20000, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':False, 'mask':False},
        
        {'inst': 'harpsichord', 'tag': tag + '-mask', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 500, 'total_steps': 20000, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':False, 'mask':True},
        # {'inst': 'harpsichord', 'tag': tag, 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 500, 'total_steps': 20000, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':False, 'mask':False},# {'inst': 'castanets', 'tag': tag, 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 500, 'total_steps': 40000, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':False},
        # {'inst': 'oboe', 'tag': tag, 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 500, 'total_steps': 40000, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':False},
        # {'inst': 'glockenspiel', 'tag': tag, 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 500, 'total_steps': 40000, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':False}
    ]

    for config in configurations:
        train(**config)

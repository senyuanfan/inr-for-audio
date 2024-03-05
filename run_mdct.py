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

def train_mdct(inst:str, tag:str, num_hidden_features=256, num_hidden_layers=5, omega=300, total_steps=1000, learning_rate=1e-4, alpha=0.0, hp=False):
    method = "mdct"
    start_time = time.time()

    filename = f'data/{inst}.wav'
    input_spec = MDCTFitting(filename, duration=5, highpass=hp)
    height, width = input_spec.dimensions

    dataloader = DataLoader(input_spec, shuffle=True, batch_size = 1, pin_memory=True, num_workers=4)

    model = Siren(in_features=2, out_features=1, hidden_features=num_hidden_features, 
                 hidden_layers=num_hidden_layers, first_omega_0=omega, outermost_linear=True)

    # model = ReLU(in_features=2, out_features=2, hidden_features=num_hidden_features, hidden_layers=num_hidden_layers)
    
    model.cuda()
    summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    huber = nn.HuberLoss(reduction='mean', delta=1.0)
    mrstft = auraloss.freq.MultiResolutionSTFTLoss()

    losses = []
    lrs = []

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
    for step in tqdm(range(total_steps), desc = "Training Progress"):
        
        model_output, coords = model(model_input)
        mse_loss = mse(model_output, ground_truth)
        mae_loss = mae(model_output, ground_truth)
        huber_loss = huber(model_output, ground_truth)
        # mrstft_loss = mrstft(model_output.view(1, 1, -1), ground_truth.view(1, 1, -1))
        loss = (1 - alpha) * mse_loss + alpha * huber_loss
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
    # plt.xlim([0, 20000])
    savename = f'results/[loss]{inst}-{method}-{tag}-{total_steps}'
    plt.savefig(savename + '.png')

    plt.figure()
    plt.plot(lrs)
    plt.title("Learning Rate")
    plt.xlabel("Step")
    # plt.xlim([0, 20000])
    plt.ylabel("Learning Rate (dB)")
    savename = f'results/[lr]{inst}-{method}-{tag}-{total_steps}'
    plt.savefig(savename + '.png')
        
    final_model_output, _ = model(model_input)
    # perform scaling 
    print("spectrogram scaling factor: ", input_spec.scale)
    spec_recovered = final_model_output.reshape(height, width) * input_spec.scale
    spec_recovered = spec_recovered.cpu().detach().numpy()

    # visualizer(input_spec.stft_complex / input_spec.scale, "original_stft.png")
    visualizer(input_spec.mdct, f'results/[spec]{inst}-{method}-{tag}-original.png')
    visualizer(spec_recovered, f"results/[spec]{inst}_{method}-{tag}-fitted.png") # move to cpu, detach from gradients, and convert to numpy
    
    # fucking annoy type converion, tensor to numpy conversion, torchaudio format conversion, etc.
    signal_recovered = mdct.ISTMDCT(spec_recovered).reshape(-1, 1).astype(np.float32)

    savename = f'results/[audio]{inst}-{method}-{tag}-{total_steps}'
    # torchaudio.save(savename + '.wav', torch.from_numpy(signal_recovered.reshape(-1, 1)), input_spec.sample_rate, format = "wav")
    wavfile.write(savename+".wav", input_spec.sample_rate, signal_recovered)

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

        {'inst': 'castanets', 'tag':'huber', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 1000, 'total_steps': 10000, 'alpha':1, 'hp':False},
        {'inst': 'violin', 'tag': 'huber', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 1000, 'total_steps': 10000, 'alpha':1, 'hp':False},
        # {'inst': 'violin', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 1000, 'total_steps': 20000, 'alpha':0.0},
        {'inst': 'oboe', 'tag': 'huber', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 1000, 'total_steps': 10000, 'alpha':1, 'hp':False},
        # {'inst': 'glockenspiel', 'num_hidden_features': 256, 'num_hidden_layers': 5, 'omega': 1000, 'total_steps': 20000, 'alpha':0.0},
    ]

    for config in configurations:
        train_mdct(**config)

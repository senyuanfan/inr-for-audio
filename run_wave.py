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

def save_configuration(config, filename):
    """ Save the configuration dictionary to a file """
    with open(filename, 'w') as file:
        for key, value in config.items():
            file.write(f"{key}: {value}\n")

def plotspec(signal, fs, title):
    # print('original signal shape', signal.shape)
    plt.specgram(signal, NFFT=1024, noverlap=512, Fs=fs, mode='magnitude', scale='dB')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

def train(experiment_path:str, tag:str, inst:str, duration:int, method='wave', num_hidden_features=256, num_hidden_layers=4, num_tanh=2, omega=22000, total_steps=25000, learning_rate=1e-4, min_learning_rate=1e-6, alpha=0.0, load_checkpoint=False, save_checkpoint=False, visualization=False):
    filename = f'data/{inst}.wav'
    experiment_folder = f'{experiment_path}/{tag}-{inst}'
    os.mkdir(experiment_folder)

    """load input data from file"""
    # sample_rate, _ = wavfile.read(filename)
    input_audio = WaveformFitting(filename, duration=duration, highpass=False) # Hardcoded input length as 10 seconds
    dataloader = DataLoader(input_audio, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

    """load model with or without tanh layers appended to the end"""
    if num_tanh > 0:
        model = SirenWithTanh(in_features=1, out_features=1, hidden_features=num_hidden_features, num_tanh=num_tanh, 
                    hidden_layers=num_hidden_layers, first_omega_0=omega, outermost_linear=True)
    else:
        model = Siren(in_features=1, out_features=1, hidden_features=num_hidden_features, 
                    hidden_layers=num_hidden_layers, first_omega_0=omega, outermost_linear=True)
        
    model.cuda()
    summary(model)

    """initialize optimizer, learning rate scheduler, and loss function"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=200, min_lr=min_learning_rate)
    mae = nn.L1Loss()
    mse = nn.MSELoss()
    # mrstft = auraloss.freq.MultiResolutionSTFTLoss()


    """load input data to the model"""
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    """start timer and training, save the best model"""
    start_time = time.time()
    best_loss = float('inf')
    losses = []
    lrs = []

    for step in tqdm(range(total_steps), desc = "Training Progress"):
        
        model_output = model(model_input)
        mae_loss = mae(model_output, ground_truth)
        mse_loss = mse(model_output, ground_truth)
        # mrstft_loss = mrstft(model_output.view(1, 1, -1), ground_truth.view(1, 1, -1))
        loss = (1 - alpha) * mse_loss + alpha * mae_loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = model
            best_iter = step
        
        losses.append(10 * np.log10(loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        current_lr = scheduler.get_last_lr()
        lrs.append(10 * np.log10(current_lr))
    
    """loss landsacpe visualization"""
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
        savename = f'{experiment_folder}/loss_landscape'
        plt.savefig(savename+'.png')

    end_time = time.time()

    """save the best model as checkpoint"""
    if save_checkpoint:
        savename = f'{experiment_folder}/best_model'
        torch.save(best_model.state_dict(), savename+'.pt')
    

    """plot loss and learning rate history"""
    plt.figure(figsize=(6,10))

    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title(f'Training Loss, Best Iteration: {best_iter}, Total time: {(end_time-start_time)/60:.1f} min')
    plt.xlabel("Step")
    plt.ylabel("Loss (dB)")
    plt.xlim([0, total_steps])

    plt.subplot(2, 1, 2)
    plt.plot(lrs)
    plt.title("Learning Rate")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate (dB)")
    plt.xlim([0, total_steps])

    savename = f'{experiment_folder}/loss_lr_history'
    plt.savefig(savename+'.png')
        
    """save best model output as the recovered signal"""    
    final_model_output = best_model(model_input)
    signal_recovered = final_model_output.cpu().detach()
    # print("signal recovered shape: ", signal_recovered.shape)
    # print("signal recovered dtype: ", signal_recovered.dtype)
    # print("signal max: ", np.max(signal_recovered.numpy()))
    # print("signal min: ", np.min(signal_recovered.numpy()))

    """save the recovered output signal"""
    savename = f'{experiment_folder}/{inst}_recovered'
    output_filename = savename+'.wav'
    # wavfile.write(savename+'.wav', input_spec.sample_rate, signal_recovered)
    torchaudio.save(savename+'.wav', signal_recovered.reshape(1, -1), input_audio.sample_rate)


    """save the spectrogram comparison"""
    fs, ref = wavfile.read(filename)  
    fs, recovered = wavfile.read(output_filename)

    print(ref[:fs*duration,0].shape)
    print(recovered.shape)

    plt.figure(figsize=(6,10))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, 
                        top=0.9, wspace=0.4,hspace=0.4)

    plt.subplot(2,1,1)
    plotspec(ref[:fs*duration,0], fs, 'Reference')
    plt.subplot(2,1,2)
    plotspec(recovered, fs, 'Recovered')

    savename = f'{experiment_folder}/spectrogram'
    plt.savefig(savename+'.png')


if __name__ == "__main__":
    
    exp_num = 30
    note = 'tanh_test'
    exp_path = f'results/{exp_num}_{note}'
    while( os.path.exists(exp_path) ):
        exp_num = exp_num + 1
        exp_path = f'results/{exp_num}_{note}'
    os.mkdir(exp_path)

    steps = 10000

    configurations = [
        # {'experiment_path':exp_path, 'tag':'tanh-0', 'inst': 'violin', 'duration': 3, 'method':'wave', 'num_hidden_features': 256, 'num_hidden_layers': 4, 'num_tanh':0,
        #  'omega': 22000, 'total_steps': steps, 'learning_rate':1e-4, 'min_learning_rate':1e-6, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':True, 'visualization':False},
        {'experiment_path':exp_path, 'tag':'tanh-2', 'inst': 'violin', 'duration': 3, 'method':'wave', 'num_hidden_features': 256, 'num_hidden_layers': 4, 'num_tanh':2,
         'omega': 22000, 'total_steps': steps, 'learning_rate':1e-4, 'min_learning_rate':1e-6, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':True, 'visualization':False},
        {'experiment_path':exp_path, 'tag':'tanh-3', 'inst': 'violin', 'duration': 3, 'method':'wave', 'num_hidden_features': 256, 'num_hidden_layers': 4, 'num_tanh':3,
         'omega': 22000, 'total_steps': steps, 'learning_rate':1e-4, 'min_learning_rate':1e-6, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':True, 'visualization':False},
        {'experiment_path':exp_path, 'tag':'tanh-4', 'inst': 'violin', 'duration': 3, 'method':'wave', 'num_hidden_features': 256, 'num_hidden_layers': 4, 'num_tanh':4,
         'omega': 22000, 'total_steps': steps, 'learning_rate':1e-4, 'min_learning_rate':1e-6, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':True, 'visualization':False},
    ]

    for config in configurations:
        train(**config)

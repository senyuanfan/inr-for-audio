import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from scipy.io import wavfile
import librosa
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

import json

def save_parameters(experiment_folder, **kwargs):
    params_path = f"{experiment_folder}/parameters.json"
    with open(params_path, 'w') as file:
        json.dump(kwargs, file, indent=4)

def plotspec(signal, fs, title):
    # print('original signal shape', signal.shape)
    plt.specgram(signal, NFFT=1024, noverlap=512, Fs=fs, mode='magnitude', scale='dB')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

def train(experiment_path:str, tag:str, inst:str, duration:int, method='wave', mode='lp', num_hidden_features=256, num_sine=0, num_snake=4, num_tanh=0, omega=22000, total_steps=25000, learning_rate=1e-4, min_learning_rate=1e-6, alpha=0.0, load_ckpt=False, prev_ckpt_path=None, visualization=False):

    filename = f'data/{inst}.wav'
    experiment_folder = f'{experiment_path}/{inst}-{method}-{tag}'
    os.mkdir(experiment_folder)

    """save hyperparameters"""
    params = {
        'experiment_path': experiment_path,
        'tag': tag,
        'inst': inst,
        'duration': duration,
        'method': method,
        'mode': mode,
        'num_hidden_features': num_hidden_features,
        'num_sine': num_sine,
        'num_snake': num_snake,
        'num_tanh': num_tanh,
        'omega': omega,
        'total_steps': total_steps,
        'learning_rate': learning_rate,
        'min_learning_rate': min_learning_rate,
        'alpha': alpha,
        'load_ckpt': load_ckpt,
        'prev_ckpt_path': prev_ckpt_path,
        'visualization': visualization
    }
    save_parameters(experiment_folder, **params)

    """load model with or without tanh/snake layers appended to the end"""

    if method == 'wave':
        input_dimension = 1
    elif method == 'mdct':
        input_dimension = 2

    """initiate or load model and optimizer"""
    if load_ckpt:
        print("Loading model from:", prev_ckpt_path)
        if prev_ckpt_path is None or not isinstance(prev_ckpt_path, str):
            raise ValueError("The checkpoint path is not set or not a string.")

        if not os.path.exists(prev_ckpt_path):
            raise FileNotFoundError(f"The specified checkpoint file does not exist: {prev_ckpt_path}")

        model = SirenWithSnakeTanh(in_features=input_dimension, out_features=1, hidden_features=num_hidden_features, num_sine=num_sine, num_snake=num_snake, num_tanh=num_tanh, 
                                outermost_linear=True, first_omega_0=omega, hidden_omega_0=30.)
        
        checkpoint = torch.load(prev_ckpt_path)
        # Load state dictionary into the model and optimizer
        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda() # need to move model to cuda before the optimizer parameters get initiated

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=200, min_lr=min_learning_rate)

    else:
        model = SirenWithSnakeTanh(in_features=input_dimension, out_features=1, hidden_features=num_hidden_features, num_sine=num_sine, num_snake=num_snake, num_tanh=num_tanh, 
                                outermost_linear=True, first_omega_0=omega, hidden_omega_0=30.)
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=200, min_lr=min_learning_rate)


    # summary(model)

    """define loss function"""

    mae = nn.L1Loss()
    mse = nn.MSELoss()
    # mrstft = auraloss.freq.MultiResolutionSTFTLoss()


    """load input data from file"""
    # sample_rate, _ = wavfile.read(filename)

    '''
    For procedual training, we need to fit the same model with lower sample rate & low passed data first,
    and then fit the same model with higher sample rate and high-passed data.

    1. input_audio_low (contains low sample rate coordinates), input_audio_high (contains full scale coordinates)
    2. dataloader_low, dataloader_high 
    '''
    if method == 'wave':
        if mode == 'lp':
            input_audio = WaveformFitting(filename, duration=duration, lp=True)
        else:
            input_audio = WaveformFitting(filename, duration=duration, lp=False)

        dataloader = DataLoader(input_audio, shuffle=True, batch_size=1, pin_memory=True, num_workers=4)
    elif method == 'mdct':
        N = 2048
        input_mdct = MDCTFitting(filename, duration=duration, N=N)
        height, width = input_mdct.dimensions
        dataloader = DataLoader(input_mdct, shuffle=True, batch_size=1, pin_memory=True, num_workers=4)
    else:
        print('specify the correct fitting method as wave or mdct')

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
        STEPS = 30
        metric = loss_landscapes.metrics.Loss(mse, model_input.cpu(), ground_truth.cpu())

        model_final = copy.deepcopy(best_model)
        loss_data_fin = loss_landscapes.random_plane(model_final.cpu(), metric, distance=2, steps=STEPS, normalization='filter', deepcopy_model=True)

        plt.figure()
        ax = plt.axes(projection='3d')
        X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
        Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
        ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title('Surface Plot of Loss Landscape')
        savename = f'{experiment_folder}/landscape.png'
        plt.savefig(savename)

    end_time = time.time()


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

    savename = f'{experiment_folder}/loss.png'
    plt.savefig(savename)
        
    """save best model output as the recovered signal"""    
    final_model_output = best_model(model_input)
    signal_recovered = final_model_output.cpu().detach()
    # print("signal recovered shape: ", signal_recovered.shape)
    # print("signal recovered dtype: ", signal_recovered.dtype)
    # print("signal max: ", np.max(signal_recovered.numpy()))
    # print("signal min: ", np.min(signal_recovered.numpy()))

    """save the recovered output signal"""
    if method=='wave':
        savename = f'{experiment_folder}/output.wav'
        output_filename = savename

        # wavfile.write(savename, input_spec.sample_rate, signal_recovered)
        print('The input sample rate is: ', input_audio.sample_rate)
        torchaudio.save(savename, signal_recovered.reshape(1, -1), input_audio.sample_rate)
    elif method=='mdct':
        print("mdct scaling factor: ", input_mdct.scale)
        spec_recovered = final_model_output.reshape(height, width) * input_mdct.scale
        spec_recovered = spec_recovered.cpu().detach().numpy()
        print("maximum mdct magnitude: ", np.max(spec_recovered))
        signal_recovered = mdct.ISTMDCT(spec_recovered, N=N).reshape(-1, 1).astype(np.float32)

        savename = f'{experiment_folder}/output.wav'
        output_filename = savename

        wavfile.write(savename, input_mdct.sample_rate, signal_recovered)
    else:
        print('specify the correct fitting method as wave or mdct')


    """save the spectrogram comparison"""
    # fs, ref = wavfile.read(filename)  
    ref, fs_ref = librosa.load(filename, sr=None)
    recovered, fs_rec = librosa.load(output_filename, sr=None)

    # print(ref[:int(fs*(duration-0.5)),0].shape)
    # print(recovered.shape)

    plt.figure(figsize=(6,10))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, 
                        top=0.9, wspace=0.4,hspace=0.4)

    plt.subplot(2,1,1)
    plotspec(ref[:int(fs_ref*duration - 1)], fs_ref, 'Reference')
    plt.subplot(2,1,2)
    plotspec(recovered, fs_rec, 'Recovered')

    savename = f'{experiment_folder}/spec.png'
    plt.savefig(savename)

    """save the best model as checkpoint"""
    ckpt_path = f'{experiment_folder}/saved_ckpt.pt'

    checkpoint = {
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, ckpt_path)

    return ckpt_path



if __name__ == "__main__":
    
    '''
    1. update exp_num
    2. update note
    3. update tag for each experiment
    '''
    exp_num = 39
    note = 'procedual'
    exp_path = f'results/{exp_num}_{note}'
    if( os.path.exists(exp_path) == False ):
        os.mkdir(exp_path)



    # configurations = [
    #     # {'experiment_path':exp_path, 'tag':'violin-snake-4', 'inst': 'violin', 'duration': 10, 'method':'wave', 'num_hidden_features': 256, 'num_snake': 4,
    #     #  'omega': 22000, 'total_steps': steps, 'learning_rate':1e-4, 'min_learning_rate':1e-6, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':True, 'visualization':False},
    #     # {'experiment_path':exp_path, 'tag':'castanets-snake-4', 'inst': 'castanets', 'duration': 10, 'method':'wave', 'num_hidden_features': 256, 'num_snake':4,
    #     #  'omega': 22000, 'total_steps': steps, 'learning_rate':1e-4, 'min_learning_rate':1e-6, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':True, 'visualization':False},
    #     # {'experiment_path':exp_path, 'tag':'dire-snake-4', 'inst': 'dire', 'duration': 10, 'method':'wave', 'num_hidden_features': 256, 'num_snake':4,
    #     #  'omega': 22000, 'total_steps': steps, 'learning_rate':1e-4, 'min_learning_rate':1e-6, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':True, 'visualization':False},

    #     # {'experiment_path':exp_path, 'tag':'mdct-snake-4', 'inst': 'violin', 'duration': 10, 'method':'mdct', 'num_hidden_features': 256, 'num_snake':4,
    #     #  'omega': 2048, 'total_steps': steps, 'learning_rate':1e-4, 'min_learning_rate':1e-7, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':True, 'visualization':True},

    #     # {'experiment_path':exp_path, 'tag':'wave-snake-4', 'inst': 'violin', 'duration': 10, 'method':'wave', 'num_hidden_features': 256, 'num_snake':4,
    #     #  'omega': 22000, 'total_steps': steps, 'learning_rate':1e-4, 'min_learning_rate':1e-7, 'alpha':0, 'load_checkpoint':False, 'save_checkpoint':True, 'visualization':True},

    #     {'experiment_path':exp_path, 'tag':'snake-lp', 'inst': 'violin', 'duration': 5, 'method':'wave', 'num_hidden_features': 256, 'num_sine':4, 'num_snake':0,
    #      'omega': 22000, 'total_steps': steps, 'learning_rate':1e-4, 'min_learning_rate':1e-7, 'alpha':0, 'load_ckpt':False, 'prev_ckpt_path':None, 'visualization':False},

    #     {'experiment_path':exp_path, 'tag':'snake-hp', 'inst': 'violin', 'duration': 5, 'method':'wave', 'num_hidden_features': 256, 'num_sine':4, 'num_snake':0,
    #      'omega': 22000, 'total_steps': steps, 'learning_rate':1e-4, 'min_learning_rate':1e-7, 'alpha':0, 'load_ckpt':True, 'prev_ckpt_path':prev_ckpt_path, 'visualization':False},
    # ]

    # for config in configurations:
    #     print('ckpt: ', prev_ckpt_path)
    #     prev_ckpt_path = train(**config)


    vis = False
    prev_ckpt_path = None

    prev_ckpt_path = train(experiment_path=exp_path, tag='sin+snake-lp', inst='oboe', mode = 'lp', omega=22000, duration=30, total_steps=10000, num_sine=2, num_snake=2, load_ckpt=False, prev_ckpt_path=prev_ckpt_path, visualization=vis)
    _ = train(experiment_path=exp_path, tag='sin+snake-full', inst='oboe', mode = 'hp', omega=22000, duration=30, total_steps=15000, num_sine=2, num_snake=2, load_ckpt=True, prev_ckpt_path=prev_ckpt_path, visualization=vis)

    prev_ckpt_path = train(experiment_path=exp_path, tag='sin-lp', inst='oboe', mode = 'lp', omega=44000, duration=30, total_steps=10000, num_sine=4, num_snake=0, load_ckpt=False, prev_ckpt_path=prev_ckpt_path, visualization=vis)
    _ = train(experiment_path=exp_path, tag='sin-full', inst='oboe', mode = 'hp', omega=44000, duration=30, total_steps=15000, num_sine=4, num_snake=0, load_ckpt=True, prev_ckpt_path=prev_ckpt_path, visualization=vis)

    # _ = train(experiment_path=exp_path, tag='sin+snake-base', inst='violin', mode = 'hp', duration=5, total_steps=25000, num_sine=2, num_snake=2, load_ckpt=False, prev_ckpt_path=prev_ckpt_path, visualization=vis)

    # _ = train(experiment_path=exp_path, tag='sin-base', inst='violin', mode = 'hp', duration=5, total_steps=25000, num_sine=4, num_snake=0, load_ckpt=False, prev_ckpt_path=prev_ckpt_path, visualization=vis)

    # prev_ckpt_path = train(experiment_path=exp_path, tag='sin+snake-lp', inst='dire', mode = 'lp', duration=5, total_steps=10000, num_sine=2, num_snake=2, load_ckpt=False, prev_ckpt_path=prev_ckpt_path, visualization=vis)
    # _ = train(experiment_path=exp_path, tag='sin+snake-full', inst='dire', mode = 'hp', duration=5, total_steps=15000, num_sine=2, num_snake=2, load_ckpt=True, prev_ckpt_path=prev_ckpt_path, visualization=vis)

    # _ = train(experiment_path=exp_path, tag='sin+snake-base', inst='dire', mode = 'hp', duration=5, total_steps=25000, num_sine=2, num_snake=2, load_ckpt=False, prev_ckpt_path=prev_ckpt_path, visualization=vis)

    # prev_ckpt_path = train(experiment_path=exp_path, tag='sin+snake-lp', inst='castanets', mode = 'lp', duration=5, total_steps=10000, num_sine=2, num_snake=2, load_ckpt=False, prev_ckpt_path=prev_ckpt_path, visualization=vis)
    # _ = train(experiment_path=exp_path, tag='sin+snake-full', inst='castanets', mode = 'hp', duration=5, total_steps=15000, num_sine=2, num_snake=2, load_ckpt=True, prev_ckpt_path=prev_ckpt_path, visualization=vis)

    # _ = train(experiment_path=exp_path, tag='sin+snake-base', inst='castanets', mode = 'hp', duration=5, total_steps=25000, num_sine=2, num_snake=2, load_ckpt=False, prev_ckpt_path=prev_ckpt_path, visualization=vis)
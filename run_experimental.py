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

import rff # for random fourier embedding in [https://github.com/jmclong/random-fourier-features-pytorch]
from kan import KAN

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

def train(experiment_path:str, tag:str, inst:str, duration:int, loss_mode='mse', mode=None, decimation=1, bwe=False, num_hidden_features=256, num_sine=2, num_snake=2, num_tanh=0, num_freq=None, omega=22000, first_linear=False, last_linear=True, hidden_omega=30, a_initial=0.5, total_steps=20000, learning_rate=1e-3, min_learning_rate=1e-6, alpha=0.0, prev_ckpt_path=None, visualization=False):

    # mode: hp for wave, log for mdct
    filename = f'data/{inst}.wav'
    experiment_folder = f'{experiment_path}/{inst}-{tag}'

    while( os.path.exists(experiment_folder) == True ):
        tag = tag + '(2)'
        experiment_folder = f'{experiment_path}/{inst}-{tag}'

    os.mkdir(experiment_folder)
    takelog = False
    decimation = int(decimation)
    
    if num_freq is not None:
        input_dimension = num_freq * 2
    else:
        input_dimension = 1
        
    """load input data from file"""
    # sample_rate, _ = wavfile.read(filename)

    '''
    For procedual training, we need to fit the same model with lower sample rate & low passed data first,
    and then fit the same model with higher sample rate and high-passed data.

    1. input_audio_low (contains low sample rate coordinates), input_audio_high (contains full scale coordinates)
    2. dataloader_low, dataloader_high 
    '''

       
    input_data = WaveformFitting(filename, duration=duration, decimation=decimation)

    # if mode == 'lp':
    #     input_data = MultiWaveformFitting(filename, duration=duration, num_channels=num_channels, lp=True)
    # else:
    #     input_data = MultiWaveformFitting(filename, duration=duration, num_channels=num_channels, lp=False)
    # input_dimension = 2

    dataloader = DataLoader(input_data, shuffle=True, batch_size=1, pin_memory=True, num_workers=4)


    model = SirenWithSnakeTanh(in_features=input_dimension, out_features=1, hidden_features=num_hidden_features, num_sine=num_sine, num_snake=num_snake, num_tanh=num_tanh, 
                num_freq=num_freq, first_linear=first_linear, last_linear=last_linear, first_omega_0=omega, hidden_omega_0=hidden_omega, a_initial=a_initial)
    
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=200, min_lr=min_learning_rate)


    # summary(model)

    """define loss function"""

    mae = nn.L1Loss()
    mse = nn.MSELoss()
    snr = auraloss.time.SNRLoss()
    mrstft = auraloss.freq.MultiResolutionSTFTLoss(perceptual_weighting=False, sample_rate=input_data.sample_rate)
    # mrstft = auraloss.freq.STFTLoss()

    """produce bwe coordinates, only works under wave mode"""
    model_input_bwe = get_coord(input_data.original_sample_rate * duration, dim = 1)
    print('bwe coord shape: ', model_input_bwe.shape)

    """load input data to the model"""
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
    model_input_bwe = model_input_bwe.reshape(1, -1, 1).cuda()
    print("bwe input dimension: ", model_input_bwe.shape)

    
    if num_freq is not None:
        encoding = rff.layers.GaussianEncoding(10.0, 1, num_freq).cuda()
        model_input = encoding(model_input)
        model_input_bwe = encoding(model_input_bwe)



    
    # get_coord(len(self.data), 1)
    gt_power = torch.mean(torch.square(model_input))

    """start timer and training, save the best model"""
    start_time = time.time()
    best_loss = float('inf')
    best_iter = -1
    losses = []
    lrs = []

    for step in tqdm(range(total_steps), desc = "Training Progress"):
        
        model_output = model(model_input)

        mrstft_loss = mrstft(model_output.view(1, 1, -1), ground_truth.view(1, 1, -1))
        if loss_mode == 'mae':
            mae_loss = mae(model_output, ground_truth)
            loss = (1 - alpha) * mae_loss + alpha * mrstft_loss
        elif loss_mode == 'snr':
            snr_loss = snr(model_output.view(1, 1, -1), ground_truth.view(1, 1, -1))
            loss = (1 - alpha) * snr_loss + alpha * mrstft_loss
        else: # loss == 'mse'
            mse_loss = mse(model_output, ground_truth)
            loss = (1 - alpha) * mse_loss + alpha * mrstft_loss

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = model
            best_iter = step
        
        # current_loss = 10 * torch.log10(gt_power / loss.item())
        # current_loss = current_loss.detach().cpu().numpy()
        # losses.append(current_loss)

        current_loss = 10 * np.log10(loss.item()+1e-10)
        # current_loss = loss.item()
        losses.append(current_loss)

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
        ax.set_zlim(0, 0.15)
        savename = f'{experiment_folder}/landscape.png'
        plt.savefig(savename)

    end_time = time.time()
    total_time = (end_time-start_time)/60


    """plot loss and learning rate history"""
    plt.figure(figsize=(6,10))

    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title(f'Training Loss, Best Iteration: {best_iter}, Total time: {total_time:.1f} min')
    plt.xlabel("Step")
    plt.ylabel("Loss")
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
    # convert 32-bit model to 16-bit model and calculate the size
    """model quantization"""    
    # model_input = model_input.half()
    # best_model = best_model.half()
    # best_model.cuda()

    param_size = 0
    buffer_size = 0
    for param in best_model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in best_model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_size = (param_size + buffer_size) / 1024 # convert to KB

    """inference step"""
    # perform bandwidth extension by evaluating the model at higher sample rate
    if bwe: # and in waveform mode
        final_model_output = best_model(model_input_bwe)
        recover_sample_rate = input_data.original_sample_rate
    else:
        final_model_output = best_model(model_input)
        recover_sample_rate = input_data.sample_rate
    
    if takelog:
        final_model_output = torch.exp(final_model_output)
    signal_recovered = final_model_output.cpu().detach()
    # print("signal recovered shape: ", signal_recovered.shape)
    # print("signal recovered dtype: ", signal_recovered.dtype)
    # print("signal max: ", np.max(signal_recovered.numpy()))
    # print("signal min: ", np.min(signal_recovered.numpy()))

    """save the recovered output signal"""
    if method == 'wave':
        savename = f'{experiment_folder}/output.wav'
        output_filename = savename

        # wavfile.write(savename, input_spec.sample_rate, signal_recovered)
        # print('The input sample rate is: ', input_data.sample_rate)
        # print('The recover sample rate is: ', recover_sample_rate)
        signal_recovered = signal_recovered.to(torch.float32)
        print('The recovered signal shape is: ', signal_recovered.shape)
        if bwe:
            torchaudio.save(savename, signal_recovered.reshape(input_data.width, input_data.height*decimation), recover_sample_rate)
        else:
            torchaudio.save(savename, signal_recovered.reshape(input_data.width, input_data.height), recover_sample_rate)

    elif method == 'mdct':
        spec_recovered = final_model_output.reshape(input_data.height, input_data.width) * input_data.scale + input_data.mean - input_data.shift
        spec_recovered = spec_recovered.cpu().detach().numpy()
        if takelog:
            # spec_recovered = np.power(10, spec_recovered)
            spec_recovered = np.exp(spec_recovered)

        print("maximum mdct magnitude: ", np.max(spec_recovered))
        signal_recovered = mdct.ISTMDCT(spec_recovered, N=N).reshape(-1, 1).astype(np.float32)

        savename = f'{experiment_folder}/output.wav'
        output_filename = savename

        wavfile.write(savename, input_data.sample_rate, signal_recovered)
    else:
        print('specify the correct fitting method as wave or mdct')


    """save the spectrogram comparison and the waveform comparison"""
    # fs, ref = wavfile.read(filename)  
    # will automatically convert loaded signal tomono
    ref, fs_ref = librosa.load(filename, sr=None)
    rec, fs_rec = librosa.load(output_filename, sr=None)
    
    # trim the reference signal length and decimate according to the decimation hyperparameter
    ref = ref[:int(fs_ref*duration - 1)]
    
    if bwe:
        d = 1
    else:
        d = decimation

    ref = decimate(ref, q=d)
    # to resolve contrast issue in plotting
    ref = ref + 1e-5
    fs_ref = fs_ref // d
    print("fs ref: ", fs_ref)
    print("fs rec: ", fs_rec)

    print("reference signal shape", ref.shape)
    print("recovered signal shape", rec.shape)

    plt.figure(figsize=(6,10))
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, 
                        top=0.9, wspace=0.4,hspace=0.4)

    plt.subplot(2,1,1)
    plotspec(ref, fs_ref, 'Reference')
    plt.subplot(2,1,2)
    plotspec(rec, fs_rec, 'Recovered')

    savename = f'{experiment_folder}/spec.png'
    plt.savefig(savename)


    plt.figure(figsize=(6,10))
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, 
                        top=0.9, wspace=0.4,hspace=0.4)
    
    plt.subplot(2, 1, 1)
    plt.plot(ref)
    plt.title('Reference')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(rec)
    plt.title('Recovered')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    savename = f'{experiment_folder}/wave.png'
    plt.savefig(savename)
    
    """save the best model as checkpoint"""
    ckpt_path = f'{experiment_folder}/saved_ckpt.pt'

    checkpoint = {
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, ckpt_path)

    """save hyperparameters"""
    params = {
        'experiment_path': experiment_path,
        'tag': tag,
        'inst': inst,
        'duration': duration,
        'num_channels': num_channels,
        'method': method,
        'arch': arch,
        'loss_mode': loss_mode,
        'mode': mode,
        'decimation': decimation,
        'bwe': bwe,
        'num_hidden_features': num_hidden_features,
        'num_sine': num_sine,
        'num_snake': num_snake,
        'num_tanh': num_tanh,
        'num_freq': num_freq,
        'omega': omega,
        'hidden_omega': hidden_omega,
        'a_initial': a_initial,
        'total_steps': total_steps,
        'learning_rate': learning_rate,
        'min_learning_rate': min_learning_rate,
        'alpha': alpha,
        'prev_ckpt_path': prev_ckpt_path,
        'curr_ckpt_path': ckpt_path,
        'visualization': visualization,
        'parameter_size(KB)': param_size/1024,
        'total_model_size(KB)': model_size,
        'total_trainig_time(min)': total_time
    }
    save_parameters(experiment_folder, **params)

    return ckpt_path



if __name__ == "__main__":
    
    '''
    1. update exp_num
    2. update note
    3. update tag for each experiment
    4. when using method 'wave', set mode to 'hp' or 'lp'; when using method 'mdct', set mode to 'log' or None, for loss function, set mode to 'snr', 'mae' or None (by default mse)
    '''
    exp_num = 77
    note = 'hierarchy'
    exp_path = f'results/{exp_num}_{note}'
    if( os.path.exists(exp_path) == False ):
        os.mkdir(exp_path)


    # insts = ['castanets', 'oboe', 'glockenspiel', 'violin']
    insts = ['oboe']
    # alphas = [0]
    # num_freqs = [None]
    # num_snakes = [2]
    # modes = ['mse']

    steps_long = 20000
    steps_short = steps_long // 4

    for inst in insts:
        prev_ckpt_path = None

        _ = train(experiment_path=exp_path, tag=f'short-full', inst=inst, duration=1, method='wave', total_steps=steps_long, loss_mode='mse', alpha=0, num_sine=2, num_snake=2)
        _ = train(experiment_path=exp_path, tag=f'long-full', inst=inst, duration=2, method='wave', total_steps=steps_long, loss_mode='mse', alpha=0, num_sine=2, num_snake=2)
        _ = train(experiment_path=exp_path, tag=f'long-d2', inst=inst, duration=2, decimation=2, method='wave', total_steps=steps_long, loss_mode='mse', alpha=0, num_sine=2, num_snake=2)



        # set num_freq to 128, will map input coordinate of 128 **pairs** of encodings [cos(2πBv),sin(2πBv)] with B randomly selected form gaussian distribution
        # _ = train(experiment_path=exp_path, tag=f'fe_snake_256_mix', inst=inst, duration=10, method='wave', total_steps=steps_long, loss_mode='mae', alpha=0.2, num_freq=256, first_linear=False, num_sine=0, num_snake=4)
        # _ = train(experiment_path=exp_path, tag=f'fe_snake_256', inst=inst, duration=10, method='wave', total_steps=steps_long, loss_mode='mse', alpha=0, num_freq=256, first_linear=False, num_sine=0, num_snake=4)
        # _ = train(experiment_path=exp_path, tag=f'fe_snake_128', inst=inst, duration=10, method='wave', total_steps=steps_long, loss_mode='mse', alpha=0, num_freq=128, first_linear=False, num_sine=0, num_snake=4)
        # _ = train(experiment_path=exp_path, tag=f'fe_snake_64', inst=inst, duration=10, method='wave', total_steps=steps_long, loss_mode='mse', alpha=0, num_freq=64, first_linear=False, num_sine=0, num_snake=4)

        # _ = train(experiment_path=exp_path, tag=f'[01]mse', inst=inst, duration=5, method='mdct', total_steps=steps_long, num_sine=4, num_snake=0,  loss_mode='mse', alpha=0, prev_ckpt_path=prev_ckpt_path)
        # _ = train(experiment_path=exp_path, tag=f'[02]mae', inst=inst, duration=5, method='mdct', total_steps=steps_long, num_sine=4, num_snake=0,  loss_mode='mae', alpha=0, prev_ckpt_path=prev_ckpt_path)

        # _ = train(experiment_path=exp_path, tag=f'[03]first_linear', inst=inst, duration=5, method='wave', total_steps=steps_long, first_linear=True, num_sine=4, num_snake=0, num_tanh=0, loss_mode='mse', alpha=0, prev_ckpt_path=prev_ckpt_path, vis=True)
        # _ = train(experiment_path=exp_path, tag=f'[04]sine', inst=inst, duration=5, method='wave', total_steps=steps_long, num_sine=4, num_snake=0, num_tanh=0, loss_mode='mse', alpha=0, prev_ckpt_path=prev_ckpt_path, visualization=True)
        # _ = train(experiment_path=exp_path, tag=f'[05]snake', inst=inst, duration=5, method='wave', total_steps=steps_long, num_sine=0, num_snake=4, num_tanh=0, loss_mode='mse', alpha=0, prev_ckpt_path=prev_ckpt_path, visualization=True)
        # # _ = train(experiment_path=exp_path, tag=f'[06]tanh', inst=inst, duration=5, method='wave', total_steps=steps_long, num_sine=0, num_snake=0, num_tanh=4, loss_mode='mse', alpha=0, prev_ckpt_path=prev_ckpt_path, visualization=True)
        # _ = train(experiment_path=exp_path, tag=f'[07]mix_sine_snake', inst=inst, duration=5, method='wave', total_steps=steps_long, num_sine=2, num_snake=2, num_tanh=0, loss_mode='mse', alpha=0, prev_ckpt_path=prev_ckpt_path, visualization=True)
        
        # _ = train(experiment_path=exp_path, tag=f'[08]full_mse', inst=inst, duration=5, method='wave', total_steps=steps_long, loss_mode='mse', alpha=0, prev_ckpt_path=prev_ckpt_path)
        # _ = train(experiment_path=exp_path, tag=f'[09]full_mae', inst=inst, duration=5, method='wave', total_steps=steps_long, loss_mode='mae', alpha=0, prev_ckpt_path=prev_ckpt_path)
        # _ = train(experiment_path=exp_path, tag=f'[10]full_mrstft', inst=inst, duration=5, method='wave', total_steps=steps_long, loss_mode='mae', alpha=1.0, prev_ckpt_path=prev_ckpt_path)

        # _ = train(experiment_path=exp_path, tag=f'[11]mix_mae_mrstft', inst=inst, duration=10, method='wave', total_steps=steps_long, loss_mode='mae', alpha=0.01)

        # prev_ckpt_path = None
        # prev_ckpt_path = train(experiment_path=exp_path, tag=f'[12]procedual_mse_mix_d8', inst=inst, duration=5, method='wave', loss_mode='mse', total_steps=steps_short, decimation=8, alpha=0.2, prev_ckpt_path=prev_ckpt_path)
        # prev_ckpt_path = train(experiment_path=exp_path, tag=f'[12]procedual_mse_mix_d4', inst=inst, duration=5, method='wave', loss_mode='mse', total_steps=steps_short, decimation=4, alpha=0.2, prev_ckpt_path=prev_ckpt_path)
        # prev_ckpt_path = train(experiment_path=exp_path, tag=f'[12]procedual_mse_mix_d2', inst=inst, duration=5, method='wave', loss_mode='mse', total_steps=steps_short, decimation=2, alpha=0.2, prev_ckpt_path=prev_ckpt_path)
        # _              = train(experiment_path=exp_path, tag=f'[12]procedual_mse_mix_d1', inst=inst, duration=5, method='wave', loss_mode='mse', total_steps=10000, decimation=1, alpha=0.2, prev_ckpt_path=prev_ckpt_path)

        # prev_ckpt_path = None
        # prev_ckpt_path = train(experiment_path=exp_path, tag=f'[13]procedual_mse_d8', inst=inst, duration=5, method='wave', loss_mode='mse', total_steps=steps_short, decimation=8, alpha=0, prev_ckpt_path=prev_ckpt_path)
        # prev_ckpt_path = train(experiment_path=exp_path, tag=f'[13]procedual_mse_d4', inst=inst, duration=5, method='wave', loss_mode='mse', total_steps=steps_short, decimation=4, alpha=0, prev_ckpt_path=prev_ckpt_path)
        # prev_ckpt_path = train(experiment_path=exp_path, tag=f'[13]procedual_mse_d2', inst=inst, duration=5, method='wave', loss_mode='mse', total_steps=steps_short, decimation=2, alpha=0, prev_ckpt_path=prev_ckpt_path)
        # _              = train(experiment_path=exp_path, tag=f'[13]procedual_mse_d1', inst=inst, duration=5, method='wave', loss_mode='mse', total_steps=10000, decimation=1, alpha=0, prev_ckpt_path=prev_ckpt_path)
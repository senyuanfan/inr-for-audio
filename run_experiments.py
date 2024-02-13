from siren_utils import *

# # parameters to change:
# 1. # of hidden layers
# 2. # of neurons per hidden layer
# 3. # of first_omega_0 and hidden_omega_0, equivalent to timestamp scaling
# 4. # input file path
# 5. # loss function
# 6. # training steps


# # return
# 1. loss graph
# 2. audio output


# input_audio = AudioFile('castanets.wav')

def train(inst, num_hidden_features = 256, num_hidden_layers = 4, omega = 5000, total_steps = 100):
    
    filename = 'data/' + inst + '.wav'
    sample_rate, _ = wavfile.read(filename)
    input_audio = AudioFile(filename, duration=10) # hard coded the input length as 10 seconds

    dataloader = DataLoader(input_audio, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

    # Note that we increase the frequency of the first layer to match the higher frequencies of the
    # audio signal. Equivalently, we could also increase the range of the input coordinates.
    # scale first_omega_0 according to the length of input

    model = Siren(in_features=1, out_features=1, hidden_features=num_hidden_features, 
                        hidden_layers=num_hidden_layers, first_omega_0=omega, outermost_linear=True)
    model.cuda()


    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
    optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
    mrstft = auraloss.freq.MultiResolutionSTFTLoss()
    steps_til_summary = 1000

    losses = []
    for step in range(total_steps):
        model_output, coords = model(model_input)    
        loss = F.mse_loss(model_output, ground_truth)
        losses.append(loss.cpu().detach().numpy())
        
        optim.zero_grad()
        loss.backward()
        optim.step()

    plt.figure()
    plt.plot(losses)
    savename = 'results/' + inst + '-' + str(num_hidden_features) + '-' + str(num_hidden_layers) + '-' + str(omega)
    plt.savefig(savename + '.png')
    
    final_model_output, coords = model(model_input)
    
    torchaudio.save(savename+'.wav', final_model_output.cpu().detach().reshape(1, -1), sample_rate)


if __name__ == "__main__":
    insts = ['castanets', 'quartet']
    hfs = [256, 512]
    hls = [3, 4, 5]
    ogs = [3000, 5000, 10000]
    tps = [5000, 10000]

    for inst in insts:
        for hf in hfs:
            for hl in hls:
                for og in ogs:
                    for tp in tps:
                        train(inst, num_hidden_features = hf, num_hidden_layers = hl, omega = og, total_steps = tp)


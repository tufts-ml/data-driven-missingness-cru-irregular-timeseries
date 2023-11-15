import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.distributions.bernoulli import Bernoulli
from time_series import TimeSeries
from utils_pvae import Rescaler, make_scheduler
from spline_cconv import ContinuousConv1D
from layers import Decoder, gan_loss
from toy_pvae import Encoder as PVAE_Encoder
from toy_layers import SeqGeneratorDiscrete
import seaborn as sns
import argparse

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')



class PVAE(nn.Module):
    def __init__(self, encoder, decoder, sigma=.2):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sigma = sigma

    def forward(self, data, time, mask, cconv_graph):
        batch_size = len(data)
        z, mu, logvar, eps = self.encoder(cconv_graph, batch_size)
        x_recon = self.decoder(z, time, mask)
        recon_loss = (1 / (2 * self.sigma**2) * F.mse_loss(
            x_recon * mask, data * mask, reduction='none') * mask).sum((1, 2))
        kl_loss = .5 * (z**2 - logvar - eps**2).sum(1)
        
        mse = (F.mse_loss(
            x_recon * mask, data * mask, reduction='none') * mask).sum((1, 2))
        
        loss = mse.mean()        
        return loss


def irregularly_sampled_data_gen_square_sine_2(n=1000, length=50, seed=0, obs_proba=0.0006):
    np.random.seed(seed)
    # obs_times = obs_times_gen(n)
    P=25
    D=12
    obs_values, ground_truth, obs_times, masks = [], [], [], []
    sampling_rate = 25
    for i in range(n):
        t_raw = np.arange(length)
        b = np.random.uniform(0, 50, 1)
        t1 = t_raw/sampling_rate
        t2 = t_raw/sampling_rate
        np.random.seed(i)
        
#         f1 = (np.arange(length)+b) % P < D
        f1 = (np.ceil(t1*sampling_rate)+b)%P < D
        mask_0 = f1<.5
        mask_1 = f1>.5
        freq=2
        phase = b/sampling_rate
        f2_clean = mask_1*np.sin(2*np.pi*freq*t2+phase)+mask_0*np.cos(2*np.pi*freq*t2+phase)*0.4
#         f2 = f2_clean+0.01*np.random.randn(len(t2))
        f2 = f2_clean
        obs_times.append(np.stack((t1, t2), axis=0))
        obs_values.append(np.stack((f1, f2), axis=0))
        #obs_values.append([f1.tolist(), f2.tolist(), f3.tolist()])
        fg1 = np.arange(length) % P < D
        fg2 = f2_clean
        #ground_truth.append([f1.tolist(), f2.tolist(), f3.tolist()])
        ground_truth.append(np.stack((fg1, fg2), axis=0))
        
        
        mnar_lims = [-.4, .4]
        m1 = np.asarray(Bernoulli(torch.tensor(0.6)).sample(f1.shape)) ## Get some irregularly sampled time points
        mnar_inds_1 = np.logical_or(f1<mnar_lims[0], f1>mnar_lims[1])
        miss_1 = np.asarray(Bernoulli(torch.tensor(obs_proba)).sample(f1.shape))
        m1[mnar_inds_1] = miss_1[mnar_inds_1]
        
        m2 = np.asarray(Bernoulli(torch.tensor(0.6)).sample(f2.shape)) ## Get some irregularly sampled time points
        mnar_inds_2 = np.logical_or(f2<mnar_lims[0], f2>mnar_lims[1])
        miss_2 = np.asarray(Bernoulli(torch.tensor(obs_proba)).sample(f2.shape))
        m2[mnar_inds_2] = miss_2[mnar_inds_2]
        masks.append(np.stack((m1, m2), axis=0))
        
        
    return obs_values, ground_truth, obs_times, masks


def get_toy_data_mnar_square_sine(args, N=1000, obs_proba=0.0006):
    dim = 2
    n=N
    length=50
    obs_values, ground_truth, obs_times, masks = irregularly_sampled_data_gen_square_sine_2(
        n, length, obs_proba=obs_proba)
#     obs_times = np.array(obs_times).reshape(n, -1)
    obs_times = np.array(obs_times)[:, 0, :]
    obs_values = np.array(obs_values)
    mask_vals = np.array(masks)
    combined_obs_values = np.zeros((n, dim, obs_times.shape[-1]))
    mask = np.zeros((n, dim, obs_times.shape[-1]))
    for i in range(dim):
        combined_obs_values[:, i, :] = obs_values[:, i]

        mask[:, i, :] = mask_vals[:, i]
    #print(combined_obs_values.shape, mask.shape, obs_times.shape, np.expand_dims(obs_times, axis=1).shape)
    combined_data = np.concatenate(
        (combined_obs_values, mask, np.expand_dims(obs_times, axis=1)), axis=1)
    combined_data = np.transpose(combined_data, (0, 2, 1))
    print(combined_data.shape)
    train_data, test_data = model_selection.train_test_split(combined_data, train_size=0.8,
                                                             random_state=42, shuffle=True)
    print(train_data.shape, test_data.shape)
    train_dataloader = DataLoader(torch.from_numpy(
        train_data).float(), batch_size=len(train_data), shuffle=False)
    test_dataloader = DataLoader(torch.from_numpy(
        test_data).float(), batch_size=len(test_data), shuffle=False)
    data_objects = {"dataset_obj": combined_data,
                    "train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": dim,
                    "ground_truth": np.array(ground_truth)}
    return data_objects


def gen_data_mnar(n_samples=500, seq_len=50, max_time=2):
    """Generates a 1-channel synthetic dataset.

    The observations are within a window of size (max_time * obs_span_rate)
    randomly occurring at the time span [0, max_time].

    Args:
        n_samples:
            Number of data cases.
        seq_len:
            Maximum number of observations in a channel.
        max_time:
            Length of time interval [0, max_time].
        poisson_rate:
            Rate of homogeneous Poisson process.
        obs_span_rate:
            The continuous portion of the time span [0, max_time]
            that observations are restricted in.
    """
    n_channels = 2
    time_unif = np.linspace(0, max_time, seq_len)
    time_unif_2ch = np.broadcast_to(time_unif, (n_channels, seq_len))
    data_unif = np.empty((n_samples, n_channels, seq_len))
    sparse_data, sparse_time, sparse_mask = [
        np.empty((n_samples, n_channels, seq_len)) for _ in range(3)]
    sampling_rate=25
    
    def gen_time_series_3(t, seed=50):        
        t0 = t[0]
        t1 = t[1]
        P = 25
        D = 12
        np.random.seed(seed)
        b = np.random.uniform(0, 50, 1)
        f1 = (np.ceil(t0*sampling_rate)+b)%P < D
        mask_0 = f1<.5
        mask_1 = f1>.5
        freq=2
        data = np.empty((2, seq_len))
        phase = b/sampling_rate
        f2 = mask_1*np.sin(2*np.pi*freq*t1+phase)+mask_0*np.cos(2*np.pi*freq*t1+phase)*0.4
        data[0] = f1
        data[1] = f2
        
        return data
    
    
    for i in range(n_samples):

        # Noise-free evenly-sampled time series
        data_unif[i] = gen_time_series_3(time_unif_2ch, seed=i)

        
        times = time_unif_2ch
        masks = np.ones_like(times)
        
        sparse_time[i] = times * masks
        sparse_mask[i] = masks
        sparse_data[i] = gen_time_series_3(time_unif_2ch, seed=i)

    # Add a small independent Gaussian noise to each channel
#     sparse_data += np.random.normal(0, .01, sparse_data.shape)

    # Pack the data to minimize the padded entries
    compact_len = sparse_mask.astype(int).sum(axis=2).max()
    compact_data, compact_time, compact_mask = [
        np.zeros((n_samples, n_channels, compact_len)) for _ in range(3)]
    for i in range(n_samples):
        for j in range(n_channels):
            idx = sparse_mask[i, j] == 1
            n_obs = idx.sum()
            compact_data[i, j, :n_obs] = sparse_data[i, j, idx]
            compact_time[i, j, :n_obs] = sparse_time[i, j, idx]
            compact_mask[i, j, :n_obs] = sparse_mask[i, j, idx]

    return compact_data, compact_time, compact_mask, data_unif, time_unif
    

def main():
    max_time = 2
    train_data, train_time, train_mask, data_unif, time_unif = gen_data_mnar(
        n_samples=400, seq_len=50, max_time=max_time)

    test_data, test_time, test_mask, test_data_unif, test_time_unif = gen_data_mnar(
        n_samples=100, seq_len=50, max_time=max_time)
    
    # add missing-not-at-randomness to the toy data
    obs_proba=0.0006
    N = len(train_data)
    np.random.seed(0)
    for n in range(N):
        f1 = train_data[n, 0, :]
        f2 = train_data[n, 1, :]
        mnar_lims = [-.4, .4]

        torch.manual_seed(n)
        m1 = np.asarray(Bernoulli(torch.tensor(0.6)).sample(f1.shape)) ## Get some irregularly sampled time points
        mnar_inds_1 = np.logical_or(f1<mnar_lims[0], f1>mnar_lims[1])
        miss_1 = np.asarray(Bernoulli(torch.tensor(obs_proba)).sample(f1.shape))
        m1[mnar_inds_1] = miss_1[mnar_inds_1]

        m2 = np.asarray(Bernoulli(torch.tensor(0.6)).sample(f2.shape)) ## Get some irregularly sampled time points
        mnar_inds_2 = np.logical_or(f2<mnar_lims[0], f2>mnar_lims[1])
        miss_2 = np.asarray(Bernoulli(torch.tensor(obs_proba)).sample(f2.shape))
        m2[mnar_inds_2] = miss_2[mnar_inds_2]
        train_mask[n, :, ] = np.stack([m1, m2], axis=1).T



    N_test = len(test_data)
    np.random.seed(0)
    for n in range(N_test):
        f1 = test_data[n, 0, :]
        f2 = test_data[n, 1, :]
        mnar_lims = [-.4, .4]
        m1 = np.asarray(Bernoulli(torch.tensor(0.6)).sample(f1.shape)) ## Get some irregularly sampled time points
        mnar_inds_1 = np.logical_or(f1<mnar_lims[0], f1>mnar_lims[1])
        miss_1 = np.asarray(Bernoulli(torch.tensor(obs_proba)).sample(f1.shape))
        m1[mnar_inds_1] = miss_1[mnar_inds_1]

        m2 = np.asarray(Bernoulli(torch.tensor(0.6)).sample(f2.shape)) ## Get some irregularly sampled time points
        mnar_inds_2 = np.logical_or(f2<mnar_lims[0], f2>mnar_lims[1])
        miss_2 = np.asarray(Bernoulli(torch.tensor(obs_proba)).sample(f2.shape))
        m2[mnar_inds_2] = miss_2[mnar_inds_2]
        test_mask[n, :, ] = np.stack([m1, m2], axis=1).T
        
    _, in_channels, seq_len = train_data.shape

    scaled_train_time = train_time
    scaled_train_data = train_data

    scaled_test_time = test_time
    scaled_test_data = test_data
    
    # Hyperparameters for training P-VAE
    batch_size = 300#128
    latent_size = 4
    epochs = 500
    lr = 0.001#1e-4
    min_lr = 5e-5
    sigma = 1
    device='cpu'

    # train on only time upto 1.5 and extrapolate after
    t = np.linspace(0, max_time, 50)
    train_mask[:, :, t>=1.5]=0


    train_dataset = TimeSeries(
        scaled_train_data, scaled_train_time, train_mask, max_time=max_time,
        device=device)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, collate_fn=train_dataset.collate_fn)

    grid_decoder = SeqGeneratorDiscrete(in_channels, latent_size, torch.tanh)
    decoder = Decoder(grid_decoder, max_time=max_time).to(device)

    cconv = ContinuousConv1D(in_channels, max_time=max_time, norm=True).to(device)
    encoder = PVAE_Encoder(latent_size, cconv).to(device)

    pvae = PVAE(encoder, decoder, sigma=sigma).to(device)

    optimizer = optim.Adam(pvae.parameters(), lr=lr)
    scheduler = make_scheduler(optimizer, lr, min_lr, epochs)

    loss_per_epoch = np.zeros(epochs)
    for epoch in range(epochs):
        for val, idx, mask, _, cconv_graph in train_loader:
            optimizer.zero_grad()
            loss = pvae(val, idx, mask, cconv_graph)
            loss.backward()
            optimizer.step()

        print('Epoch : %d, \nloss : '%epoch)
        print(loss.item())
        loss_per_epoch[epoch]=loss.item()

    
    # test model and visualize extrapolations
    test_dataset = TimeSeries(
    scaled_test_data, scaled_test_time, test_mask, max_time=max_time,
    device=device)


    batch_size = len(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             collate_fn=test_dataset.collate_fn)
    (val, idx, mask, _, cconv_graph) = next(iter(test_loader))
    in_channels = val.shape[1]
    max_time = max_time
    t = torch.linspace(0, max_time, 50, device=device)
    t = t.expand(batch_size, in_channels, len(t)).contiguous()
    t_mask = torch.ones_like(t)

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z = pvae.encoder(cconv_graph, batch_size)
        if not torch.is_tensor(z):   # P-VAE encoder returns a list
            z = z[0]
        imp_data = pvae.decoder(z, t, t_mask)
        data_noise = torch.empty_like(z).normal_()
        gen_data = decoder(data_noise, t, t_mask)
    
    f, axs = plt.subplots(in_channels, 3, figsize=(22, 10))
    sns.set_style("whitegrid") # or use "white" if we don't want grid lines
    sns.set_context("notebook", font_scale=1.3)
    max_time=2
    t = np.linspace(0, max_time, 50)


    n_plot_seqs = 3
    for ii in range(n_plot_seqs):
        for d in range(2):
            plot_inds_T = train_mask[ii, d, :]==1
            plot_t = t[plot_inds_T]
            axs[d, ii].scatter(plot_t, train_data[ii, d, 
                                                     :][plot_inds_T], color='k', label='true')
            axs[d, ii].plot(t, train_data[ii, d, :], 'k--')
            axs[d, ii].scatter(plot_t, gen_data[ii, d, 
                                                    :][plot_inds_T], color='r', label='predicted')
            axs[d, ii].plot(t, gen_data[ii, d, :], 'r--')
            axs[d, ii].set_xticks(np.arange(0, 2, .25))
            axs[d, ii].set_xlim([0, 2])
            axs[d, ii].legend()        

    f.savefig('results/true_vs_predicted_pvae.png')
    plt.show()

if __name__ == '__main__':
    main()

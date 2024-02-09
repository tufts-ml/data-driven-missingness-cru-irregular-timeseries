###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

# Create a synthetic dataset
from __future__ import absolute_import, division
from __future__ import print_function
import os
import matplotlib
if os.path.exists("/Users/yulia"):
	matplotlib.use('TkAgg')
else:
	matplotlib.use('Agg')

import numpy as np
import numpy.random as npr
from scipy.special import expit as sigmoid
import pickle
import matplotlib.pyplot as plt
import matplotlib.image
import torch
import lib.utils as utils
from torch.distributions.bernoulli import Bernoulli
from sklearn import model_selection
from torch.utils.data import DataLoader, TensorDataset

# ======================================================================================

def get_next_val(init, t, tmin, tmax, final = None):
	if final is None:
		return init
	val = init + (final - init) / (tmax - tmin) * t
	return val


def generate_periodic(time_steps, init_freq, init_amplitude, starting_point, 
	final_freq = None, final_amplitude = None, phi_offset = 0.):

	tmin = time_steps.min()
	tmax = time_steps.max()

	data = []
	t_prev = time_steps[0]
	phi = phi_offset
	for t in time_steps:
		dt = t - t_prev
		amp = get_next_val(init_amplitude, t, tmin, tmax, final_amplitude)
		freq = get_next_val(init_freq, t, tmin, tmax, final_freq)
		phi = phi + 2 * np.pi * freq * dt # integrate to get phase

		y = amp * np.sin(phi) + starting_point
		t_prev = t
		data.append([t,y])
	return np.array(data)

def assign_value_or_sample(value, sampling_interval = [0.,1.]):
	if value is None:
		int_length = sampling_interval[1] - sampling_interval[0]
		return np.random.random() * int_length + sampling_interval[0]
	else:
		return value

    
def irregularly_sampled_data_gen_square_sine_2(n=1000, length=50, seed=0, obs_proba=0.0006):
    np.random.seed(seed)
    # obs_times = obs_times_gen(n)
    P=25
    D=12
    obs_values, ground_truth, obs_times, masks = [], [], [], []
    sampling_rate = 25
    for i in range(n):
        t_raw = np.arange(length)
        np.random.seed(i)
        b = np.random.uniform(0, 50, 1)
        t1 = t_raw/sampling_rate
        t2 = t_raw/sampling_rate

        f1 = (np.ceil(t1*sampling_rate)+b)%P < D
        mask_0 = f1<.5
        mask_1 = f1>.5
        freq=2
        phase = b/sampling_rate
        f2_clean = mask_1*np.sin(2*np.pi*freq*t2+phase)+mask_0*np.cos(2*np.pi*freq*t2+phase)*0.4
        f2 = f2_clean
        obs_times.append(np.stack((t1, t2), axis=0))
        obs_values.append(np.stack((f1, f2), axis=0))
        #obs_values.append([f1.tolist(), f2.tolist(), f3.tolist()])
        fg1 = np.arange(length) % P < D
        fg2 = f2_clean
        #ground_truth.append([f1.tolist(), f2.tolist(), f3.tolist()])
        ground_truth.append(np.stack((fg1, fg2), axis=0))

        torch.manual_seed(i)
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


    return torch.from_numpy(np.transpose(obs_values, (0, 2, 1))), ground_truth, torch.from_numpy(np.array(obs_times)[0, 0, :]), torch.from_numpy(np.transpose(masks, (0, 2, 1)))

'''
def get_toy_data_mnar_square_sine(args={}, N=1000, obs_proba=0.0006):
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
                    "train_dataloader": utils.inf_generator(train_dataloader),
                    "test_dataloader": utils.inf_generator(test_dataloader),
                    "input_dim": dim,
                    "ground_truth": np.array(ground_truth),
                    "n_train_batches" : len(train_dataloader),
                    "n_test_batches" : len(test_dataloader)}
    return data_objects    
'''

class TimeSeries:
	def __init__(self, device = torch.device("cpu")):
		self.device = device
		self.z0 = None

	def init_visualization(self):
		self.fig = plt.figure(figsize=(10, 4), facecolor='white')
		self.ax = self.fig.add_subplot(111, frameon=False)
		plt.show(block=False)

	def visualize(self, truth):
		self.ax.plot(truth[:,0], truth[:,1])

	def add_noise(self, traj_list, time_steps, noise_weight):
		n_samples = traj_list.size(0)

		# Add noise to all the points except the first point
		n_tp = len(time_steps) - 1
		noise = np.random.sample((n_samples, n_tp))
		noise = torch.Tensor(noise).to(self.device)

		traj_list_w_noise = traj_list.clone()
		# Dimension [:,:,0] is a time dimension -- do not add noise to that
		traj_list_w_noise[:,1:,0] += noise_weight * noise
		return traj_list_w_noise



class Periodic_1d(TimeSeries):
	def __init__(self, device = torch.device("cpu"), 
		init_freq = 0.3, init_amplitude = 1.,
		final_amplitude = 10., final_freq = 1., 
		z0 = 0.):
		"""
		If some of the parameters (init_freq, init_amplitude, final_amplitude, final_freq) is not provided, it is randomly sampled.
		For now, all the time series share the time points and the starting point.
		"""
		super(Periodic_1d, self).__init__(device)
		
		self.init_freq = init_freq
		self.init_amplitude = init_amplitude
		self.final_amplitude = final_amplitude
		self.final_freq = final_freq
		self.z0 = z0

	def sample_traj(self, time_steps, n_samples = 1, noise_weight = 1.,
		cut_out_section = None):
		"""
		Sample periodic functions. 
		"""
		traj_list = []
		for i in range(n_samples):
			init_freq = assign_value_or_sample(self.init_freq, [0.4,0.8])
			if self.final_freq is None:
				final_freq = init_freq
			else:
				final_freq = assign_value_or_sample(self.final_freq, [0.4,0.8])
			init_amplitude = assign_value_or_sample(self.init_amplitude, [0.,1.])
			final_amplitude = assign_value_or_sample(self.final_amplitude, [0.,1.])

			noisy_z0 = self.z0 + np.random.normal(loc=0., scale=0.1)

			traj = generate_periodic(time_steps, init_freq = init_freq, 
				init_amplitude = init_amplitude, starting_point = noisy_z0, 
				final_amplitude = final_amplitude, final_freq = final_freq)

			# Cut the time dimension
			traj = np.expand_dims(traj[:,1:], 0)
			traj_list.append(traj)

		# shape: [n_samples, n_timesteps, 2]
		# traj_list[:,:,0] -- time stamps
		# traj_list[:,:,1] -- values at the time stamps
		traj_list = np.array(traj_list)
		traj_list = torch.Tensor().new_tensor(traj_list, device = self.device)
		traj_list = traj_list.squeeze(1)

		traj_list = self.add_noise(traj_list, time_steps, noise_weight)
		return traj_list


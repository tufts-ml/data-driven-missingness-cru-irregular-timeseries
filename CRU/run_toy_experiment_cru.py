# Modeling Irregular Time Series with Continuous Recurrent Units (CRUs)
# Copyright (c) 2022 Robert Bosch GmbH
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
from lib.models import load_model
import torch
import datetime
import numpy as np
import sys
import os
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.bernoulli import Bernoulli
from sklearn import model_selection


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


def create_dataloader_for_cru(n_seqs=200, obs_proba=0.0006): 
    args = {}
    data_obj = get_toy_data_mnar_square_sine(args, N=n_seqs, obs_proba=obs_proba)

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    dim = data_obj["input_dim"]
    device='cpu'

    for train_batch in train_loader:
        train_batch = train_batch.to(device)
        batch_len  = train_batch.shape[0]
        train_X_NTD, train_mask_obs_NTD, train_times_NT \
            = train_batch[:, :, :dim].double(), train_batch[:, :, dim:2*dim], train_batch[:, :, -1].double()
        train_mask_times_NT = train_times_NT<1.5
        train_mask_obs_NTD[~train_mask_times_NT, :]=0
        train_eval_mask_times = train_times_NT>=0
        train_y_N = torch.ones(len(train_X_NTD))


    for test_batch in test_loader:
        test_batch = test_batch.to(device)
        batch_len  = test_batch.shape[0]
        test_X_NTD, test_mask_obs_NTD, test_times_NT \
            = test_batch[:, :, :dim].double(), test_batch[:, :, dim:2*dim], test_batch[:, :, -1].double()
        test_mask_times_NT = test_times_NT<1.5
        test_mask_obs_NTD[~test_mask_times_NT, :]=0
        
        test_eval_mask_times = test_times_NT>=0
        test_y_N = torch.ones(len(test_X_NTD))


    ds = 1
    N = train_X_NTD.shape[0]#15000

    train_dataset = data.TensorDataset(train_X_NTD[:N, ::ds, :], train_y_N[:N], 
                                       train_eval_mask_times[:N, ::ds], train_times_NT[:N, ::ds], 
                                       torch.ones_like(train_y_N[:N]).bool(),
                                       train_mask_obs_NTD[:N, ::ds, :])


    test_dataset = data.TensorDataset(test_X_NTD[:, ::ds, :], test_y_N, 
                                      test_eval_mask_times[:, ::ds], 
                                      test_times_NT[:, ::ds],
                                      torch.ones_like(test_y_N).bool(),
                                      test_mask_obs_NTD[:, ::ds, :])


    train_dl = data.DataLoader(train_dataset, batch_size=160, shuffle=False, num_workers=2, drop_last=False)
    test_dl = data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2, drop_last=False)
    
    return train_dl, test_dl



parser = argparse.ArgumentParser('CRU')
# train configs
parser.add_argument('--epochs', type=int, default=75, help="Number of epochs.")
parser.add_argument('--lr',  type=float, default=0.0075, help="Learning rate.")
parser.add_argument('--lr-decay',  type=float, default=1, help="Learning rate decay.")
parser.add_argument('--weight-decay',  type=float, default=0, help="Weight decay.")
parser.add_argument('-b', '--batch-size', type=int, default=128, help="Batch size for training and test set.")
parser.add_argument('--grad-clip',  action='store_true', help="If to use gradient clipping.")
parser.add_argument('--ts', type=float, default=1, help="Scaling factor of timestamps for numerical stability.")
parser.add_argument('--save-intermediates', type=str, default=None, help="Directory path for saving model intermediates (post_mean, post_cov, prior_mean, prior_cov, kalman_gain, y, y_var). If None, no intermediates are saved.")
parser.add_argument('--log-rythm', type=int, default=20, help="Save heatmaps of model intermediates to tensorboard every log-rythm epoch. Ignored if tensorboard not used.")
parser.add_argument('--task', default='seq_gen', type=str, help="Possible tasks are interpolation, extrapolation, regression, one_step_ahead_prediction.")
parser.add_argument('--anomaly-detection',  action='store_true', help="If to trace NaN values in backpropagation for debugging.")
parser.add_argument('--tensorboard',  action='store_true', help="If to use tensorboard for logging additional to standard logger.")
# CRU transition model 
parser.add_argument('-lsd', '--latent-state-dim', type=int, default=4, help="Latent state dimension. Accepts only even values because latent observation dimenions = latent state dimension / 2")
parser.add_argument('--hidden-units', type=int, default=60, help="Hidden units of encoder and decoder.")
parser.add_argument('--num-basis', type=int, default=10, help="Number of basis matrices to use in transition model for locally-linear transitions. K in paper")
parser.add_argument('--bandwidth', type=int, default=3, help="Bandwidth for basis matrices A_k. b in paper")
parser.add_argument('--enc-var-activation', type=str, default='square', help="Variance activation function in encoder. Possible values elup1, exp, relu, square")
parser.add_argument('--dec-var-activation', type=str, default='exp', help="Variance activation function in decoder. Possible values elup1, exp, relu, square")
parser.add_argument('--trans-net-hidden-activation', type=str, default='tanh', help="Activation function for transition net.")
parser.add_argument('--trans-net-hidden-units', type=list, default=[], help="Hidden units of transition net.")
parser.add_argument('--trans-var-activation', type=str, default='relu', help="Activation function for transition net.")
parser.add_argument('--learn-trans-covar', type=bool, default=True, help="If to learn transition covariance.")
parser.add_argument('--learn-initial-state-covar', action='store_true', help="If to learn the initial state covariance.")
parser.add_argument('--initial-state-covar', type=int, default=1, help="Value of initial state covariance.")
parser.add_argument('--trans-covar', type=float, default=0.1, help="Value of initial transition covariance.")
parser.add_argument('--t-sensitive-trans-net',  action='store_true', help="If to provide the time gap as additional input to the transition net. Used for RKN-Delta_t model in paper")
parser.add_argument('--f-cru',  type=bool, default=True, help="If to use fast transitions based on eigendecomposition of the state transitions (f-CRU variant).")
parser.add_argument('--rkn',  action='store_true', help="If to use discrete state transitions (RKN baseline).")
parser.add_argument('--orthogonal', type=bool, default=True, help="If to use orthogonal basis matrices in the f-CRU variant.")
# data configs
parser.add_argument('--dataset', type=str, default='toy_mnar', help="Dataset to use. Available datasets are physionet, mimic and toy_mnar")
parser.add_argument('--sample-rate', type=float, default=1, help='Sample time points to increase irregularity of timestamps. For example, if sample_rate=0.5 half of the time points are discarded at random in the data preprocessing.')
parser.add_argument('--impute-rate', type=float, default=None, help='Remove time points for interpolation. For example, if impute_rate=0.3 the model is given 70% of the time points and tasked to reconstruct the entire series.')
parser.add_argument('--unobserved-rate', type=float, default=0.2, help='Percentage of features to remove per timestamp in order to increase sparseness across dimensions (applied only for USHCN)')
parser.add_argument('--cut-time', type=int, default=None, help='Timepoint at which extrapolation starts.')
parser.add_argument('--num-workers', type=int, default=2, help="Number of workers to use in dataloader.")
parser.add_argument('--pin-memory', type=bool, default=True, help="If to pin memory in dataloader.")
parser.add_argument('--data-random-seed', type=int, default=0, help="Random seed for subsampling timepoints and features.")
parser.add_argument('-rs', '--random_seed', type=int, default=0, help="Random seed for initializing model parameters.")
parser.add_argument('--quantization',  type=float, default=0.016, help="quantization on the physionet dataset.")
parser.add_argument('--mnar',  type=str, default='True', help="set mnar assumption to true/false")


args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.mnar=='True':
    args.mnar=True
else:
    args.mnar=False

identifier = 'random_seed=%s-lr=%.4f-mnar=%s'%(args.random_seed, args.lr, args.mnar)

if __name__ == '__main__':

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    n_seqs = 800
    obs_proba = 0.0006
    train_dl, test_dl = create_dataloader_for_cru(obs_proba=obs_proba, n_seqs=n_seqs)
    
    model = load_model(args)
    
    perf_df = model.train(train_dl=train_dl, valid_dl=test_dl, test_dl=test_dl,
            identifier=identifier, 
            logger=None)
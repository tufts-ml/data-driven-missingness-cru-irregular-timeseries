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
#
# This source code is derived from Pytorch RKN Implementation (https://github.com/ALRhub/rkn_share)
# Copyright (c) 2021 Philipp Becker (Autonomous Learning Robots Lab @ KIT)
# licensed under MIT License
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import torch
import numpy as np
import time as t
from datetime import datetime
import os
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
from lib.utils import TimeDistributed, log_to_tensorboard, make_dir
from lib.encoder import Encoder
from lib.decoder import SplitDiagGaussianDecoder, BernoulliDecoder
from lib.CRULayer import CRULayer
from lib.CRUCell import var_activation, var_activation_inverse
from lib.losses import rmse, mse, GaussianNegLogLik, bernoulli_nll
from lib.data_utils import  align_output_and_target, adjust_obs_for_extrapolation
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

optim = torch.optim
nn = torch.nn


# taken from https://github.com/ALRhub/rkn_share/ and modified
class CRU(nn.Module):

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def __init__(self, target_dim: int, lsd: int, args, use_cuda_if_available: bool = True, bernoulli_output: bool = False):
        """
        :param target_dim: output dimension
        :param lsd: latent state dimension
        :param args: parsed arguments
        :param use_cuda_if_available: if to use cuda or cpu
        :param use_bernoulli_output: if to use a convolutional decoder (for image data)
        """
        super().__init__()
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")

        self._lsd = lsd
        if self._lsd % 2 == 0:
            self._lod = int(self._lsd / 2) 
        else:
            raise Exception('Latent state dimension must be even number.')
        self.args = args

        # parameters TODO: Make configurable
        self._enc_out_normalization = "pre"
        self._initial_state_variance = 10.0
        self._learning_rate = self.args.lr
        self.bernoulli_output = bernoulli_output
        # main model

        self._cru_layer = CRULayer(
            latent_obs_dim=self._lod, args=args).to(self._device)

        Encoder._build_hidden_layers = self._build_enc_hidden_layers
        enc = Encoder(self._lod, output_normalization=self._enc_out_normalization,
                      enc_var_activation=args.enc_var_activation).to(dtype=torch.float64)
        
        SplitDiagGaussianDecoder._build_hidden_layers_mean = self._build_dec_hidden_layers_mean
        SplitDiagGaussianDecoder._build_hidden_layers_var = self._build_dec_hidden_layers_var
        
        if bernoulli_output:
            BernoulliDecoder._build_hidden_layers = self._build_dec_hidden_layers
            self._dec = TimeDistributed(BernoulliDecoder(self._lod, out_dim=1, args=args).to(
                self._device, dtype=torch.float64), num_outputs=1, low_mem=True)
            self._input_dec = TimeDistributed(SplitDiagGaussianDecoder(self._lod, out_dim=target_dim, dec_var_activation=args.dec_var_activation).to(
                dtype=torch.float64), num_outputs=2).to(self._device)
            
            self._enc = TimeDistributed(
                enc, num_outputs=2, low_mem=True).to(self._device)
        else:
            self._dec = TimeDistributed(SplitDiagGaussianDecoder(self._lod, out_dim=target_dim, dec_var_activation=args.dec_var_activation).to(
                dtype=torch.float64), num_outputs=2).to(self._device)
            
        if self.args.mnar:
            BernoulliDecoder._build_hidden_layers = self._build_dec_hidden_layers
            self._mnar_dec = TimeDistributed(BernoulliDecoder(self._lod, out_dim=target_dim, args=args).to(
            self._device, dtype=torch.float64), num_outputs=1, low_mem=True)
            
        self._enc = TimeDistributed(enc, num_outputs=2).to(self._device)
        
        # build (default) initial state
        self._initial_mean = torch.zeros(1, self._lsd).to(
            self._device, dtype=torch.float64)
        log_ic_init = var_activation_inverse(self._initial_state_variance)
        self._log_icu = torch.nn.Parameter(
            log_ic_init * torch.ones(1, self._lod).to(self._device, dtype=torch.float64))
        self._log_icl = torch.nn.Parameter(
            log_ic_init * torch.ones(1, self._lod).to(self._device, dtype=torch.float64))
        self._ics = torch.zeros(1, self._lod).to(
            self._device, dtype=torch.float64)

        # params and optimizer
        self._params = list(self._enc.parameters())
        self._params += list(self._cru_layer.parameters())
        self._params += list(self._dec.parameters())
        if self.args.mnar:
            self._params += list(self._mnar_dec.parameters())
        
        if bernoulli_output:
            self._params += list(self._input_dec.parameters())
        self._params += [self._log_icu, self._log_icl]

        self._optimizer = optim.Adam(self._params, lr=self.args.lr)
        self._shuffle_rng = np.random.RandomState(
            42)  # rng for shuffling batches

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _build_enc_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for encoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _build_dec_hidden_layers_mean(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for mean decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _build_dec_hidden_layers_var(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for variance decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError
    
    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def forward(self, obs_batch: torch.Tensor, time_points: torch.Tensor = None, obs_valid: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass on a batch
        :param obs_batch: batch of observation sequences
        :param time_points: timestamps of observations
        :param obs_valid: boolean if timestamp contains valid observation 
        """
        y, y_var = self._enc(obs_batch)
        post_mean, post_cov, prior_mean, prior_cov, kalman_gain = self._cru_layer(y, y_var, self._initial_mean,
                                                                                    [var_activation(self._log_icu), var_activation(
                                                                                        self._log_icl), self._ics],
                                                                                    obs_valid=obs_valid, time_points=time_points)
        # output an image
        if self.bernoulli_output:
            '''
            out_mean = self._dec(post_mean)
            out_var = None
            '''
            out_mean, out_var = self._input_dec(
                prior_mean, torch.cat(prior_cov, dim=-1))
            
            out_pred = self._dec(post_mean)
        # output prediction for the next time step
        elif self.args.task == 'one_step_ahead_prediction':
            out_mean, out_var = self._dec(
                prior_mean, torch.cat(prior_cov, dim=-1))

        # output filtered observation
        else:
            out_mean, out_var = self._dec(
                post_mean, torch.cat(post_cov, dim=-1))
            
            out_pred=None
        if self.args.mnar:
            out_mask_mean = self._mnar_dec(post_mean)
        
        else:
            out_mask_mean=None
        intermediates = {
            'post_mean': post_mean,
            'post_cov': post_cov,
            'prior_mean': prior_mean,
            'prior_cov': prior_cov,
            'kalman_gain': kalman_gain,
            'y': y,
            'y_var': y_var,
            'mnar_mask_probas' : out_mask_mean,
        }

        return out_mean, out_var, out_mask_mean, out_pred, intermediates
#         else:
#             return out_mean, out_var, intermediates

    # new code component
    def interpolation(self, data, track_gradient=True):
        """Computes loss on interpolation task

        :param data: batch of data
        :param track_gradient: if to track gradient for backpropagation
        :return: loss, outputs, inputs, intermediate variables, metrics on imputed points
        """
        if self.bernoulli_output:
            obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
                j.to(self._device) for j in data]
            mask_obs = None
        else:
            obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
                j.to(self._device) for j in data]

        obs_times = self.args.ts * obs_times

        with torch.set_grad_enabled(track_gradient):
            output_mean_NT, output_var_NT, intermediates = self.forward(
                obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)
            
            if self.bernoulli_output:
                output_mean = output_mean_NT[:, -1, 0]
                loss = bernoulli_nll(truth, output_mean, uint8_targets=False)
                mask_imput = (~obs_valid[...,None, None, None]) * mask_truth
                imput_loss = np.nan #TODO: compute bernoulli loss on imputed points
                imput_mse = mse(truth, output_mean, mask=mask_truth)
#                 imput_mse = mse(truth.flatten(start_dim=2), output_mean.flatten(start_dim=2), mask=mask_imput.flatten(start_dim=2))
            else:
                output_mean = output_mean_NT[:, -1, 0]
                output_var = output_var_NT[:, -1, 0]
                loss = GaussianNegLogLik(
                    output_mean, truth, output_var, mask=mask_truth)
                # compute metric on imputed points only
                mask_imput = (~obs_valid[...,None]) * mask_truth
                imput_loss = GaussianNegLogLik(output_mean, truth, output_var, mask=mask_imput)
#                 imput_mse = mse(truth, output_mean, mask=mask_imput)
                imput_mse = mse(truth, output_mean, mask=mask_truth)
        
        return loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse

    
    def seq_generation(self, data, track_gradient=True):
        """Computes loss for sequence generation

        :param data: batch of data
        :param track_gradient: if to track gradient for backpropagation
        :return: loss, outputs, inputs, intermediate variables, metrics on imputed points
        """
        if self.bernoulli_output:
            obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
                j.to(self._device) for j in data]
#             mask_obs = None
        else:
            obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
                j.to(self._device) for j in data]
        
        obs_times = self.args.ts * obs_times
        
        with torch.set_grad_enabled(track_gradient):
#             if self.args.mnar:
            output_mean_NT, output_var_NT, output_mask_proba_NT, out_pred_NT, intermediates = self.forward(
            obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)                
#             else:
#                 output_mean_NT, output_var_NT, intermediates = self.forward(
#                 obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)
            
            if self.bernoulli_output:
                output_mean = output_mean_NT[:, -1, 0]
                loss = bernoulli_nll(truth, output_mean, uint8_targets=False)
                mask_imput = (~obs_valid[...,None, None, None]) * mask_truth
                imput_loss = np.nan #TODO: compute bernoulli loss on imputed points
                imput_mse = mse(truth, output_mean, mask=mask_truth)
#                 imput_mse = mse(truth.flatten(start_dim=2), output_mean.flatten(start_dim=2), mask=mask_imput.flatten(start_dim=2))
            else:
                output_mean = output_mean_NT
                output_var = output_var_NT
                
                if self.args.mnar:
                    loss = GaussianNegLogLik(output_mean, obs, 
                                             output_var, mask=mask_obs) + bernoulli_nll(mask_obs, 
                                                                                        output_mask_proba_NT,
                                                                                        uint8_targets=False)
                else:
                    loss = GaussianNegLogLik(
                        output_mean, obs, output_var, mask=mask_obs)

                imput_loss = loss
                imput_mse = mse(obs, output_mean, mask=mask_obs)
                
        return loss, output_mean, output_var, obs, obs, mask_obs, mask_obs, intermediates, imput_loss, imput_mse

    
    
    
    # new code component
    def classification(self, data, track_gradient=True):
        """Computes loss on per seq classification task

        :param data: batch of data
        :param track_gradient: if to track gradient for backpropagation
        :return: loss, outputs, inputs, intermediate variables, metrics on imputed points
        """
        if self.bernoulli_output:
            obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
                j.to(self._device) for j in data]
#             mask_obs = None
        else:
            obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
                j.to(self._device) for j in data]

        obs_times = self.args.ts * obs_times

        with torch.set_grad_enabled(track_gradient):
#             if self.args.mnar:
            output_mean_NT, output_var_NT, output_mask_proba_NT, out_pred_NT, intermediates = self.forward(
            obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)                
#         else:
#             output_mean_NT, output_var_NT, intermediates = self.forward(
#             obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)
            '''
            output_mean_NT, output_var_NT, intermediates = self.forward(
                obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)
            '''
            
            if self.bernoulli_output:
                                
#                 end_inds = torch.argmax(torch.sum(mask_obs, axis=1), axis=1)
#                 N = out_pred_NT.shape[0]
#                 output_mean = out_pred_NT[torch.arange(N), end_inds, 0]
                
                # get indices of the last observed value for each sequence
                output_mean = out_pred_NT[:, -1, 0]
                if self.args.mnar:
                    loss = bernoulli_nll(truth, output_mean, uint8_targets=False) + GaussianNegLogLik(output_mean_NT, obs, 
                                                                                                      output_var_NT, 
                                                                                                      mask=mask_obs) +\
                    bernoulli_nll(mask_obs, output_mask_proba_NT, uint8_targets=False) 
                    
                else:
                    # loss = classification loss + reconstruction loss
                    loss = bernoulli_nll(truth, output_mean, uint8_targets=False) + GaussianNegLogLik(output_mean_NT, obs, 
                                                                                                  output_var_NT, 
                                                                                                  mask=mask_obs
                                                                                                 )
                mask_imput = (~obs_valid[...,None, None, None]) * mask_truth
                imput_loss = np.nan #TODO: compute bernoulli loss on imputed points
                imput_mse = mse(obs, output_mean_NT, mask=mask_obs)
            else:
                output_mean = output_mean_NT[:, -1, 0]
                output_var = output_var_NT[:, -1, 0]
                loss = GaussianNegLogLik(
                    output_mean, truth, output_var, mask=mask_truth)
                # compute metric on imputed points only
                mask_imput = (~obs_valid[...,None]) * mask_truth
                imput_loss = GaussianNegLogLik(output_mean, truth, output_var, mask=mask_imput)
#                 imput_mse = mse(truth, output_mean, mask=mask_imput)
                imput_mse = mse(obs, output_mean_NT, mask=mask_obs)
        
        return loss, output_mean, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse
    
    '''
    # new code component
    def extrapolation(self, data, track_gradient=True):
        """Computes loss on extrapolation task

        :param data: batch of data
        :param track_gradient: if to track gradient for backpropagation
        :return: loss, outputs, inputs, intermediate variables, metrics on imputed points
        """
        obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
            j.to(self._device) for j in data]

#         obs, obs_valid = adjust_obs_for_extrapolation(
#             obs, obs_valid, obs_times, self.args.cut_time)
        obs_times = self.args.ts * obs_times
        
        eval_inds = obs_times>=self.args.cut_time
        mask_interp = mask_obs*1
        mask_interp[eval_inds]=0
        
        mask_eval = np.logical_and(np.logical_not(mask_interp), mask_obs)
        
        
        
#         mask_obs[eval_inds,:]=0
        
        
        with torch.set_grad_enabled(track_gradient):
            output_mean_NT, output_var_NT, output_mask_proba_NT, out_pred_NT, intermediates = self.forward(
                obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)
            
            if self.args.mnar:
                    loss = GaussianNegLogLik(output_mean_NT, obs, 
                                             output_var_NT, mask=mask_obs) + bernoulli_nll(mask_obs, 
                                                                                        output_mask_proba_NT,
                                                                                        uint8_targets=False)
            else:
                    loss = GaussianNegLogLik(
                        output_mean_NT, obs, output_var_NT, mask=mask_obs)
            
#             loss = GaussianNegLogLik(
#                 output_mean_NT, truth, output_var_NT, mask=mask_truth)
            
#             # compute metric on imputed points only
#             mask_imput = (~obs_valid[..., None]) * mask_truth
#             imput_loss = GaussianNegLogLik(
#                 output_mean, truth, output_var, mask=mask_imput)
#             imput_mse = mse(truth, output_mean, mask=mask_imput)
            
            imput_loss = loss
            imput_mse = mse(obs, output_mean_NT, mask=mask_interp)
            
            intermediates['mask_eval']=mask_eval
        return loss, output_mean_NT, output_var_NT, obs, obs, mask_obs, mask_obs, intermediates, imput_loss, imput_mse
    '''
    
    # new code component
    def extrapolation(self, data, track_gradient=True):
        """Computes loss on extrapolation task

        :param data: batch of data
        :param track_gradient: if to track gradient for backpropagation
        :return: loss, outputs, inputs, intermediate variables, metrics on imputed points
        """
        obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
            j.to(self._device) for j in data]
        obs, obs_valid = adjust_obs_for_extrapolation(
            obs, obs_valid, obs_times, self.args.cut_time)
                
        obs_times = self.args.ts * obs_times
        
        
        
        
        with torch.set_grad_enabled(track_gradient):
            output_mean, output_var, output_mask_proba_NT, out_pred_NT, intermediates = self.forward(
                            obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)
            
            if self.args.mnar:
                loss = GaussianNegLogLik(output_mean, truth, 
                                             output_var, mask=mask_truth) + bernoulli_nll(mask_truth, 
                                                                                        output_mask_proba_NT,
                                                                                        uint8_targets=False)
            else:
                loss = GaussianNegLogLik(
                    output_mean, truth, output_var, mask=mask_truth)
            
            # compute metric on imputed points only
            mask_imput = (~obs_valid[..., None]) * mask_truth
            imput_loss = GaussianNegLogLik(
                output_mean, truth, output_var, mask=mask_imput)
            imput_mse = mse(truth, output_mean, mask=mask_imput)
            
            intermediates['mask_eval']=mask_imput
            intermediates['obs_times']=obs_times/self.args.ts
        return loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse    

    # new code component
    def regression(self, data, track_gradient=True):
        """Computes loss for sequence generation

        :param data: batch of data
        :param track_gradient: if to track gradient for backpropagation
        :return: loss, outputs, inputs, intermediate variables, metrics on imputed points
        """
        if self.bernoulli_output:
            obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
                j.to(self._device) for j in data]
        else:
            obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
                j.to(self._device) for j in data]
        
        obs_times = self.args.ts * obs_times
        
        with torch.set_grad_enabled(track_gradient):
            output_mean_NT, output_var_NT, output_mask_proba_NT, out_pred_NT, intermediates = self.forward(
            obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)                
            
            if self.bernoulli_output:
                output_mean = output_mean_NT[:, -1, 0]
                loss = bernoulli_nll(truth, output_mean, uint8_targets=False)
                mask_imput = (~obs_valid[...,None, None, None]) * mask_truth
                imput_loss = np.nan #TODO: compute bernoulli loss on imputed points
                imput_mse = mse(truth, output_mean, mask=mask_truth)
            else:
                output_mean = output_mean_NT
                output_var = output_var_NT
                
                if self.args.mnar:
                    loss = GaussianNegLogLik(output_mean, obs, 
                                             output_var, mask=mask_obs) + bernoulli_nll(mask_obs, 
                                                                                        output_mask_proba_NT,
                                                                                        uint8_targets=False)
                else:
                    loss = GaussianNegLogLik(
                        output_mean, obs, output_var, mask=mask_obs)

                imput_loss = loss
                imput_mse = mse(obs, output_mean, mask=mask_obs)
                
        return loss, output_mean, output_var, obs, obs, mask_obs, mask_obs, intermediates, imput_loss, imput_mse

    # new code component
    def one_step_ahead_prediction(self, data, track_gradient=True):
        """Computes loss on one-step-ahead prediction

        :param data: batch of data
        :param track_gradient: if to track gradient for backpropagation
        :return: loss, input, intermediate variables and computed output
        """
        obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
            j.to(self._device) for j in data]
        obs_times = self.args.ts * obs_times
        with torch.set_grad_enabled(track_gradient):
            output_mean, output_var, intermediates = self.forward(
                obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)
            output_mean, output_var, truth, mask_truth = align_output_and_target(
                output_mean, output_var, truth, mask_truth)
            loss = GaussianNegLogLik(
                output_mean, truth, output_var, mask=mask_truth)

        return loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates

    # new code component
    def train_epoch(self, dl, optimizer):
        """Trains model for one epoch 

        :param dl: dataloader containing training data
        :param optimizer: optimizer to use for training
        :return: evaluation metrics, computed output, input, intermediate variables
        """
        epoch_ll = 0
        epoch_rmse = 0
        epoch_mse = 0
        epoch_mse_oob = 0
        N = 0
        N_oob = 0
        if self.args.save_intermediates is not None:
            mask_obs_epoch = []
            intermediates_epoch = []

        if self.args.task == 'extrapolation' or self.args.task == 'interpolation':
            epoch_imput_ll = 0
            epoch_imput_mse = 0
        
        y_list = []
        y_pred_list = []
        for i, data in enumerate(dl):
            
            if self.args.task == 'interpolation':
                loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse = self.interpolation(
                    data)
            elif self.args.task == 'classification':
                loss, output_mean, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse = self.classification(
                    data)
                output_var=None

            elif self.args.task == 'extrapolation':
                loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse = self.extrapolation(
                    data)
            elif self.args.task == 'seq_gen':
                loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse = self.seq_generation(
                    data)

            elif self.args.task == 'regression':
                loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse = self.regression(
                    data)

            elif self.args.task == 'one_step_ahead_prediction':
                loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates = self.one_step_ahead_prediction(
                    data)

            else:
                raise Exception('Unknown task')

            # check for NaNs
            if torch.any(torch.isnan(loss)):
                print('--NAN in loss')
            for name, par in self.named_parameters():
                if torch.any(torch.isnan(par)):
                    print('--NAN before optimiser step in parameter ', name)
            torch.autograd.set_detect_anomaly(
                self.args.anomaly_detection)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            if self.args.grad_clip:
                nn.utils.clip_grad_norm_(self.parameters(), 1)
            optimizer.step()

            # check for NaNs in gradient
            for name, par in self.named_parameters():
                if par.grad is None:
                    continue
                if torch.any(torch.isnan(par.grad)):
                    print('--NAN in gradient ', name)
                if torch.any(torch.isnan(par)):
                    print('--NAN after optimiser step in parameter ', name)

            # aggregate metrics and intermediates over entire epoch
            epoch_ll += loss.item()
            epoch_rmse += rmse(truth, output_mean, mask_truth).item()
            epoch_mse += mse(truth, output_mean, mask_truth).item()
            
            print('Done with batch %d'%i)
            
            # get the mse in the oob region
            
            if self.args.task == 'extrapolation':
                mask_oob=intermediates['mask_eval']
            else:
                mask_oob = np.logical_or(truth<=-.4, truth>=.4)
            mask_oob_obs = np.logical_and(mask_oob, mask_truth)
            
            if mask_oob_obs.sum()>0:
                epoch_mse_oob += mse(truth, output_mean, mask_oob_obs).item()
                N_oob += mask_oob_obs.sum()
            
            
            N+=len(truth)
            if self.args.task == 'extrapolation' or self.args.task == 'interpolation' or self.args.task == 'regression':
                epoch_imput_ll += imput_loss.item()
                epoch_imput_mse += imput_mse.item()
                imput_metrics = [epoch_imput_ll/(i+1), epoch_imput_mse/(i+1)]
            else:
                imput_metrics = None
            
            if self.args.task == 'classification':
                y_list.append(truth)
                y_pred_list.append(output_mean)
            
            if self.args.save_intermediates is not None:
                intermediates_epoch.append(intermediates)
                mask_obs_epoch.append(mask_obs)
        
        if self.args.task == 'classification':
            y_N = torch.hstack(y_list)
            y_pred_N = torch.hstack(y_pred_list).detach().numpy()

            roc_auc = roc_auc_score(y_N, y_pred_N)
            auprc = average_precision_score(y_N, y_pred_N)
        else:
            roc_auc = None
            auprc = None    
        # save for plotting
        if self.args.save_intermediates is not None:
            torch.save(mask_obs_epoch, os.path.join(
                self.args.save_intermediates, 'train_mask_obs.pt'))
            torch.save(intermediates_epoch, os.path.join(
                self.args.save_intermediates, 'train_intermediates.pt'))
        return epoch_ll/N, epoch_rmse/N, epoch_mse/N, [output_mean, output_var], intermediates, [obs, truth, mask_truth], imput_metrics, roc_auc, auprc, epoch_mse_oob/(i+1)

    # new code component
    def eval_epoch(self, dl):
        """Evaluates model on the entire dataset

        :param dl: dataloader containing validation or test data
        :return: evaluation metrics, computed output, input, intermediate variables
        """
        epoch_ll = 0
        epoch_rmse = 0
        epoch_mse = 0
        epoch_mse_oob = 0
        N = 0
        N_oob = 0
        
        if self.args.task == 'extrapolation' or self.args.task == 'interpolation':
            epoch_imput_ll = 0
            epoch_imput_mse = 0

        if self.args.save_intermediates is not None:
            mask_obs_epoch = []
            intermediates_epoch = []
        
        y_list = []
        y_pred_list = []
        N=0
        for i, data in enumerate(dl):

            if self.args.task == 'interpolation':
                loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse = self.interpolation(
                    data, track_gradient=False)
            elif self.args.task == 'classification':
                loss, output_mean, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse = self.classification(
                    data, track_gradient=False)
                output_var = None
            elif self.args.task == 'seq_gen':
                loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse = self.seq_generation(
                    data, track_gradient=False)
            elif self.args.task == 'extrapolation':
                loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse = self.extrapolation(
                    data, track_gradient=False)

            elif self.args.task == 'regression':
                loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse = self.regression(
                    data, track_gradient=False)

            elif self.args.task == 'one_step_ahead_prediction':
                loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates = self.one_step_ahead_prediction(
                    data, track_gradient=False)
            
            epoch_ll += loss.item()
            epoch_rmse += rmse(truth, output_mean, mask_truth).item()
            epoch_mse += mse(truth, output_mean, mask_truth).item()
            
            # get the mse in the oob region
            if self.args.task == 'extrapolation':
                mask_oob=intermediates['mask_eval']
            else:
                mask_oob = np.logical_or(truth<=-.4, truth>=.4)
            mask_oob_obs = np.logical_and(mask_oob, mask_truth)
            
            if mask_oob_obs.sum()>0:
                epoch_mse_oob += mse(truth, output_mean, mask_oob_obs).item()
                N_oob += mask_oob_obs.sum()
            
            
            
            N+=len(truth)
            if self.args.task == 'extrapolation' or self.args.task == 'interpolation':
                epoch_imput_ll += imput_loss.item()
                epoch_imput_mse += imput_mse.item()
                imput_metrics = [epoch_imput_ll/(i+1), epoch_imput_mse/(i+1)]
            else:
                imput_metrics = None
            
            if self.args.task == 'classification':
                y_list.append(truth)
                y_pred_list.append(output_mean)
            
            
            if self.args.save_intermediates is not None:
                intermediates_epoch.append(intermediates)
                mask_obs_epoch.append(mask_obs)
        
        if self.args.task == 'classification':
            y_N = torch.hstack(y_list)
            y_pred_N = torch.hstack(y_pred_list).detach().numpy()

            roc_auc = roc_auc_score(y_N, y_pred_N)
            auprc = average_precision_score(y_N, y_pred_N)
        else:
            roc_auc = None
            auprc = None
        
        # save for plotting
        if self.args.save_intermediates is not None:
            torch.save(output_mean, os.path.join(
                self.args.save_intermediates, 'valid_output_mean.pt'))
            torch.save(obs, os.path.join(
                self.args.save_intermediates, 'valid_obs.pt'))
            torch.save(output_var, os.path.join(
                self.args.save_intermediates, 'valid_output_var.pt'))
            torch.save(truth, os.path.join(
                self.args.save_intermediates, 'valid_truth.pt'))
            torch.save(intermediates_epoch, os.path.join(
                self.args.save_intermediates, 'valid_intermediates.pt'))
            torch.save(mask_obs_epoch, os.path.join(
                self.args.save_intermediates, 'valid_mask_obs.pt'))
        return epoch_ll/N, epoch_rmse/N, epoch_mse/N, [output_mean, output_var], intermediates, [obs, truth, mask_truth], imput_metrics, roc_auc, auprc, epoch_mse_oob/(i+1)

    # new code component
    def train(self, train_dl, valid_dl, test_dl, identifier, logger, epoch_start=0):
        """Trains model on trainset and evaluates on test data. Logs results and saves trained model.

        :param train_dl: training dataloader
        :param valid_dl: validation dataloader
        :param identifier: logger id
        :param logger: logger object
        :param epoch_start: starting epoch
        """

        optimizer = optim.Adam(self.parameters(), self.args.lr)
        def lr_update(epoch): return self.args.lr_decay ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_update)
        
        make_dir(f'../results/tensorboard/{self.args.dataset}')
        writer = SummaryWriter(f'../results/tensorboard/{self.args.dataset}/{identifier}')
        perf_dict_list = []
        

        for epoch in range(epoch_start, self.args.epochs):
            start = datetime.now()
            if logger is not None:
                logger.info(f'Epoch {epoch} of {self.args.epochs} starts: {start.strftime("%H:%M:%S")}')

            # train
            train_ll, train_rmse, train_mse, train_output, intermediates, train_input, train_imput_metrics, auc_train, auprc_train, train_mse_oob = self.train_epoch(
                train_dl, optimizer)
            end_training = datetime.now()
            if self.args.tensorboard:
                log_to_tensorboard(self, writer=writer,
                                mode='train',
                                metrics=[train_ll, train_rmse, train_mse],
                                output=train_output,
                                input=train_input,
                                intermediates=intermediates,
                                epoch=epoch,
                                imput_metrics=train_imput_metrics,
                                log_rythm=self.args.log_rythm)

            # eval
            valid_ll, valid_rmse, valid_mse, valid_output, intermediates_val, valid_input, valid_imput_metrics, auc_valid, auprc_valid, valid_mse_oob = self.eval_epoch(
                valid_dl)
            test_ll, test_rmse, test_mse, test_output, intermediates_test, test_input, test_imput_metrics, auc_test, auprc_test, test_mse_oob = self.eval_epoch(
                test_dl)
            
            if self.args.tensorboard:
                log_to_tensorboard(self, writer=writer,
                                mode='valid',
                                metrics=[valid_ll, valid_rmse, valid_mse],
                                output=valid_output,
                                input=valid_input,
                                intermediates=intermediates,
                                epoch=epoch,
                                imput_metrics=valid_imput_metrics,
                                log_rythm=self.args.log_rythm)

            end = datetime.now()
            if logger is not None:
                logger.info(f'Training epoch {epoch} took: {(end_training - start).total_seconds()}')
                logger.info(f'Epoch {epoch} took: {(end - start).total_seconds()}')
                logger.info(f' train_nll: {train_ll:3f}, train_mse: {train_mse:3f}')
                logger.info(f' valid_nll: {valid_ll:3f}, valid_mse: {valid_mse:3f}')
                logger.info(f' test_nll: {test_ll:3f}, valid_mse: {test_mse:3f}')

                if auc_train is not None:
                    logger.info(f' train_auc: {auc_train:3f}, train_auprc: {auprc_train:3f}')
                    logger.info(f' valid_auc: {auc_valid:3f}, valid_auprc: {auprc_valid:3f}')
                    logger.info(f' test_auc: {auc_test:3f}, test_auprc: {auprc_test:3f}')

                if self.args.task == 'extrapolation' or self.args.impute_rate is not None:
                    if self.bernoulli_output:
                        logger.info(f' train_mse_imput: {train_imput_metrics[1]:3f}')
                        logger.info(f' valid_mse_imput: {valid_imput_metrics[1]:3f}')
                    else:
                        logger.info(f' train_nll_imput: {train_imput_metrics[0]:3f}, train_mse_imput: {train_imput_metrics[1]:3f}')
                        logger.info(f' valid_nll_imput: {valid_imput_metrics[0]:3f}, valid_mse_imput: {valid_imput_metrics[1]:3f}')

            scheduler.step()
            perf_dict = {'epoch' : epoch,
                        'train_nll' : train_ll,
                        'valid_nll' : valid_ll,
                        'test_nll' : test_ll,
                        'train_mse' : train_mse,
                        'valid_mse' : valid_mse,
                        'test_mse' : test_mse,
                        'train_rmse' : train_rmse,
                        'valid_rmse' : valid_rmse,
                        'test_rmse' : test_rmse, 
                        'train_auc' : auc_train,
                        'valid_auc' : auc_valid,
                        'test_auc' : auc_test,
                        'train_auprc' : auprc_train,
                        'valid_auprc' : auprc_valid,
                        'test_auprc' : auprc_test,
                        'train_mse_oob' : train_mse_oob,
                        'valid_mse_oob' : valid_mse_oob,
                        'test_mse_oob' : test_mse_oob}
            
            print(perf_dict, flush=True)
            perf_dict_list.append(perf_dict)
            perf_df = pd.DataFrame(perf_dict_list)
            
            
            save_dir = 'training_results/%s'%self.args.dataset                        
            if self.args.task=='seq_gen':
                # plot some generated sequences
                d=2
                t = np.linspace(0, 2, 50)
                latent_state_dim = self.args.latent_state_dim
                n_plot_seqs = 2
                if intermediates_test['mnar_mask_probas'] is not None:
                    f, axs = plt.subplots(d+2+latent_state_dim, n_plot_seqs, figsize=(16, 8+2*latent_state_dim), sharex=True)
                    sns.set_style("whitegrid") # or use "white" if we don't want grid lines
                    sns.set_context("notebook", font_scale=1.3)
                    mask = test_input[2].detach().numpy()
                    for ii in range(n_plot_seqs):
                        true_seq = test_input[1].detach().numpy()
                        predicted_seq = test_output[0].detach().numpy() 
                        z_t_mean = intermediates_test['post_mean'][ii].detach().numpy() 
                        mnar_mask_probas = intermediates_test['mnar_mask_probas'][ii].detach().numpy()
                        for dd in range(d):
                            plot_inds_T = mask[ii, :, dd]==1
                            plot_t = t[plot_inds_T]
                            axs[dd, ii].scatter(plot_t, true_seq[ii, :, dd][plot_inds_T], color='k', 
#                                                 label='true'
                                               )
                            axs[dd, ii].plot(t, true_seq[ii, :, dd], 'k--', label='true')
                            axs[dd, ii].plot(t, predicted_seq[ii, :, dd], 'r--', label='predicted')
                            axs[dd, ii].set_xlim([0, 1.5]) 
                            axs[dd, ii].axhline(-.4, c='k')
                            axs[dd, ii].axhline(.4, c='k')
                            axs[0, ii].set_title('sequence number : %s'%(ii))
                            axs[dd+2, ii].plot(t, mnar_mask_probas[:, dd], 'k^', 
#                                                label=r'$p(s_%d(t)=1|z(t))$'%dd
                                              )
                            axs[dd+2, ii].set_ylabel(r'$p_{obs}(d=%d)$'%dd)
#                             axs[dd+2, ii].legend()
                            axs[dd+2, ii].set_ylim([0, .7])
                            axs[2, ii].set_title('Learned MNAR parameters for sequence')
                        for kk in range(latent_state_dim):
                            axs[kk+4, ii].plot(t, z_t_mean[:, kk], 'g--', 
#                                                label=r'$E[z_%s(t)]$'%kk
                                              )
                            axs[kk+4, ii].set_ylabel(r'$E[z_%s(t)]$'%kk)
                        axs[4, ii].set_title('Mean of LTI-SDE')
                
                    
                else:
                    f, axs = plt.subplots(d+latent_state_dim, 3, figsize=(22, 5+2*latent_state_dim), sharex=True) 
                    sns.set_style("whitegrid") # or use "white" if we don't want grid lines
                    sns.set_context("notebook", font_scale=1.3)
                    
                    for ii in range(3):
                        predicted_seq = test_output[0][ii].detach().numpy()                    
                        z_t_mean = intermediates_test['post_mean'][ii].detach().numpy()  
                        for dd in range(d):
                            axs[dd, ii].plot(t, predicted_seq[:, dd], 'b')
                            axs[dd, ii].set_xlim([0, 2])
                            axs[0, ii].set_title('generated sequence number : %s'%ii)
                        for kk in range(latent_state_dim):
                            axs[kk+2, ii].plot(t, z_t_mean[:, kk], 'g--', 
#                                                label=r'$E[z_%s(t)]$'%kk
                                              )
                            axs[kk+2, ii].set_title(r'$E[z_%s(t)]$'%kk)
                
                for ax in axs.flatten()[:1]:
                    ax.legend(
#                         loc='upper right', 
                              bbox_to_anchor=[.85, .85])
                
                
                f.savefig(os.path.join(save_dir, 'generated_seqs_%s_%s.png'%(self.args.dataset, identifier)), bbox_inches='tight')
                
                
                # plot the true and reconstructed sequences
                true_seq_NTD_train = train_input[1].detach().numpy()
                pred_seq_NTD_train = train_output[0].detach().numpy()
                mask_NTD_train = train_input[2].detach().numpy()

                true_seq_NTD_test = test_input[1].detach().numpy()
                pred_seq_NTD_test = test_output[0].detach().numpy()
                mask_NTD_test = test_input[2].detach().numpy()
                
                
                # plot true and predicted values for 3 sequences in train and test
                n_plot_seqs = 3

                for split, true_seq_NTD, pred_seq_NTD, mask_NTD in [('train', 
                                                                     true_seq_NTD_train, 
                                                                     pred_seq_NTD_train, 
                                                                     mask_NTD_train),
                                                                   ('test', 
                                                                    true_seq_NTD_test, 
                                                                    pred_seq_NTD_test, 
                                                                    mask_NTD_test)]:

                    f, axs = plt.subplots(2, n_plot_seqs, figsize=(22, 10), 
                          sharex=True
                         )
                    sns.set_style("whitegrid") # or use "white" if we don't want grid lines
                    sns.set_context("notebook", font_scale=1.3)
                    ticks = [-1, -.6, -.3, 0., .3, .6, 1]
                    for ii in range(n_plot_seqs):
                        for d in range(2):
                            plot_inds_T = mask_NTD[ii, :, d]==1
                            plot_t = t[plot_inds_T]
                            axs[d, ii].scatter(plot_t, true_seq_NTD[ii, :, d][plot_inds_T], color='k', label='true')
                            axs[d, ii].plot(t, true_seq_NTD[ii, :, d], 'k--')
                            axs[d, ii].scatter(plot_t, pred_seq_NTD[ii, :, d][plot_inds_T], color='r', label='predicted')
                            axs[d, ii].plot(t, pred_seq_NTD[ii, :, d], 'r--')
                            axs[d, ii].set_xticks(np.arange(0, 2, .25))
                            axs[d, ii].set_xlim([0, 2])
                            axs[d, ii].legend()
                                        
                    # get the mse in the extrapolated segment
                    extrap_t_start = np.where(t>1.5)[0][0]
                    true_extrap_vals = true_seq_NTD[:, extrap_t_start:, :].flatten()
                    pred_extrap_vals = pred_seq_NTD[:, extrap_t_start:, :].flatten()
                    
                    
                    mse_extrap = mean_squared_error(true_extrap_vals, pred_extrap_vals)
                    f.suptitle('True vs predicted (Extrap MSE : %.2f)'%(mse_extrap))
                    
                    f.savefig(os.path.join(save_dir, '%s_true_vs_predicted_%s_%s.png'%(split, 
                                                                                       self.args.dataset, 
                                                                                       identifier)))
                
                
                plt.close('all')
                perf_df.to_csv(os.path.join(save_dir, 'CRU_%s_%s.csv'%(self.args.dataset, identifier)), index=False)
                
            elif ((self.args.dataset=='physionet')|(self.args.dataset=='mimic'))&((self.args.task=='extrapolation')):
                truth = test_input[1].detach().numpy()
                output_mean = test_output[0].detach().numpy()
                mask_truth = test_input[2].detach().numpy()
                obs_times = intermediates_test['obs_times']
                mask_imput = intermediates_test['mask_eval']
                
                # get the normalization estimates for each feature
                if self.args.dataset=='physionet':
                    norm_csv = os.path.join('lib/physionet/normalization_estimates_test.csv')
                elif self.args.dataset=='mimic':
                    norm_csv = os.path.join('lib/mimic4/normalization_estimates.csv')
                norm_df = pd.read_csv(norm_csv)
#                 
                print('*********************After epoch %d**********************'%epoch)
                save_dir = os.path.join(save_dir, 'extrapolation')
                if (epoch+1)%5==0:
                    save_path = os.path.join(save_dir, 'CRU_%s.pt'%(identifier))
                    torch.save(self.state_dict(), save_path)
        
                
                for feat_ind in range(len(norm_df)):
                    curr_feat_min = norm_df.loc[feat_ind, 'min']
                    curr_feat_max = norm_df.loc[feat_ind, 'max']
                    feat_name = norm_df.loc[feat_ind, 'feature']
                    truth_unnorm = truth[:, :, feat_ind]*(curr_feat_max-curr_feat_min)+curr_feat_min
                    output_mean_unnmorm = output_mean[:, :, feat_ind]*(curr_feat_max-curr_feat_min)+curr_feat_min
                    
                    
                    
                    for patient_ind in range(20):
                        # true and predicted for the full sequence
                        all_obs_times = obs_times[patient_ind][mask_truth[patient_ind, :, feat_ind]==1]
                        if len(all_obs_times)==0:
                            continue
                        
                        all_truth = truth_unnorm[patient_ind][mask_truth[patient_ind, :, feat_ind]==1]
                        all_pred = output_mean_unnmorm[patient_ind][mask_truth[patient_ind, :, feat_ind]==1]

                        extrap_times = obs_times[patient_ind][mask_imput[patient_ind, :, feat_ind]==1]
                        extrap_truth = truth_unnorm[patient_ind][mask_imput[patient_ind, :, feat_ind]==1]
                        extrap_pred = output_mean_unnmorm[patient_ind][mask_imput[patient_ind, :, feat_ind]==1]


                        f, axs = plt.subplots(1, 1)
                        sns.set_style("whitegrid") # or use "white" if we don't want grid lines
                        sns.set_context("notebook", font_scale=1.3)   

                        axs.plot(all_obs_times, all_truth, 'bx', label='true')
                        axs.plot(all_obs_times, all_pred, 'b^', label='predicted')

                        axs.plot(extrap_times, extrap_truth, 'rx', label='true (extrapolation)')
                        axs.plot(extrap_times, extrap_pred, 'r^', label='predicted (extrapolation)')                
                        axs.set_title('Predicting %s'%feat_name)
                        axs.set_xlabel('Time')
                        axs.set_ylabel('%s'%feat_name)
                        axs.legend()

                        f.savefig(os.path.join(save_dir, 'true_vs_predicted_%s_pid=%s_feat=%s.png'%(identifier, 
                                                                           patient_ind, feat_name)))
                        
                        plt.close()
                    # add the feature-level rmse on the unnormalized scale to the performance 
                    rmse_test_extrap = rmse(torch.tensor(truth_unnorm), 
                                            output_mean_unnmorm, mask_imput[:, :, feat_ind]).item()
                    rmse_test_all = rmse(torch.tensor(truth_unnorm), 
                                            output_mean_unnmorm, torch.tensor(mask_truth[:, :, feat_ind])).item()
                    perf_df['test_rmse_extrap_%s'%feat_name]=rmse_test_extrap
                    perf_df['test_rmse_full_%s'%feat_name]=rmse_test_all
                    
                    print('================%s============='%feat_name)
                    print('RMSE Extrap (test) : %.3f'%rmse_test_extrap, flush=True)
                    print('RMSE Full(test) : %.3f'%rmse_test_all, flush=True)                
                
                perf_df.to_csv(os.path.join(save_dir, 'CRU_%s.csv'%(identifier)), index=False)                

        return perf_df     
            
#         make_dir(f'../results/models/{self.args.dataset}')
#         torch.save({'epoch': epoch,
#                     'model_state_dict': self.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'loss': train_ll,
#                     }, f'../results/models/{self.args.dataset}/{identifier}.tar')

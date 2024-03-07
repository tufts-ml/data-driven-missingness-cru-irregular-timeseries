import argparse
from lib.utils import get_logger, count_parameters
from lib.data_utils import load_data
from lib.models import load_model
import torch
import datetime
import numpy as np
import sys
import os
from torch.utils.data import DataLoader
import torch.utils.data as data
import sys
import pandas as pd
from lib.data_utils import adjust_obs_for_extrapolation
from lib.losses import rmse


parser = argparse.ArgumentParser('CRU')
# train configs
parser.add_argument('--epochs', type=int, default=5, help="Number of epochs.")
parser.add_argument('--lr',  type=float, default=0.01, help="Learning rate.")
parser.add_argument('--lr-decay',  type=float, default=1, help="Learning rate decay.")
parser.add_argument('--weight-decay',  type=float, default=0, help="Weight decay.")
parser.add_argument('-b', '--batch-size', type=int, default=128, help="Batch size for training and test set.")
parser.add_argument('--grad-clip',  action='store_true', help="If to use gradient clipping.")
parser.add_argument('--ts', type=float, default=1, help="Scaling factor of timestamps for numerical stability.")
parser.add_argument('--save-intermediates', type=str, default=None, help="Directory path for saving model intermediates (post_mean, post_cov, prior_mean, prior_cov, kalman_gain, y, y_var). If None, no intermediates are saved.")
parser.add_argument('--log-rythm', type=int, default=20, help="Save heatmaps of model intermediates to tensorboard every log-rythm epoch. Ignored if tensorboard not used.")
parser.add_argument('--task', default='regression', type=str, help="Possible tasks are interpolation, extrapolation, regression, one_step_ahead_prediction.")
parser.add_argument('--anomaly-detection',  action='store_true', help="If to trace NaN values in backpropagation for debugging.")
parser.add_argument('--tensorboard',  action='store_true', help="If to use tensorboard for logging additional to standard logger.")
# CRU transition model 
parser.add_argument('-lsd', '--latent-state-dim', type=int, default=None, help="Latent state dimension. Accepts only even values because latent observation dimenions = latent state dimension / 2")
parser.add_argument('--hidden-units', type=int, default=60, help="Hidden units of encoder and decoder.")
parser.add_argument('--num-basis', type=int, default=15, help="Number of basis matrices to use in transition model for locally-linear transitions. K in paper")
parser.add_argument('--bandwidth', type=int, default=3, help="Bandwidth for basis matrices A_k. b in paper")
parser.add_argument('--enc-var-activation', type=str, default='elup1', help="Variance activation function in encoder. Possible values elup1, exp, relu, square")
parser.add_argument('--dec-var-activation', type=str, default='elup1', help="Variance activation function in decoder. Possible values elup1, exp, relu, square")
parser.add_argument('--trans-net-hidden-activation', type=str, default='tanh', help="Activation function for transition net.")
parser.add_argument('--trans-net-hidden-units', type=list, default=[], help="Hidden units of transition net.")
parser.add_argument('--trans-var-activation', type=str, default='elup1', help="Activation function for transition net.")
parser.add_argument('--learn-trans-covar', type=bool, default=True, help="If to learn transition covariance.")
parser.add_argument('--learn-initial-state-covar', action='store_true', help="If to learn the initial state covariance.")
parser.add_argument('--initial-state-covar', type=int, default=1, help="Value of initial state covariance.")
parser.add_argument('--trans-covar', type=float, default=0.1, help="Value of initial transition covariance.")
parser.add_argument('--t-sensitive-trans-net',  action='store_true', help="If to provide the time gap as additional input to the transition net. Used for RKN-Delta_t model in paper")
parser.add_argument('--f-cru',  action='store_true', help="If to use fast transitions based on eigendecomposition of the state transitions (f-CRU variant).")
parser.add_argument('--rkn',  action='store_true', help="If to use discrete state transitions (RKN baseline).")
parser.add_argument('--orthogonal', type=bool, default=True, help="If to use orthogonal basis matrices in the f-CRU variant.")
# data configs
parser.add_argument('--dataset', type=str, default='eicu', help="Dataset to use. Available datasets are physionet, eicu and mimic and toy")
parser.add_argument('--sample-rate', type=float, default=1, help='Sample time points to increase irregularity of timestamps. For example, if sample_rate=0.5 half of the time points are discarded at random in the data preprocessing.')
parser.add_argument('--impute-rate', type=float, default=None, help='Remove time points for interpolation. For example, if impute_rate=0.3 the model is given 70% of the time points and tasked to reconstruct the entire series.')
parser.add_argument('--unobserved-rate', type=float, default=0.2, help='Percentage of features to remove per timestamp in order to increase sparseness across dimensions (applied only for USHCN)')
parser.add_argument('--cut-time', type=int, default=None, help='Timepoint at which extrapolation starts.')
parser.add_argument('--num-workers', type=int, default=2, help="Number of workers to use in dataloader.")
parser.add_argument('--pin-memory', type=bool, default=True, help="If to pin memory in dataloader.")
parser.add_argument('--data-random-seed', type=int, default=0, help="Random seed for subsampling timepoints and features.")
parser.add_argument('-rs', '--random-seed', type=int, default=0, help="Random seed for initializing model parameters.")
parser.add_argument('--quantization',  type=float, default=0.016, help="quantization on the physionet dataset.")
parser.add_argument('--mnar', type=str, help="Whether to model mnar")

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.mnar=='True':
    args.mnar=True
else:
    args.mnar=False


if args.mnar:
    identifier = 'mimic_random_seed=%s-lr=%.4f-mnar=True-basis=%d-bandwidth=%d-lsd=%d'%(args.random_seed, 
                                                                                            args.lr,
                                                                                            args.num_basis,
                                                                                            args.bandwidth,
                                                                                            args.latent_state_dim)
else:
    identifier = 'mimic_random_seed=%s-lr=%.4f-mnar=False-basis=%d-bandwidth=%d-lsd=%d'%(args.random_seed, 
                                                                                             args.lr,
                                                                                             args.num_basis,
                                                                                             args.bandwidth,
                                                                                             args.latent_state_dim)


if __name__ == '__main__':

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind+2):]
    input_command = " ".join(input_command)
    
    log_path = os.path.join("logs", args.dataset, args.task + '_' + identifier + ".log")
    if not os.path.exists(f"logs/{args.dataset}"):
        os.makedirs(f"logs/{args.dataset}")

    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    
    args.classif=True

    model = load_model(args)
        

    data_dir = 'data/classifier_train_test_split_dir'
    suffix = '_irregular_ts'
    
    train_X_NTD = torch.Tensor(np.load(os.path.join(data_dir, 'X_train%s.npy'%suffix))).double()
    train_times_NT = torch.Tensor(np.load(os.path.join(data_dir, 'train_times_NT%s.npy'%suffix))).double()
    train_mask_times_NT = torch.Tensor(np.load(os.path.join(data_dir, 'train_mask_times_NT%s.npy'%suffix))).bool()
    train_mask_obs_NTD = torch.Tensor(np.load(os.path.join(data_dir, 'train_mask_obs_NTD%s.npy'%suffix)))
    
    valid_X_NTD = torch.Tensor(np.load(os.path.join(data_dir, 'X_valid%s.npy'%suffix))).double()
    valid_times_NT = torch.Tensor(np.load(os.path.join(data_dir, 'valid_times_NT%s.npy'%suffix))).double()
    valid_mask_times_NT = torch.Tensor(np.load(os.path.join(data_dir, 'valid_mask_times_NT%s.npy'%suffix))).bool()
    valid_mask_obs_NTD = torch.Tensor(np.load(os.path.join(data_dir, 'valid_mask_obs_NTD%s.npy'%suffix)))
    
    test_X_NTD = torch.Tensor(np.load(os.path.join(data_dir, 'X_test%s.npy'%suffix))).double()
    test_times_NT = torch.Tensor(np.load(os.path.join(data_dir, 'test_times_NT%s.npy'%suffix))).double()
    test_mask_times_NT = torch.Tensor(np.load(os.path.join(data_dir, 'test_mask_times_NT%s.npy'%suffix))).bool()
    test_mask_obs_NTD = torch.Tensor(np.load(os.path.join(data_dir, 'test_mask_obs_NTD%s.npy'%suffix)))
    
    ds = 1
    N = 5000#train_X_NTD.shape[0]
    train_dataset = data.TensorDataset(train_X_NTD[:N, ::ds, :],  
                                       train_X_NTD[:N, ::ds, :],
                                       train_mask_times_NT[:N, ::ds], 
                                       train_times_NT[:N, ::ds], 
                                       train_mask_obs_NTD[:N, ::ds, :],
                                       train_mask_obs_NTD[:N, ::ds, :])
    
    valid_dataset = data.TensorDataset(valid_X_NTD[:, ::ds, :], 
                                       valid_X_NTD[:, ::ds, :],
                                       valid_mask_times_NT[:, ::ds], 
                                       valid_times_NT[:, ::ds],
                                       valid_mask_obs_NTD[:, ::ds, :],
                                       valid_mask_obs_NTD[:, ::ds, :])
    
    test_dataset = data.TensorDataset(test_X_NTD[:, ::ds, :], 
                                      test_X_NTD[:, ::ds, :], 
                                      test_mask_times_NT[:, ::ds], 
                                      test_times_NT[:, ::ds],
                                      test_mask_obs_NTD[:, ::ds, :],
                                      test_mask_obs_NTD[:, ::ds, :])
    
    
    
    
    
    
    train_dl = data.DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=2, drop_last=True)
    valid_dl = data.DataLoader(valid_dataset, batch_size=len(valid_X_NTD), shuffle=False, num_workers=2, drop_last=True)
    test_dl = data.DataLoader(test_dataset, batch_size=len(test_X_NTD), shuffle=False, num_workers=2, drop_last=True)
    
    # initialize the decoder's bias to be the mean of the training set
    counts_F = train_mask_obs_NTD.sum(dim=(0, 1))
    target_sums_F = (train_mask_obs_NTD * train_X_NTD).sum(dim=(0, 1))
    target_means_F = target_sums_F/counts_F
    model._dec.state_dict()['_module._out_layer_mean.bias']=target_means_F
    
    
    # compute baseline of training set mean
    norm_csv = os.path.join(data_dir, 'normalization_estimates.csv')
    norm_df = pd.read_csv(norm_csv)
    
    target_means_F_unnorm = torch.zeros_like(target_means_F)
    for feat_ind in range(len(norm_df)):
        curr_feat_min = norm_df.loc[feat_ind, 'min']
        curr_feat_max = norm_df.loc[feat_ind, 'max']
        target_means_F_unnorm[feat_ind] = target_means_F[feat_ind]*(curr_feat_max-curr_feat_min) + curr_feat_min
    
    
    for data in valid_dl:
        obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [j.to('cpu') for j in data]
        obs, obs_valid = adjust_obs_for_extrapolation(
                obs, obs_valid, obs_times, args.cut_time)
    
    
    #evaluate only on extrapolated segment
    mask_imput = (~obs_valid[...,None]) * mask_truth
    baseline_preds_mean = target_means_F_unnorm[np.newaxis, np.newaxis, :]*torch.ones_like(truth)
    
    feat_names = norm_df['feature'].unique()
    per_feat_baseline_rmse = dict.fromkeys(feat_names)
    
    
    truth_unnorm = torch.zeros_like(truth)
    for feat_ind in range(len(norm_df)):
        curr_feat_min = norm_df.loc[feat_ind, 'min']
        curr_feat_max = norm_df.loc[feat_ind, 'max']
        truth_unnorm[:, :, feat_ind]=truth[:, :, feat_ind]*(curr_feat_max-curr_feat_min) + curr_feat_min
        
        rmse_baseline = rmse(truth_unnorm[:, :, feat_ind], baseline_preds_mean[:, :, feat_ind], mask=mask_imput[:, :, feat_ind])
        per_feat_baseline_rmse[feat_names[feat_ind]]=rmse_baseline
    
    
    
    
    del train_X_NTD, train_mask_times_NT, train_times_NT, train_mask_obs_NTD, train_dataset
    del valid_X_NTD, valid_mask_times_NT, valid_times_NT, valid_mask_obs_NTD, valid_dataset
    del test_X_NTD, test_mask_times_NT, test_times_NT, test_mask_obs_NTD, test_dataset
    
    logger.info(f'parameters: {count_parameters(model)}')
    model.train(train_dl=train_dl, valid_dl=valid_dl, test_dl=test_dl,
                identifier=identifier, logger=logger)
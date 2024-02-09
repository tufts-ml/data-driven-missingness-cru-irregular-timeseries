import torch
import torchcde
import numpy as np
import sys
import os
import controldiffeq
import math
import seaborn as sns
import matplotlib.pyplot as plt
from torch.distributions.bernoulli import Bernoulli
from sklearn.model_selection import train_test_split


class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    def forward(self, z):
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimensions, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        return z


######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(NeuralCDE, self).__init__()
        self.hidden_channels = hidden_channels

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)

    def forward(self, times, coeffs):
        spline = controldiffeq.NaturalCubicSpline(times, coeffs)

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        z0 = self.initial(spline.evaluate(times[0]))

        ######################
        # Actually solve the CDE.
        ######################
        z_T = controldiffeq.cdeint(dX_dt=spline.derivative,
                                   z0=z0,
                                   func=self.func,
                                   t=times,
                                   atol=1e-2,
                                   rtol=1e-2)
        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
#         z_T = z_T[1]
        pred_y = self.readout(z_T)
        return pred_y
    
def get_data_square_sine(n=1000, length=50, obs_proba=0.0006):
    P=25
    D=12
    X_list, X_clean_list, obs_values, ground_truth, obs_times, masks = [], [], [], [], [], []
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
        
        ## Add MNAR missingness
        torch.manual_seed(i)
        mnar_lims = [-.4, .4]
        m1 = np.asarray(Bernoulli(torch.tensor(0.6)).sample(f1.shape)) ## Get some irregularly sampled time points
        mnar_inds_1 = np.logical_or(f1<mnar_lims[0], f1>mnar_lims[1])
        miss_1 = np.asarray(Bernoulli(torch.tensor(obs_proba)).sample(f1.shape))
        m1[mnar_inds_1] = miss_1[mnar_inds_1]
        
        # mask obs greater than 1.5 for extrapolation
        m1[t1>=1.5]=0
        
        f1_w_missing = f1.copy()
        f1_w_missing[m1==0]=np.nan
        
        m2 = np.asarray(Bernoulli(torch.tensor(0.6)).sample(f2.shape)) ## Get some irregularly sampled time points
        mnar_inds_2 = np.logical_or(f2<mnar_lims[0], f2>mnar_lims[1])
        miss_2 = np.asarray(Bernoulli(torch.tensor(obs_proba)).sample(f2.shape))
        m2[mnar_inds_2] = miss_2[mnar_inds_2]
        # mask obs greater than 1.5 for extrapolation
        m2[t2>=1.5]=0
        
        masks.append(np.stack((m1, m2), axis=0).T)
        
        f2_w_missing = f2.copy()
        f2_w_missing[m2==0]=np.nan
        
        obs_times.append(np.stack((t1, t2), axis=0))
        obs_values.append(np.stack((f1, f2), axis=0))
#         X_list.append(np.stack((t1, f1*m1, f2*m2), axis=0).T)
        X_list.append(np.stack((t1, f1_w_missing, f2_w_missing), axis=0).T)
        X_clean_list.append(np.stack((t1, f1, f2), axis=0).T)
        
        X = torch.from_numpy(np.array(X_list)).float()
        X_clean = torch.from_numpy(np.array(X_clean_list)).float()
        
        t = torch.from_numpy(t1).float()
        
    return t, X, X_clean, torch.from_numpy(np.array(masks))



if __name__ == '__main__':
    
    # load the data
    t_ss, X_ss, X_clean_ss, mask_ss = get_data_square_sine()
    
    # split into train test
    train_X_ss, test_X_ss, train_X_clean_ss, test_X_clean_ss, train_mask_ss, test_mask_ss = train_test_split(X_ss, 
                                                                                                             X_clean_ss, 
                                                                                                             mask_ss, 
                                                                                                             test_size=0.2,
                                                                                                             shuffle=False)
    train_t_ss = t_ss
    ######################
    # input_channels=3 because we have both the horizontal and vertical position of a point in the spiral, and time.
    # hidden_channels=8 is the number of hidden channels for the evolving z_t, which we get to choose.
    # output_channels=1 because we're doing binary classification.
    ######################
    model = NeuralCDE(input_channels=3, hidden_channels=16, output_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    ######################
    # Now we turn our dataset into a continuous path. We do this here via natural cubic spline interpolation.
    # The resulting `train_coeffs` are some tensors describing the path.
    # For most problems, it's advisable to save these coeffs and treat them as the dataset, as this interpolation can take
    # a long time.
    ######################
    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(train_t_ss, train_X_ss)

    train_dataset = torch.utils.data.TensorDataset(*train_coeffs, train_X_ss)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
    
    for epoch in range(50):
        for batch in train_dataloader:
            *batch_coeffs, batch_X = batch
            pred_X = model(train_t_ss, batch_coeffs).squeeze(-1)
    #         batch_TND = batch_X[:, :, 1:].view(pred_X.shape[0], pred_X.shape[1], 2)
            batch_TND = batch_X[:, :, 1:].transpose(0,1)

            non_nan_inds = torch.logical_not(torch.isnan(batch_TND))
            loss = torch.nn.functional.mse_loss(pred_X[non_nan_inds], batch_TND[non_nan_inds])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))


    test_t_ss = t_ss
    # test_t_ss, test_X_ss, test_X_clean_ss, test_mask_ss = get_data_square_sine()
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(test_t_ss, test_X_ss)
    test_pred_X_TND = model(test_t_ss, test_coeffs).squeeze(-1).detach().numpy()
    
    
    n_plot_seqs = 3
    f, axs = plt.subplots(2, n_plot_seqs, figsize=(22, 10))
    sns.set_style("whitegrid") # or use "white" if we don't want grid lines
    sns.set_context("notebook", font_scale=1.3)
    
    save_file = 'results/true_vs_predicted_neural_cde.png'
    print('results saved to : %s'%save_file)
    for sample_ind in range(n_plot_seqs):
        axs[0, sample_ind].plot(test_t_ss, 
                    test_X_clean_ss[sample_ind, :, 1], 'k--')

        axs[0, sample_ind].plot(test_t_ss, 
                    test_pred_X_TND[:, sample_ind, 0], 'r--')

        plot_0_inds = test_mask_ss[sample_ind, :, 0]==1

        axs[0, sample_ind].plot(test_t_ss[plot_0_inds], 
                    test_X_ss[sample_ind, :, 1][plot_0_inds], 'k.',
                    label='true')

        axs[0, sample_ind].plot(test_t_ss[plot_0_inds], 
                    test_pred_X_TND[:, sample_ind, 0][plot_0_inds], 'r.',
                    label='predicted')

        axs[0, sample_ind].set_xticks(np.arange(0, 2, .25))
        axs[0, sample_ind].set_xlim([0, 2])

        axs[1, sample_ind].plot(test_t_ss, 
                    test_X_clean_ss[sample_ind, :, 2], 'k--')
        axs[1, sample_ind].plot(test_t_ss, 
                    test_pred_X_TND[:, sample_ind, 1], 'r--')

        plot_1_inds = test_mask_ss[sample_ind, :, 1]==1

        axs[1, sample_ind].plot(test_t_ss[plot_1_inds], 
                    test_X_ss[sample_ind, :, 2][plot_1_inds], 
                    'k.',
                    label='true')
        axs[1, sample_ind].plot(test_t_ss[plot_1_inds], 
                    test_pred_X_TND[:, sample_ind, 1][plot_1_inds], 
                    'r.',
                    label='predicted')

        axs[1, sample_ind].set_xticks(np.arange(0, 2, .25))
        axs[1, sample_ind].set_xlim([0, 2])


        axs[0, sample_ind].legend()
        axs[1, sample_ind].legend()
    f.savefig(save_file)

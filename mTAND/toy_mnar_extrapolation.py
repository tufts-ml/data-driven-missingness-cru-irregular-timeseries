# pylint: disable=E1101, E0401, E1102, W0621, W0221
import argparse
import numpy as np
import torch
import torch.optim as optim

from random import SystemRandom
import models
import utils
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error


parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--gen-hidden', type=int, default=50)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--k-iwae', type=int, default=10)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_rnn')
parser.add_argument('--dec', type=str, default='mtan_rnn')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--quantization', type=float, default=0.016,
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true',
                    help="Include binary classification loss")
parser.add_argument('--norm', action='store_true')
parser.add_argument('--kl', action='store_true')
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=1)
parser.add_argument('--dec-num-heads', type=int, default=1)
parser.add_argument('--length', type=int, default=20)
parser.add_argument('--num-ref-points', type=int, default=128)
parser.add_argument('--dataset', type=str, default='toy_mnar_square_sine')
parser.add_argument('--enc-rnn', action='store_false')
parser.add_argument('--dec-rnn', action='store_false')
parser.add_argument('--sample-tp', type=float, default=1.0)
parser.add_argument('--only-periodic', type=str, default=None)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--N', type=int, default=500)
args = parser.parse_args()


if __name__ == '__main__':
    experiment_id = int(SystemRandom().random() * 100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    data_obj = utils.get_toy_data_mnar_square_sine(args, N=args.N)
    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    dim = data_obj["input_dim"]
        
    # model
    if args.enc == 'enc_rnn3':
        rec = models.enc_rnn3(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, 
            args.rec_hidden, 8, learn_emb=True, device=device)
    elif args.enc == 'mtan_rnn':
        rec = models.enc_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden, 
            embed_time=8, learn_emb=True, num_heads=args.enc_num_heads, device=device)
   
        
    if args.dec == 'rnn3':
        dec = models.dec_rnn3(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, 
            args.gen_hidden, 8, learn_emb=True, device=device)
    elif args.dec == 'mtan_rnn':
        dec = models.dec_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.gen_hidden, 
            embed_time=8, learn_emb=True, num_heads=args.dec_num_heads, device=device)
    
    
    params = (list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec))
    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 1))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 3))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 10))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 20))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 30))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 50))
    
    
    perf_dict_list = []
    for itr in range(1, args.niters + 1):
        train_loss = 0
        train_n = 0
        avg_reconst, avg_kl, mse = 0, 0, 0
        if args.kl:
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))
        else:
            kl_coef = 1

        for train_batch in train_loader:
            train_batch = train_batch.to(device)
            batch_len = train_batch.shape[0]
            observed_data = train_batch[:, :, :dim]
            observed_mask = train_batch[:, :, dim:2 * dim]
            
            observed_tp = train_batch[:, :, -1]
            
            # train only upto time point 1.5
            eval_tp_mask = observed_tp>1.5
            observed_mask = observed_mask*np.logical_not(eval_tp_mask[:, :, np.newaxis])
            
            eval_mask = torch.zeros_like(observed_mask) + eval_tp_mask[:, :, np.newaxis]
            
            if args.sample_tp and args.sample_tp < 1:
                subsampled_data, subsampled_tp, subsampled_mask = utils.subsample_timepoints(
                    observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            else:
                subsampled_data, subsampled_tp, subsampled_mask = \
                    observed_data, observed_tp, observed_mask
            out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
            qz0_mean = out[:, :, :args.latent_dim]
            qz0_logvar = out[:, :, args.latent_dim:]
            # epsilon = torch.randn(qz0_mean.size()).to(device)
            epsilon = torch.randn(
                args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            ).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            pred_x = dec(
                z0,
                observed_tp[None, :, :].repeat(args.k_iwae, 1, 1).view(-1, observed_tp.shape[1])
            )
            # nsample, batch, seqlen, dim
            pred_x = pred_x.view(args.k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2])
            # compute loss
            logpx, analytic_kl = utils.compute_losses(
                dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
            loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(args.k_iwae))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_len
            train_n += batch_len
            avg_reconst += torch.mean(logpx) * batch_len
            avg_kl += torch.mean(analytic_kl) * batch_len
            mse += utils.mean_squared_error(
                observed_data, pred_x.mean(0), eval_mask) * batch_len
        
        test_mse = utils.evaluate(dim, rec, dec, test_loader, args, 1, eval_mask=eval_mask)
        print('Iter: {}, avg loss: {:.4f}, avg reconst: {:.4f}, avg kl: {:.4f}, mse: {:.6f}'
            .format(itr, train_loss / train_n, -avg_reconst / train_n, avg_kl / train_n, mse / train_n), flush=True)
        curr_perf_dict = {'Iter':itr,
                         'loss':train_loss / train_n,
                         'reconstruction_loss' : -avg_reconst / train_n,
                         'kl_divergence' : avg_kl / train_n,
                         'mse' : mse / train_n,
                         'test_mse' : test_mse} 
        perf_dict_list.append(curr_perf_dict)
        
        output_fname_prefix = args.dataset + '_' + str(args.seed) + '_N=' + str(args.N) + '_' + args.enc + '_' + args.dec + '_' + str(experiment_id)
        
        if itr % 10 == 0:
            print('Test Mean Squared Error', test_mse, flush=True)
        if itr % 10 == 0 and args.save:
            torch.save({
                'args': args,
                'epoch': itr,
                'rec_state_dict': rec.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': -loss,
            }, 'results/toy_extrapolation/' + output_fname_prefix + '.h5')
        
        
            perf_df = pd.DataFrame(perf_dict_list)
            perf_df.to_csv('results/toy_extrapolation/%s.csv'%output_fname_prefix, index=False)
            # plot generated sequences            
            t = np.linspace(0, 2, 50)
            d = pred_x.shape[-1]
            
            if d==1:
                f, axs = plt.subplots(1, 3, figsize=(16, 5))
                for ii in range(3):
                    predicted_seq = pred_x[0][ii].detach().numpy()
                    axs[ii].plot(t, predicted_seq[:, 0], 'b')
#                     axs[ii].scatter(observed_tp[ii][torch.squeeze(observed_mask[ii]==1)], 
#                                     observed_data[ii][observed_mask[ii]==1], marker='x', color='k')
            else:
                f, axs = plt.subplots(d, 3, figsize=(22, 5))
                sns.set_style("whitegrid") # or use "white" if we don't want grid lines
                sns.set_context("notebook", font_scale=1.3)
                for ii in range(3):
                    predicted_seq = pred_x[0][ii][:len(t), :].detach().numpy()
                    for dd in range(d):
                        axs[dd, ii].plot(t, predicted_seq[:, dd], 'b')
#                         axs[dd, ii].set_xticks(np.arange(0, np.pi, np.pi/10))
                        axs[dd, ii].set_xlim([0, 2])
                        
            f.savefig('results/toy_extrapolation/generated_seqs_%s.png'%output_fname_prefix)
            plt.close('all')
            
            # plot the true vs predicted for extrapolation
            n_plot_seqs = 3
            f, axs = plt.subplots(2, n_plot_seqs, figsize=(22, 10), 
                          sharex=True
                         )
            sns.set_style("whitegrid") # or use "white" if we don't want grid lines
            sns.set_context("notebook", font_scale=1.3)
            ticks = [-1, -.6, -.3, 0., .3, .6, 1]
            pred_seq_NTD = pred_x[0].detach().numpy()
            for ii in range(n_plot_seqs):
                for d in range(2):
                    plot_inds_T = observed_mask[ii, d*len(t):(d+1)*len(t), d]==1
                    plot_t = t[plot_inds_T]
                    axs[d, ii].scatter(plot_t, observed_data[ii, d*len(t):(d+1)*len(t), 
                                                             d][plot_inds_T], color='k', label='true')
                    axs[d, ii].plot(t, observed_data[ii, d*len(t):(d+1)*len(t), d], 'k--')
                    axs[d, ii].scatter(plot_t, pred_seq_NTD[ii, d*len(t):(d+1)*len(t), 
                                                            d][plot_inds_T], color='r', label='predicted')
                    axs[d, ii].plot(t, pred_seq_NTD[ii, d*len(t):(d+1)*len(t), d], 'r--')
                    axs[d, ii].set_xticks(np.arange(0, 2, .25))
                    axs[d, ii].set_xlim([0, 2])
                    axs[d, ii].legend()


            f.suptitle('True vs predicted (Extrap MSE : %.2f)'%(curr_perf_dict['mse'].item()))
            f.savefig('results/toy_extrapolation/true_vs_predicted_%s.png'%output_fname_prefix)
            
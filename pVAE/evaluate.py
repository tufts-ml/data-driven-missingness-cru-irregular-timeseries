import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict

class Evaluator:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 log_dir, eval_args={}):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.log_dir = log_dir
        self.eval_args = eval_args

#         self.train_auc_log, self.test_auc_log, self.val_auc_log = [], [], []
        self.best_auc, self.best_val_auc = [-float('inf')] * 2

    def evaluate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            train_auc, train_auprc, train_y, train_y_pred, train_loss_breakdown = self.compute_auc_auprc(self.train_loader)
            print('==============Epoch %d==========='%epoch)
            print('train AUC:', train_auc)
            print('train AUPRC:', train_auprc)
#             self.train_auc_log.append((epoch, train_auc))
            
            val_auc, val_auprc, val_y, val_y_pred, val_loss_breakdown = self.compute_auc_auprc(self.val_loader)
            print('val AUC:', val_auc)
            print('val AUPRC:', val_auprc)
#             self.val_auc_log.append((epoch, val_auc))

            test_auc, test_auprc, test_y, test_y_pred, test_loss_breakdown = self.compute_auc_auprc(self.test_loader)
            print('test AUC:', test_auc)
            print('test AUPRC:', test_auprc)
#             self.test_auc_log.append((epoch, test_auc))
        self.model.train()

        return train_auc, val_auc, test_auc, train_auprc, val_auprc, test_auprc, train_y, val_y, test_y, train_y_pred, val_y_pred, test_y_pred, train_loss_breakdown, val_loss_breakdown, test_loss_breakdown

    def compute_auc_auprc(self, data_loader):
        y_true, y_score = [], []
        loss_breakdown = defaultdict(float)
        for (val, idx, mask, y, _, cconv_graph) in data_loader:
            score, loss_info = self.model.predict(
                val, idx, mask, cconv_graph, **self.eval_args, return_loss=True)
            y_score.append(score.cpu().numpy())
            y_true.append(y.cpu().numpy())
        
            for loss_name, loss_val in loss_info.items():
                loss_breakdown[loss_name] += loss_val
        
        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score), y_true, y_score, loss_breakdown

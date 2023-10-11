from data.data_factory import data_provider
from exp.exp_basic import *
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, _ = self._get_data(flag='train')
        test_data, _ = self._get_data(flag='test')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        
        #TODO: make these more generalized
        self.args.enc_in = train_data.feature_df.shape[-1]    
        self.args.num_class = len(train_data.class_names)
        self.multiclass = (self.args.num_class > 1)
        
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.multiclass: criterion = nn.CrossEntropyLoss()
        else: criterion = nn.BCEWithLogitsLoss()
            
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.long() if self.multiclass else label.float()
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.squeeze()
                loss = criterion(pred, label.squeeze())
                total_loss.append(loss)

        total_loss = torch.vstack(total_loss).mean().item()

        self.model.train()
        return total_loss

    def train(self):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = ReduceLROnPlateau(
            model_optim, 'min', 
            patience=3, min_lr=1e-6
        )
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                
                label = label.long() if self.multiclass else label.float()
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs.squeeze(), label.squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            # test_loss, test_accuracy = self.vali(test_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | \
                    Train Loss: {train_loss:.5g} Vali Loss: {vali_loss:.5g}")
            early_stopping(vali_loss, self.model, self.output_folder)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            scheduler.step(vali_loss)
            # if (epoch + 1) % 5 == 0:
            #     adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.load_best_model()
        return self.model

    def test(self, load_model=True, flag='test'):
        _, test_loader = self._get_data(flag=flag)
        if load_model:
            self.load_best_model()

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = np.concatenate(trues, axis=0)
        
        if self.multiclass:
            probs = torch.nn.functional.softmax(preds, dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
            predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
            
        else:
            probs = torch.nn.functional.sigmoid(preds)
            predictions = torch.round(probs).squeeze().cpu().numpy()

        probs = probs.detach().cpu().numpy()
        # trues = trues.flatten().cpu().numpy()
        accuracy = accuracy_score(predictions, trues)

        # save results
        print(f'accuracy:{accuracy:0.5f}')
        with open("result_classification.txt", 'a') as f:
            f.write(stringify_setting(self.args, complete=True)  + "  \n")
            f.write(f'flag {flag}, accuracy:{accuracy:0.5f}\n\n')
        
        np.save(os.path.join(self.output_folder, f'{flag}_pred.npy'), predictions)
        np.save(os.path.join(self.output_folder, f'{flag}_true.npy'), trues)
        return

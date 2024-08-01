import numpy as np
import torch
import matplotlib.pyplot as plt

from captum._utils.typing import (
    TensorOrTupleOfTensorsGeneric,
)

plt.switch_backend('agg')

def avg_over_output_horizon(attr, inputs, args):
    if type(inputs) == tuple:
        # tuple of batch x seq_len x features
        attr = tuple([
            attr_.reshape(
                # batch x pred_len x seq_len x features
                (inputs[0].shape[0], -1, args.seq_len, attr_.shape[-1])
            # take mean over the output horizon
            ).mean(axis=1) for attr_ in attr
        ])
    else:
        # batch x seq_len x features
        attr = attr.reshape(
            # batch x pred_len x seq_len x features
            (inputs.shape[0], -1, args.seq_len, attr.shape[-1])
        # take mean over the output horizon
        ).mean(axis=1)
    
    return attr

def round_up(attr, decimals=6):
    if type(attr) == tuple:
        # tuple of batch x seq_len x features
        attr = tuple([
            torch.round(a, decimals=decimals) for a in attr
        ])
    else:
        # batch x seq_len x features
        attr = torch.round(attr, decimals=decimals)
    
    return attr

def reshape_over_output_horizon(attr, inputs, args):
    if type(inputs) == tuple:
        # tuple of batch x seq_len x features
        attr = tuple([
            attr_.reshape(
                # batch x pred_len x seq_len x features
                (inputs[0].shape[0], -1, args.seq_len, attr_.shape[-1])
            # take mean over the output horizon
            ) for attr_ in attr
        ])
        # print([a.shape for a in attr])
    else:
        # batch x seq_len x features
        attr = attr.reshape(
            # batch x pred_len x seq_len x features
            (inputs.shape[0], -1, args.seq_len, attr.shape[-1])
        # take mean over the output horizon
        )
        # print(attr.shape)
    return attr

#TODO: debug error for some cases
def reshape_attr(
    attr: TensorOrTupleOfTensorsGeneric, 
    inputs: TensorOrTupleOfTensorsGeneric
):
    sample_size = inputs[0].shape[0] if type(inputs) == tuple  else inputs.shape[0]
    if type(attr) == tuple:
        # tuple of batch x seq_len x features
        attr = tuple([
            attr_.reshape(
                # batch x pred_len x seq_len x features
                ((sample_size, -1) + attr_.shape[1:])
            # take mean over the output horizon
            ) for attr_ in attr
        ])
    else:
        # batch x seq_len x features
        attr = attr.reshape(
            # batch x pred_len x seq_len x features
            ((sample_size, -1) + (attr.shape[1:]))
        # take mean over the output horizon
        )
    
    return attr

def normalize_scale(
    data: torch.Tensor, dim=1,
    norm_type='standard'
):
    if norm_type == "standard":
        mean = data.mean(dim=dim, keepdim=True)
        std = data.std(dim=dim, keepdim=True)
        return (data - mean) / (std + torch.finfo(torch.float32).eps)

    elif norm_type == "minmax":
        max_val = torch.amax(data, dim=dim, keepdim=True)[0]
        min_val = torch.amin(data, dim=dim, keepdim=True)[0]
        return (data - min_val) / (max_val - min_val + torch.finfo(torch.float32).eps)
        
    elif norm_type == "l1":
        sum_val = data.abs().sum(dim=dim, keepdim=True)
        
        # this converts neg to absolute values
        return data.abs() / (sum_val + torch.finfo(torch.float32).eps)
    else:
        raise (NameError(f'Normalize method "{norm_type}" not implemented'))

# min max scale a torch tensor across a dimension
def min_max_scale(a:torch.Tensor, dim=1, absolute=True):
    if absolute: a = a.abs()
    min_values = a.min(dim=dim, keepdims=True).values
    max_values = a.max(dim=dim, keepdims=True).values

    scaled = (a - min_values)/(max_values - min_values)
    scaled[scaled!=scaled] = 0
    return scaled 

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
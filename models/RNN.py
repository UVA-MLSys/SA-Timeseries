from torch import nn
import torch
from utils.timefeatures import time_features_from_frequency_str

class _RNN_Base(nn.Module):
    def __init__(
        self, configs, bias=True, 
        bidirectional=True, init_weights=True
    ):
        super(_RNN_Base, self).__init__()
        self.configs = configs
        
        # model configurations
        rnn_dropout = fc_dropout = configs.dropout
        hidden_size = configs.d_model
        n_layers = configs.e_layers
        
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else nn.Identity()
        self.bn = nn.BatchNorm1d(num_features=hidden_size* (1 + bidirectional))
        
        if configs.task_name == 'classification':
            input_size = configs.enc_in
            self.projection = nn.Linear(hidden_size * (1 + bidirectional), configs.num_class)
        else:
            input_size = configs.enc_in + len(time_features_from_frequency_str(configs.freq))
            self.projection = nn.Linear(hidden_size * (1 + bidirectional), configs.pred_len*configs.c_out)
        
        self.rnn = self._cell(
            input_size, hidden_size, num_layers=n_layers, 
            bias=bias, batch_first=True, dropout=rnn_dropout, 
            bidirectional=bidirectional
        )
        
        if init_weights: self.apply(self._weights_init)
        
    def classification(self, x_enc):
        # batch_size x seq_len x n_vars
        output, _ = self.rnn(x_enc) # output from all sequence steps: [batch_size x seq_len x hidden_size * (1 + bidirectional)]
        output = output[:, -1]  # output from last sequence step : [batch_size x hidden_size * (1 + bidirectional)]
        
        output = self.projection(self.dropout(self.bn(output)))
        return output
    
    def forecast(self, x_enc, x_mark_enc):
        x_enc = torch.cat([x_enc, x_mark_enc], dim=2)
        # batch_size x seq_len x n_vars
        output, _ = self.rnn(x_enc) # output from all sequence steps: [batch_size x seq_len x hidden_size * (1 + bidirectional)]
        output = output[:, -1]  # output from last sequence step : [batch_size x hidden_size * (1 + bidirectional)]
        
        output = self.projection(self.dropout(self.bn(output)))
        output = output.reshape((-1, self.configs.pred_len, self.configs.c_out))
        f_dim = -1 if self.configs.features == 'MS' else 0
        return output[:, :, f_dim:]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None): 
        if self.configs.task_name == 'classification':
            return self.classification(x_enc)
        else:
            return self.forecast(x_enc, x_mark_enc)
    
    def _weights_init(self, m): 
        # same initialization as keras. Adapted from the initialization developed 
        # by JUN KODA (https://www.kaggle.com/junkoda) in this notebook
        # https://www.kaggle.com/junkoda/pytorch-lstm-with-tensorflow-like-initialization
        for name, params in m.named_parameters():
            if "weight_ih" in name: 
                nn.init.xavier_normal_(params)
            elif 'weight_hh' in name: 
                nn.init.orthogonal_(params)
            elif 'bias_ih' in name:
                params.data.fill_(0)
                # Set forget-gate bias to 1
                n = params.size(0)
                params.data[(n // 4):(n // 2)].fill_(1)
            elif 'bias_hh' in name:
                params.data.fill_(0)
        
# class RNN(_RNN_Base):
#     _cell = nn.RNN
    
# class LSTM(_RNN_Base):
#     _cell = nn.LSTM
    
# class GRU(_RNN_Base):
#     _cell = nn.GRU
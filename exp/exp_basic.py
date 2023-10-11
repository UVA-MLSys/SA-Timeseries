import os
import torch
from data.data_factory import data_provider
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, LSTM, TCN
    

def stringify_setting(args, complete=False):
    if not complete:
        setting = f"{args.data_path.split('.')[0]}_{args.model}"
        if args.des:
            setting += '_' + args.des
        return setting
    
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}'.format(
        args.task_name,
        args.model,
        args.data_path.split('.')[0],
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil
    )
    
    return setting

class Exp_Basic(object):
    model_dict = {
        'TimesNet': TimesNet,
        'Autoformer': Autoformer,
        'Transformer': Transformer,
        'Nonstationary_Transformer': Nonstationary_Transformer,
        'DLinear': DLinear,
        'FEDformer': FEDformer,
        'Informer': Informer,
        'LightTS': LightTS,
        'Reformer': Reformer,
        'ETSformer': ETSformer,
        'PatchTST': PatchTST,
        'Pyraformer': Pyraformer,
        'MICN': MICN,
        'Crossformer': Crossformer,
        'FiLM': FiLM,
        'LSTM': LSTM,
        'TCN': TCN
    }
    
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
        self.setting = stringify_setting(args)
        self.output_folder = os.path.join(args.result_path, self.setting)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)
        print(f'Experiments will be saved in {self.output_folder}')
        
        self.dataset_map = {}

    def _build_model(self):
        raise NotImplementedError
    
    def load_best_model(self):
        best_model_path = os.path.join(self.output_folder, 'checkpoint.pth')
        print(f'Loading model from {best_model_path}')
        self.model.load_state_dict(torch.load(best_model_path))

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag='test'):
        if flag not in self.dataset_map:
            self.dataset_map[flag] = data_provider(self.args, flag)
            
        return self.dataset_map[flag] 

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

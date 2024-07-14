import torch, gc
import numpy as np
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from typing import Callable, Optional, Tuple, Union

# Source:https://github.com/layer6ai-labs/WinIT
class WinIT:
    def __init__(self, model, data, args):
        self.model = model
        self.args = args
        self.seq_len = args.seq_len
        self.task_name = args.task_name
        self.data = data
        self.rng = np.random.default_rng(args.seed)
        
        if self.task_name =='classification':
            self.metric = 'kl'
        else:
            self.metric = 'pd'
        
    def _compute_metric(
        self, p_y_exp: torch.Tensor, p_y_hat: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the metric for comparisons of two distributions.

        Args:
            p_y_exp:
                The current expected distribution. Shape = (batch_size, num_states)
            p_y_hat:
                The modified (counterfactual) distribution. Shape = (batch_size, num_states)

        Returns:
            The result Tensor of shape (batch_size).

        """
        if self.metric == "kl":
            kl_loss = torch.nn.KLDivLoss(reduction="none")(torch.log(p_y_hat), p_y_exp)
            # return torch.sum(kl_loss, -1)
            return kl_loss
        if self.metric == "js":
            average = (p_y_hat + p_y_exp) / 2
            lhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_hat)
            rhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_exp)
            return torch.sum((lhs + rhs) / 2, -1)
        if self.metric == "pd":
            # batch_size x output_horizon x num_states
            diff = torch.abs(p_y_hat - p_y_exp)
            
            # sum over all dimension except batch, output_horizon
            # multioutput multi-horizon not supported yet
            return torch.sum(diff, dim=-1)
        
        raise Exception(f"unknown metric. {self.metric}")
    
    def generate_counterfactuals(self, batch_size, input_index, feature_index):
        if input_index is None:
            choices = self.data[:, :, feature_index].reshape(-1)
        else:
            choices = self.data[input_index][:][:, :, feature_index].reshape(-1)

        sampled_index = np.random.choice(range(len(choices)), size=(batch_size*self.seq_len))
        samples = choices[sampled_index].reshape((batch_size, self.seq_len))
        
        return samples
    
    def format_output(self, outputs):
        if self.task_name == 'classification':
            return torch.nn.functional.softmax(outputs, dim=1)
        else:
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            return outputs
            
    def attribute(
        self, inputs:Union[torch.Tensor , Tuple[torch.Tensor]], 
        additional_forward_args: Tuple[torch.Tensor], 
        attributions_fn=None
    ):
        model = self.model
        if type(inputs) != tuple:
            inputs = tuple([inputs])
            
        attr = []
        with torch.no_grad():
            y_original = self.format_output(
                model(*inputs, *additional_forward_args)
            ) 
            
            for input_index in range(len(inputs)):
                batch_size, seq_len, n_features = inputs[input_index].shape
                
                # batch_size, n_output, seq_len, n_features
                iS_array = torch.zeros(size=(
                    batch_size, y_original.shape[1], seq_len, n_features
                ), device=inputs[input_index].device)
            
                for feature in range(n_features):
                    #TODO: do without cloning
                    cloned = inputs[input_index].clone()
                    
                    #TODO: use baselines
                    if len(inputs) == 1:
                        counterfactuals = self.generate_counterfactuals(
                            batch_size, None, feature
                        )
                    else:
                        counterfactuals = self.generate_counterfactuals(
                            batch_size, input_index, feature
                        )
                    
                    for t in range(seq_len)[::-1]:
                        # mask last t timesteps
                        cloned[:, t, feature] = counterfactuals[:, t]
                        
                        inputs_hat = []
                        for i in range(len(inputs)):
                            if i == input_index: inputs_hat.append(cloned)
                            else: inputs_hat.append(inputs[i])
                        
                        y_perturbed = self.format_output(
                            model(*tuple(inputs_hat), *additional_forward_args)
                        )
                        
                        iSab = self._compute_metric(y_original, y_perturbed)
                        iSab = torch.clip(iSab, -1e6, 1e6)
                        iS_array[:, :, t, feature] = iSab
                        
                        del y_perturbed, inputs_hat
                    del cloned, counterfactuals
                    gc.collect()
            
                # batch_size, n_output, seq_len, n_features        
                iS_array[:, :, 1:] -= iS_array[:, :, :-1] 
                
                # reverse order along the time axis
                iS_array = iS_array.flip(dims=(2,))
                
                if attributions_fn is not None:
                    attr.append(attributions_fn(iS_array))
                else: attr.append(iS_array)
           
        if len(attr) == 1: return attr[0]
        else: return tuple(attr)
    
    def get_name(self):
        return 'WinIT'
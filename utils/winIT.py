import torch
import numpy as np
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from typing import Callable, Optional


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
            return torch.sum(torch.nn.KLDivLoss(reduction="none")(torch.log(p_y_hat), p_y_exp), -1)
        if self.metric == "js":
            average = (p_y_hat + p_y_exp) / 2
            lhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_hat)
            rhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_exp)
            return torch.sum((lhs + rhs) / 2, -1)
        if self.metric == "pd":
            diff = torch.abs(p_y_hat - p_y_exp)
            
            # sum over all dimension except batch
            summed = torch.sum(diff, dim=-1) # tuple(range(diff.ndim)[1:])
            return summed
        
        raise Exception(f"unknown metric. {self.metric}")
    
    def generate_counterfactuals(self, batch_size, input_index, feature_index):
        
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
        self, inputs, additional_forward_args, 
        attributions_fn=None
    ):
        model = self.model
        y_original = self.format_output(model(*inputs, *additional_forward_args))
        attr = []
        
        for input_index, input in enumerate(inputs):
            batch_size, seq_len, n_features = input.shape
            
            if self.task_name == 'classification':
                iS_array = torch.zeros(size=(
                    batch_size, self.args.num_class, seq_len, n_features
                ))
            else:
                iS_array = torch.zeros(size=(
                    batch_size, self.args.pred_len, seq_len, n_features
                ))
            
            for feature in range(n_features):
                cloned = input.clone()
                counterfactuals = self.generate_counterfactuals(
                    batch_size, input_index, feature
                )
                
                for t in range(seq_len)[::-1]:
                    # mask last t timesteps
                    cloned[:, t:, feature] = counterfactuals[:, t:]
                    
                    inputs_hat = []
                    for i in range(len(inputs)):
                        if i == input_index: inputs_hat.append(cloned)
                        else: inputs_hat.append(inputs[i])
                
                    y_perturbed = self.format_output(model(*tuple(inputs_hat), *additional_forward_args))
                    
                    iSab = self._compute_metric(y_original, y_perturbed)
                    iSab = torch.clip(iSab, -1e6, 1e6)
                    iS_array[:, :, t, feature] = iSab
                        
            iS_array[:, :, 1:] -= iS_array[:, :, :-1]
            
            if attributions_fn is not None:
                attr.append(attributions_fn(iS_array))
            else: attr.append(iS_array)
            
        return tuple(attr)
    
    def get_name(self):
        return 'WinIT'
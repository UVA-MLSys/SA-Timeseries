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
            kl_loss = torch.nn.KLDivLoss(reduction="none")(torch.log(p_y_hat), p_y_exp)
            # return torch.sum(kl_loss, -1)
            return kl_loss
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
        
    def _attribute(
        self, inputs, additional_forward_args, 
        attributions_fn=None
    ):
        model = self.model
        
        with torch.no_grad():
            if type(additional_forward_args) == tuple:
                y_original = self.format_output(
                    model(inputs, *additional_forward_args)
                )
            else:
                y_original = self.format_output(
                    model(inputs, additional_forward_args)
                )
        
        batch_size, seq_len, n_features = inputs.shape
        
        n_output = self.args.num_class if self.task_name == 'classification' else self.args.pred_len
        iS_array = torch.zeros(size=(
            batch_size, n_output, seq_len, n_features
        ), device=inputs.device)
        
        with torch.no_grad():
            for feature in range(n_features):
                inputs_hat = inputs.clone()
                counterfactuals = self.generate_counterfactuals(
                    batch_size, None, feature
                )
                
                for t in range(seq_len)[::-1]:
                    # mask last t timesteps
                    inputs_hat[:, t, feature] = counterfactuals[:, t]
                
                    if type(additional_forward_args) == tuple:
                        y_perturbed = self.format_output(
                            model(inputs_hat, *additional_forward_args)
                        )
                    else:
                        y_perturbed = self.format_output(
                            model(inputs_hat, additional_forward_args)
                        )
                    
                    iSab = self._compute_metric(y_original, y_perturbed)
                    iSab = torch.clip(iSab, -1e6, 1e6)
                    iS_array[:, :, t, feature] = iSab
                    
                    del y_perturbed, iSab
                del counterfactuals
        
        # i(S, a, b) i(S)^b_a − i(S)^b_a+1
        iS_array[:, :, 1:] -= iS_array[:, :, :-1] 
        
        # reverse order along the time axis
        # equivalent to iS_array[:, :, ::-1] which doesn't work 
        # sometimes since torch doesn't support it
        iS_array = iS_array.flip(dims=(2,))
        
        if attributions_fn is not None:
            return attributions_fn(iS_array)
        else: return iS_array
    
    def attribute(
        self, inputs, additional_forward_args, 
        attributions_fn=None
    ):
        if type(inputs) == tuple:
            return self._attribute_tuple(
                inputs, additional_forward_args, attributions_fn
            )
        else:
            return self._attribute(
                inputs, additional_forward_args, attributions_fn
            )
            
    def _attribute_tuple(
        self, inputs, additional_forward_args, 
        attributions_fn=None
    ):
        model = self.model
        with torch.no_grad():
            if type(additional_forward_args) == tuple:
                y_original = model(*inputs, *additional_forward_args)
            else:
                y_original = model(*inputs, additional_forward_args)
            
        y_original = self.format_output(y_original)
            
        attr = []
        for input_index in range(len(inputs)):
            batch_size, seq_len, n_features = inputs[input_index].shape
            
            # batch_size, n_output, seq_len, n_features
            iS_array = torch.zeros(size=(
                batch_size, y_original.shape[1], seq_len, n_features
            ), device=inputs[input_index].device)
            
            with torch.no_grad():
                for feature in range(n_features):
                    cloned = inputs[input_index].clone()
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
                        
                        if type(additional_forward_args) == tuple:
                            y_perturbed = self.format_output(
                                model(*tuple(inputs_hat), *additional_forward_args)
                            )
                        else:
                            y_perturbed = self.format_output(
                                model(*tuple(inputs_hat), additional_forward_args)
                            )
                        
                        iSab = self._compute_metric(y_original, y_perturbed)
                        iSab = torch.clip(iSab, -1e6, 1e6)
                        iS_array[:, :, t, feature] = iSab
                        
                        del y_perturbed, inputs_hat
                    del cloned, counterfactuals
            
            # batch_size, n_output, seq_len, n_features        
            iS_array[:, :, 1:] -= iS_array[:, :, :-1] 
            
            # reverse order along the time axis
            iS_array = iS_array.flip(dims=(2,))
            
            if attributions_fn is not None:
                attr.append(attributions_fn(iS_array))
            else: attr.append(iS_array)
            
        return tuple(attr)
    
    def get_name(self):
        return 'WinIT'
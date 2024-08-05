import torch, gc, copy

from typing import Any, Tuple, Union
from utils.tools import normalize_scale
from captum.attr import IntegratedGradients
from exp.exp_basic import dual_input_users

class TSR:
    def __init__(self, model, args):
        self.args = args
        
        if 'RNN' in args.model:
            clone = copy.deepcopy(model)
            clone.train() # deep lift moedl needs to be in training mode
            self.explainer = IntegratedGradients(clone)
        else:
            self.explainer = IntegratedGradients(model)
        
    def get_time_relevance_score(
        self, inputs, additional_forward_args, baselines
    ):
        input_is_tuple = type(inputs) == tuple
        if not input_is_tuple:
            inputs = tuple([inputs])
            baselines = tuple([baselines])
            
        attr_original = self.compute_grads(inputs, additional_forward_args, baselines)
            
        with torch.no_grad():
            time_relevance_score = []
            for input_index in range(len(inputs)):
                cloned = inputs[input_index].clone()
                batch_size, n_output, seq_len, n_features = attr_original[input_index].shape
                score = torch.zeros(
                    (batch_size*n_output, seq_len), device=inputs[input_index].device
                )
                assignment = inputs[input_index][0, 0, 0]
                
                for t in range(inputs[input_index].shape[1]):
                    prev_value = cloned[:, t]
                    cloned[:, t] = assignment
                    
                    inputs_hat = []
                    for i in range(len(inputs)):
                        if i == input_index:
                            inputs_hat.append(cloned)
                        else:
                            inputs_hat.append(inputs[i])
                
                    attr_perturbed = self.compute_grads(tuple(inputs_hat), additional_forward_args, baselines)
                    
                    attr_diff = abs(attr_perturbed[0] - attr_original[0])
                    score[:, t] = torch.sum(attr_diff, dim=(2, 3)).flatten()
                    cloned[:, t] = prev_value
                    
                    del attr_perturbed, attr_diff
                    gc.collect()
                
                time_relevance_score.append(score)
        
        # time_relevance_score shape will be ((N x O) x seq_len) after summation
        time_relevance_score = tuple(
            # tsr.sum((tuple(i for i in range(2, len(tsr.shape)))))
            tsr.reshape((-1, tsr.shape[-1]))
            for tsr in time_relevance_score
        )
        time_relevance_score = tuple(
            normalize_scale(
                tsr, dim=1, norm_type="minmax"
            ) for tsr in time_relevance_score
        )
        
        if input_is_tuple:
            return time_relevance_score
        else: return time_relevance_score[0]
        
    def compute_grads(self, inputs,additional_forward_args, baselines):
        attr_list = []
        if self.args.task_name == 'classification':
            targets = self.args.num_class
        else: targets = self.args.pred_len
        
        for target in range(targets):
            # temporary speedup
            # if target > 0:
            #     attr_list.append(attr)
            #     continue
            
            # these models use the multiple inputs in the forward function
            if type(inputs) == tuple and self.args.model not in dual_input_users:
                new_additional_forward_args = tuple([
                    input for input in inputs[1:]
                ]) + additional_forward_args
                
                # output is a tuple of length 1, since only one input is used
                attr = self.explainer.attribute(
                    inputs=inputs[0], baselines=baselines[0], target=target,
                    additional_forward_args=new_additional_forward_args
                )
                
                attr = tuple([attr] + [
                    torch.zeros_like(inputs[i], device=inputs[i].device) for i in range(1, len(inputs))]
                )
                
            else: attr = self.explainer.attribute(
                inputs=inputs, baselines=baselines, target=target,
                additional_forward_args=additional_forward_args
            )
            attr_list.append(attr)
            
        if type(inputs) == tuple:
            attr = []
            for input_index in range(len(inputs)):
                attr_per_input = torch.stack([score[input_index] for score in attr_list])
                # pred_len x batch x seq_len x features -> batch x pred_len x seq_len x features
                attr_per_input = attr_per_input.permute(1, 0, 2, 3)
                attr.append(attr_per_input)
                
            attr = tuple(attr)
        else:
            attr = torch.stack(attr_list)
            # pred_len x batch x seq_len x features -> batch x pred_len x seq_len x features
            attr = attr.permute(1, 0, 2, 3)
        return attr
            
    def attribute(
        self, inputs:Union[torch.Tensor , Tuple[torch.Tensor]], 
        additional_forward_args: Tuple[torch.Tensor], 
        baselines, threshold=0.55
    ):
        input_is_tuple = type(inputs) == tuple
        if not input_is_tuple:
            inputs = tuple([inputs])
            baselines = tuple([baselines])
            
        time_relevance_score = self.get_time_relevance_score(
                inputs, additional_forward_args, baselines
            )
        
        # batch_size x n_output x seq_len x n_features
        attr_original = self.compute_grads(inputs, additional_forward_args, baselines)
            
        is_above_threshold = tuple(
            score > torch.quantile(score, threshold, dim=1, keepdim=True) for score in time_relevance_score
        )
        
        with torch.no_grad():
            feature_relevance_score = []
            for input_index in range(len(inputs)):
                batch_size, n_output, seq_len, n_features = attr_original[input_index].shape
                
                assignment = inputs[input_index][0, 0, 0]
                score = torch.zeros(
                    (batch_size*n_output, seq_len, n_features), 
                    device=inputs[input_index].device
                )
                
                for t in range(seq_len):
                    #TODO: revise
                    above_threshold = is_above_threshold[input_index][:, t]

                    for f in range(n_features):
                        if not above_threshold.any():
                            score[:, t] = 0.1
                            continue
                        
                        prev_value = inputs[input_index][:, t, f]
                        inputs[input_index][:, t, f] = assignment
                        
                        attr_perturbed = self.compute_grads(
                            inputs, additional_forward_args, baselines
                        )

                        inputs[input_index][:, t, f] = prev_value
                        # right now only the first element in the tuple is used
                        for original, perturbed in zip(attr_original, attr_perturbed):
                            diff = abs(perturbed - original)
                            score[:, t, f] += torch.sum(diff, dim=(2, 3)).flatten()
                            
                        score[~above_threshold, t] = 0.1
                        del attr_perturbed
                        gc.collect()
                        
                    # normalize across the feature dimension
                    score = normalize_scale(score, dim=0, norm_type="minmax")
                    
                feature_relevance_score.append(score)
            
        time_relevance_score = tuple(
            tsr.reshape(input.shape[:2] + (1,) * len(input.shape[2:]))
            for input, tsr in zip(feature_relevance_score, time_relevance_score)
        )
        
        attributions = tuple(
            (tsr * frs) for tsr, frs in zip(
                time_relevance_score,
                feature_relevance_score
            )
        )
            
        if input_is_tuple: return attributions
        else: return attributions[0]    
    
    def get_name():
        return 'TSR'
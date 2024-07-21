import torch
import numpy as np
from utils.tools import reshape_over_output_horizon

def get_total_data(dataloader, device, add_x_mark=True):        
    if add_x_mark:
        return (
            torch.vstack([item[0] for item in dataloader]).float().to(device), 
            torch.vstack([item[2] for item in dataloader]).float().to(device)
        )
    else:
        return torch.vstack([item[0] for item in dataloader]).float().to(device)

def get_baseline(inputs, mode='random'):
    if type(inputs) == tuple:
        return tuple([get_baseline(input, mode) for input in inputs])
    
    batch_size, seq_len, n_features = inputs.shape[0], inputs.shape[1], inputs.shape[2]    
    device = inputs.device
    
    if mode =='zero': baselines = torch.zeros_like(inputs, device=device).float()
    elif mode == 'random': baselines = torch.randn_like(inputs, device=device).float()
    elif mode == 'aug':
        inputs = inputs.reshape((-1, n_features))
        baselines = torch.zeros_like(inputs, device=device).float()
        
        for f in range(n_features):
            choices = inputs[:, f]
            sampled_index = np.random.choice(
                range(inputs.shape[0]), size=inputs.shape[0], replace=True
            )
            baselines[:, f] = choices[sampled_index]
        baselines = baselines.reshape((batch_size, seq_len, n_features))
    elif mode == 'normal':
        means = torch.mean(inputs, dim=(0, 1))
        std = torch.std(inputs, dim=(0, 1))
        baselines = torch.normal(means, std).repeat(
            batch_size, inputs.shape[1], 1
        ).float()
    elif mode == 'mean': 
        baselines = torch.mean(
                inputs, axis=0
        ).repeat(batch_size, 1, 1).float()
    else:
        print(f'baseline mode options: [zero, random, aug, mean, normal]')
        raise NotImplementedError
    
    return baselines

def compute_attr(
    name, inputs, baselines, explainer,
    additional_forward_args, args
):
    # name = explainer.get_name()
    if args.task_name == 'classification':
        targets = args.num_class
    else: targets = args.pred_len
        
    if type(inputs) == tuple:
        sliding_window_shapes = tuple([
            (1,1) for _ in inputs
        ])
    else: sliding_window_shapes = (1,1)
    
    if name == 'wtsr':
        attr = explainer.attribute(
            inputs=inputs,
            sliding_window_shapes=sliding_window_shapes,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            threshold=0.5, normalize=True,
            attributions_fn=abs
        )
    
    # these methods need a target specified for multi-output models
    elif name in [
        'deep_lift', 'lime', 'integrated_gradients', 
        'gradient_shap', 'tsr'
    ]:
        attr_list = []
        
        for target in range(targets):
            if name == 'tsr':
                attr = explainer.attribute(
                    inputs=inputs,
                    sliding_window_shapes=sliding_window_shapes,
                    baselines=baselines, target=target,
                    additional_forward_args=additional_forward_args,
                    threshold=0.55, normalize=True,
                    attributions_fn=abs
                )
            # gradient based methods can't differentiate when an input isn't used in the model
            elif name in ['deep_lift', 'integrated_gradients', 'gradient_shap']:
                if type(inputs) == tuple:
                    new_additional_forward_args = tuple([
                        input for input in inputs[1:]
                    ]) + additional_forward_args
                    
                    # output is a tuple of length 1, since only one input is used
                    attr = explainer.attribute(
                        inputs=inputs[0], baselines=baselines[0], target=target,
                        additional_forward_args=new_additional_forward_args
                    )
                    
                    zero_attr = torch.zeros_like(inputs[0], device=inputs[0].device)
                    attr = tuple([attr] + [zero_attr for i in range(1, len(inputs))])
                    
                else: attr = explainer.attribute(
                    inputs=inputs, baselines=baselines, target=target,
                    additional_forward_args=additional_forward_args
                )
            else: attr = explainer.attribute(
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
        
    elif name in ['feature_ablation']:
        attr = explainer.attribute(
            inputs=inputs,
            attributions_fn=abs,
            additional_forward_args=additional_forward_args
        )
        
    elif name in ['feature_permutation', 'winIT']:
        attr = explainer.attribute(
            inputs=inputs, attributions_fn=abs,
            additional_forward_args=additional_forward_args
        )
    elif name == 'fit':
        attr = explainer.attribute(
            inputs=inputs,
            additional_forward_args=additional_forward_args
        )
    elif name == 'occlusion':
        attr = explainer.attribute(
            inputs=inputs,
            baselines=baselines,
            sliding_window_shapes = sliding_window_shapes,
            additional_forward_args=additional_forward_args,
            attributions_fn=abs
        )
    elif name=='augmented_occlusion':
        attr = explainer.attribute(
            inputs=inputs,
            sliding_window_shapes = sliding_window_shapes,
            additional_forward_args=additional_forward_args,
            attributions_fn=abs
        )
    else:
        raise NotImplementedError
        
    return reshape_over_output_horizon(attr, inputs, args)

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
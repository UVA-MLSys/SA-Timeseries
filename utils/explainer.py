import torch

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
    
    device = inputs.device
    if mode =='zero': baselines = torch.zeros_like(inputs, device=device).float()
    elif mode == 'random': baselines = torch.randn_like(inputs, device=device).float()
    elif mode == 'aug':
        means = torch.mean(inputs, dim=(0, 1))
        std = torch.std(inputs, dim=(0, 1))
        baselines = torch.normal(means, std).repeat(
            inputs.shape[0], inputs.shape[1], 1
        ).float()
    elif mode == 'mean': 
        baselines = torch.mean(
                inputs, axis=0
        ).repeat(inputs.shape[0], 1, 1).float()
    else:
        print(f'baseline mode options: [zero, random, aug, mean]')
        raise NotImplementedError
    
    return baselines

def min_max_scale(arr, dim:int=0):
    assert dim in [0, 1], f'Dimension must be 0 or 1, found {dim}'
    mx, mn = torch.max(arr, dim=dim).values, torch.min(arr, dim=dim).values
    denom = (mx - mn)
    
    if dim == 0: scaled = (arr - mn)/denom
    else: scaled = ((arr.T - mn)/denom).T
    
    # replace nan values with 0
    # possible when all values are same, hence denom is 0
    scaled[scaled != scaled] = 0
    
    return scaled

def compute_classifier_attr(
    inputs, baselines, explainer,
    additional_forward_args, args
):
    name = explainer.get_name()
    
    if name in ['Deep Lift', 'Lime', 'Integrated Gradients', 'Gradient Shap', 'Feature Ablation']:
        attr = explainer.attribute(
            inputs=inputs, baselines=baselines,
            additional_forward_args=additional_forward_args,
            attributions_fn=abs
        )
    
    elif name in ['Feature Permutation', 'WinIT']:
        attr = explainer.attribute(
            inputs=inputs, attributions_fn=abs,
            additional_forward_args=additional_forward_args
        )
    elif name == 'Occlusion' or name=='Augmented Occlusion':
        if type(inputs) == tuple:
            sliding_window_shapes = tuple([(1,1) for _ in inputs])
        else:
            sliding_window_shapes = (1,1)
            
        if name == 'Occlusion':
            attr = explainer.attribute(
                inputs=inputs, baselines=baselines,
                sliding_window_shapes = sliding_window_shapes,
                additional_forward_args=additional_forward_args,
                attributions_fn=abs
            )
        else:
            attr = explainer.attribute(
                inputs=inputs, attributions_fn=abs,
                sliding_window_shapes = sliding_window_shapes,
                additional_forward_args=additional_forward_args
            )
    else:
        raise NotImplementedError
    
    return avg_over_output_horizon(attr, inputs, args)

def compute_regressor_attr(
    inputs, baselines, explainer,
    additional_forward_args, args
):
    name = explainer.get_name()
    if name in ['Deep Lift', 'Lime', 'Integrated Gradients', 'Gradient Shap']:
        attr_list = []
        for target in range(args.pred_len):
            score = explainer.attribute(
                inputs=inputs, baselines=baselines, target=target,
                additional_forward_args=additional_forward_args
            )
            attr_list.append(score)
        
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
        
    elif name in ['Feature Ablation']:
        attr = explainer.attribute(
            inputs=inputs, baselines=baselines,
            additional_forward_args=additional_forward_args
        )
    elif name in ['Feature Permutation', 'WinIT']:
        attr = explainer.attribute(
            inputs=inputs,
            additional_forward_args=additional_forward_args
        )
    elif name == 'Occlusion' or name=='Augmented Occlusion':
        if type(inputs) == tuple:
            sliding_window_shapes = tuple([(1,1) for _ in inputs])
        else:
            sliding_window_shapes = (1,1)
            
        if name == 'Occlusion':
            attr = explainer.attribute(
                inputs=inputs,
                baselines=baselines,
                sliding_window_shapes = sliding_window_shapes,
                additional_forward_args=additional_forward_args
            )
        else:
            attr = explainer.attribute(
                inputs=inputs,
                sliding_window_shapes = sliding_window_shapes,
                additional_forward_args=additional_forward_args
            )
    else:
        raise NotImplementedError
        
    return avg_over_output_horizon(attr, inputs, args)

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

def compute_classifier_tsr_attr(
    args, explainer, inputs, sliding_window_shapes, 
    strides=None, baselines=None,
    additional_forward_args=None, threshold=0.0, normalize=True
):
    attr_list = []
    for target in range(args.num_class):
        score = explainer.attribute(
            inputs=inputs, sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            baselines=baselines, target=target,
            additional_forward_args=additional_forward_args,
            threshold=threshold, normalize=normalize
        )
        attr_list.append(score)
        
    if type(inputs) == tuple:
            attr = []
            for input_index in range(len(inputs)):
                attr_per_input = torch.stack([score[input_index] for score in attr_list])
                # num_class x batch x seq_len x features -> batch x num_class x seq_len x features
                attr_per_input = attr_per_input.permute(1, 0, 2, 3)
                attr.append(attr_per_input)
                
            attr = tuple(attr)
    else:
        attr = torch.stack(attr_list)
        # num_class x batch x seq_len x features -> batch x num_class x seq_len x features
        attr = attr.permute(1, 0, 2, 3)
    
    return avg_over_output_horizon(attr, inputs, args)

def compute_regressor_tsr_attr(
    args, explainer, inputs, sliding_window_shapes, 
    strides=None, baselines=None,
    additional_forward_args=None, threshold=0.0, normalize=True
):
    attr_list = []
    for target in range(args.pred_len):
        score = explainer.attribute(
            inputs=inputs, sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            baselines=baselines, target=target,
            additional_forward_args=additional_forward_args,
            threshold=threshold, normalize=normalize
        )
        attr_list.append(score)
        # temporarily added to increase speed
        # break 
        
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
    
    return avg_over_output_horizon(attr, inputs, args)
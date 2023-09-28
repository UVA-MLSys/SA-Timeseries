import torch

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

def generate_mask(input, mask_type='zero'):
    if mask_type == 'zero':
        return 0
    elif mask_type == 'random':
        return torch.randn(size=input.shape)
        
    return None

def compute_attr(
    inputs, baselines, explainer,
    additional_forward_args, args
):
    assert type(inputs) == torch.Tensor, \
        f'Only input type tensor supported, found {type(inputs)} instead.'
    name = explainer.get_name()
    
    # these methods don't support having multiple outputs at the same time
    if name in ['Deep Lift', 'Lime', 'Integrated Gradients', 'Gradient Shap']:
        attr_list = []
        for target in range(args.pred_len):
            score = explainer.attribute(
                inputs=inputs, baselines=baselines, target=target,
                additional_forward_args=additional_forward_args
            )
            attr_list.append(score)
        
        attr = torch.stack(attr_list)
        # pred_len x batch x seq_len x features -> batch x pred_len x seq_len x features
        attr = attr.permute(1, 0, 2, 3)
        
    elif name == 'Feature Ablation':
        attr = explainer.attribute(
            inputs=inputs, baselines=baselines,
            additional_forward_args=additional_forward_args
        )
    elif name == 'Occlusion':
        attr = explainer.attribute(
            inputs=inputs,
            baselines=baselines,
            sliding_window_shapes = (1,1),
            additional_forward_args=additional_forward_args
        )
    elif name == 'Augmented Occlusion':
        attr = explainer.attribute(
            inputs=inputs,
            sliding_window_shapes = (1,1),
            additional_forward_args=additional_forward_args
        )
    else:
        print(f'{name} not supported.')
        raise NotImplementedError
    
    # batch x seq_len x features
    attr = attr.reshape(
            # batch x pred_len x seq_len x features
            (inputs.shape[0], args.pred_len, args.seq_len, attr.shape[-1])
        # take mean over the output horizon
        ).mean(axis=1)
    
    return attr

def compute_tsr_attr(
    inputs, baselines, explainer, additional_forward_args, args
):
    actual_attr = compute_attr(inputs, baselines, explainer, additional_forward_args, args)
    # batch x seq_len
    time_attr = torch.zeros((inputs.shape[0], args.seq_len), dtype=float)

    for t in range(args.seq_len):
        new_inputs = inputs.clone() 
        # batch x seq_len x features
        new_inputs[:, t] = inputs[0, 0, -1] # test with new_inputs[:, :t+1] and other masking

        new_attr_per_time = compute_attr(
            new_inputs, baselines, explainer, 
            additional_forward_args, args
        )
        
        # sum the attr difference for each input in the batch
        # batch x seq_len x features -> batch
        time_attr[:, t] = (actual_attr - new_attr_per_time
            ).abs().sum(axis=(1, 2))
    
    # for each input in the batch, normalize along the time axis
    time_attr = min_max_scale(time_attr, dim=1)

    # find median along the time axis
    # mean_time_importance = np.quantile(time_attr, .55, axis=1)   
    
    n_features = inputs.shape[-1]
    input_attr = torch.zeros((inputs.shape[0], n_features), dtype=float)
    time_scaled_attr = torch.zeros_like(actual_attr)

    for t in range(args.seq_len):
        # if time_attr[t] < mean_time_importance:
        #     featureContibution = torch.ones(input_attr, dtype=float)/n_features
        for f in range(n_features):
            new_inputs = inputs.clone() # batch x seq_len x features
            new_inputs[:, t, f] = inputs[0, 0, f]
            
            attr = compute_attr(
                new_inputs, baselines, explainer, 
                additional_forward_args, args
            )
            input_attr[:, f] = (actual_attr - attr).abs().sum(axis=(1, 2))
        
        input_attr = min_max_scale(input_attr, dim=1)
        
        for f in range(n_features):
            time_scaled_attr[:, t, f] = time_attr[:, t] * input_attr[:, f]
            
    return time_scaled_attr
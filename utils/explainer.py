import torch

def get_total_data(dataloader, device, add_x_mark=True):        
    if add_x_mark:
        return (
            torch.vstack([item[0] for item in dataloader]).float().to(device), 
            torch.vstack([item[1] for item in dataloader]).float().to(device)
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
    task = 'multiclass' if args.num_class > 1 else 'binary'
    
    if name in ['Deep Lift', 'Lime', 'Integrated Gradients', 'Gradient Shap', 'Feature Ablation']:
        attr = explainer.attribute(
            inputs=inputs, baselines=baselines, 
            additional_forward_args=additional_forward_args
        )
        
    elif name == 'Feature Permutation':
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
                inputs=inputs, baselines=baselines,
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
    
    if type(inputs) == tuple:
        # tuple of batch x seq_len x features
        attr = tuple([
            score.reshape(
                # batch x seq_len x features
                (inputs[0].shape[0], args.seq_len, score.shape[-1])
            # take mean over the output horizon
            ).mean(axis=1) for score in attr
        ])
    else:
        # batch x seq_len x features
        attr = attr.reshape(
            # batch x seq_len x features
            (inputs.shape[0], args.seq_len, attr.shape[-1])
        # take mean over the output horizon
        ).mean(axis=1)
    
    
    return attr

def compute_attr(
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
    elif name == 'Feature Permutation':
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
    
    if type(inputs) == tuple:
        # tuple of batch x seq_len x features
        attr = tuple([
            score.reshape(
                # batch x pred_len x seq_len x features
                (inputs[0].shape[0], args.pred_len, args.seq_len, score.shape[-1])
            # take mean over the output horizon
            ).mean(axis=1) for score in attr
        ])
    else:
        # batch x seq_len x features
        attr = attr.reshape(
            # batch x pred_len x seq_len x features
            (inputs.shape[0], args.pred_len, args.seq_len, attr.shape[-1])
        # take mean over the output horizon
        ).mean(axis=1)
        
    
    return attr

def compute_classifier_tsr_attr(
    args, explainer, inputs, sliding_window_shapes, 
    strides=None, baselines=None,
    additional_forward_args=None, threshold=0.0, normalize=True
):
    attr = explainer.attribute(
        inputs=inputs, sliding_window_shapes=sliding_window_shapes,
        strides=strides,
        baselines=baselines, 
        additional_forward_args=additional_forward_args,
        threshold=threshold, normalize=normalize
    )
        
    if type(inputs) == tuple:
        # tuple of batch x seq_len x features
        attr = tuple([
            score.reshape(
                # batch x seq_len x features
                (inputs[0].shape[0], args.seq_len, score.shape[-1])
            # take mean over the output horizon
            ).mean(axis=1) for score in attr
        ])
    else:
        # batch x seq_len x features
        attr = attr.reshape(
            # batch x seq_len x features
            (inputs.shape[0], args.seq_len, attr.shape[-1])
        # take mean over the output horizon
        ).mean(axis=1)
    
    return attr

def compute_tsr_attr(
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
    
    
    if type(inputs) == tuple:
        # tuple of batch x seq_len x features
        attr = tuple([
            score.reshape(
                # batch x pred_len x seq_len x features
                (inputs[0].shape[0], args.pred_len, args.seq_len, score.shape[-1])
            # take mean over the output horizon
            ).mean(axis=1) for score in attr
        ])
    else:
        # batch x seq_len x features
        attr = attr.reshape(
            # batch x pred_len x seq_len x features
            (inputs.shape[0], args.pred_len, args.seq_len, attr.shape[-1])
        # take mean over the output horizon
        ).mean(axis=1)
    
    return attr


# def compute_tsr_attr(
#     inputs, baselines, explainer, additional_forward_args, args, device
# ):
#     actual_attr = compute_attr(inputs, baselines, explainer, additional_forward_args, args)
#     # batch x seq_len
#     time_attr = torch.zeros((inputs.shape[0], args.seq_len), dtype=float, device=device)

#     # assignment = torch.randn((inputs.shape[0], inputs.shape[-1]), dtype=float)
#     new_inputs = inputs.clone() 
#     for t in range(args.seq_len):
#         prev_value = new_inputs[:, :t]
#         # batch x seq_len x features
#         new_inputs[:, :t] = 0 # assignment # inputs[0, 0, -1] # test with new_inputs[:, :t+1] and other masking

#         new_attr_per_time = compute_attr(
#             new_inputs, baselines, explainer, 
#             additional_forward_args, args
#         )
        
#         # sum the attr difference for each input in the batch
#         # batch x seq_len x features -> batch
#         time_attr[:, t] = (actual_attr - new_attr_per_time
#             ).abs().sum(axis=(1, 2))
#         new_inputs[:, :t] = prev_value
    
#     # for each input in the batch, normalize along the time axis
#     time_attr = min_max_scale(time_attr, dim=1)
    
#     # new_attr = (time_attr.T * actual_attr.T).T
#     # return new_attr

#     # find median along the time axis
#     # mean_time_importance = np.quantile(time_attr, .55, axis=1)   
    
#     n_features = inputs.shape[-1]
#     input_attr = torch.zeros((inputs.shape[0], n_features), dtype=float, device=device)
#     time_scaled_attr = torch.zeros_like(actual_attr)

#     # assignment = torch.randn((inputs.shape[0],inputs.shape[1]), dtype=float)
#     # for f in range(n_features):
#     #     prev_value = new_inputs[:, :, f]
#     #     new_inputs[:, :, f] = assignment
#     #     attr = compute_attr(
#     #         new_inputs, baselines, explainer, 
#     #         additional_forward_args, args
#     #     )
#     #     input_attr[:, f] = (actual_attr - attr).abs().sum(axis=(1, 2))
#     #     new_inputs[:, :, f] = prev_value
        
#     # input_attr = min_max_scale(input_attr, dim=1)
#     # for t in range(args.seq_len):
#     #     for f in range(n_features):
#     #         time_scaled_attr[:, t, f] = time_attr[:, t] * input_attr[:, f]
#     # return time_scaled_attr
    
#     # new_inputs = inputs.clone()
#     for t in range(args.seq_len):
#         # if time_attr[t] < mean_time_importance:
#         #     featureContibution = torch.ones(input_attr, dtype=float)/n_features
#         for f in range(n_features):
#              # batch x seq_len x features
#             prev_value = new_inputs[:, :t, f]
#             new_inputs[:, :t, f] = 0 # inputs[0, 0, f] # assignment[:, f] # inputs[0, 0, f]
            
#             attr = compute_attr(
#                 new_inputs, baselines, explainer, 
#                 additional_forward_args, args
#             )
#             input_attr[:, f] = (actual_attr - attr).abs().sum(axis=(1, 2))
#             new_inputs[:, :t, f] = prev_value
        
#         input_attr = min_max_scale(input_attr, dim=1)
        
#         for f in range(n_features):
#             time_scaled_attr[:, t, f] = time_attr[:, t] * input_attr[:, f]
            
#     return time_scaled_attr
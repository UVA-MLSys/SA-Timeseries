import torch
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
    
    batch_size = inputs.shape[0]    
    device = inputs.device
    
    if mode =='zero': baselines = torch.zeros_like(inputs, device=device).float()
    elif mode == 'random': baselines = torch.randn_like(inputs, device=device).float()
    elif mode == 'aug':
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
        print(f'baseline mode options: [zero, random, aug, mean]')
        raise NotImplementedError
    
    return baselines

def compute_attr(
    name, inputs, baselines, explainer,
    additional_forward_args, 
    args, avg_attr=True
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
            threshold=0.55, normalize=True,
            attributions_fn=abs
        )
    
    elif name in [
        'deep_lift', 'lime', 'integrated_gradients', 
        'gradient_shap', 'tsr', 'wtsr'
    ]:
        attr_list = []
        
        for target in range(targets):
            if name in ['tsr', 'wtsr']:
                threshold = 0.55 # if name == 'tsr' else 0
                attr = explainer.attribute(
                    inputs=inputs,
                    sliding_window_shapes=sliding_window_shapes,
                    baselines=baselines, target=target,
                    additional_forward_args=additional_forward_args,
                    threshold=threshold, normalize=True,
                    attributions_fn=abs
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
            inputs=inputs, baselines=baselines, 
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
        
    if avg_attr:
        return avg_over_output_horizon(attr, inputs, args)
    else:
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
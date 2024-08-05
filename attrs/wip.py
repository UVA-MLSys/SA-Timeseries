import numpy as np
import torch, gc

from captum.attr._utils.attribution import Attribution, GradientAttribution
from captum.attr._utils.common import (
    _format_input_baseline,
    _format_and_verify_sliding_window_shapes,
    _format_and_verify_strides,
)
from captum._utils.common import _format_output
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

from torch import Tensor
from typing import Any, Callable, Tuple, Union
from utils.tools import normalize_scale
from tint.attr.occlusion import FeatureAblation, Occlusion

from captum.attr import IntegratedGradients
from utils.explainer import compute_attr

class WinIT3:
    def __init__(self, model, data, args):
        self.model = model
        self.args = args
        self.seq_len = args.seq_len
        self.task_name = args.task_name
        self.data = data
        self.explainer = IntegratedGradients(self.model)
        self.occluder = Occlusion(self.model)
    
    def format_output(self, outputs):
        if self.task_name == 'classification':
            return torch.nn.functional.softmax(outputs, dim=1)
        else:
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            return outputs
        
    def get_feature_relevance_score(self, inputs, additional_forward_args, baselines):
        input_is_tuple = type(inputs) == tuple
        if not input_is_tuple:
            inputs = tuple([inputs])
            baselines = tuple([baselines])
            
        tsr_sliding_window_shapes = tuple(
            (input.shape[2], 1) for input in inputs
        )
        time_relevance_score = self.occluder.attribute(
            inputs=inputs,
            sliding_window_shapes=tsr_sliding_window_shapes,
            additional_forward_args=additional_forward_args,
            attributions_fn=abs,
            baselines=baselines,
            #TODO: uncomment after new release
            # kwargs_run_forward=kwargs,
        )
    
        # time_relevance_score shape will be ((N x O) x seq_len) after summation
        time_relevance_score = tuple(
            tsr.sum(dim=1)
            for tsr in time_relevance_score
        )
        time_relevance_score = tuple(
            normalize_scale(
                tsr, dim=1, norm_type="l1"
            ) for tsr in time_relevance_score
        )
        
        if input_is_tuple:
            return time_relevance_score
        else: return time_relevance_score[0]
        
    def get_time_relevance_score(self, inputs, additional_forward_args, baselines):
        input_is_tuple = type(inputs) == tuple
        if not input_is_tuple:
            inputs = tuple([inputs])
            baselines = tuple([baselines])
            
        tsr_sliding_window_shapes = tuple(
            (1,) + input.shape[2:] for input in inputs
        )
        time_relevance_score = self.occluder.attribute(
            inputs=inputs,
            sliding_window_shapes=tsr_sliding_window_shapes,
            additional_forward_args=additional_forward_args,
            attributions_fn=abs,
            baselines=baselines,
            #TODO: uncomment after new release
            # kwargs_run_forward=kwargs,
        )
    
        # time_relevance_score shape will be ((N x O) x seq_len) after summation
        time_relevance_score = tuple(
            tsr.sum((tuple(i for i in range(2, len(tsr.shape)))))
            for tsr in time_relevance_score
        )
        time_relevance_score = tuple(
            normalize_scale(
                tsr, dim=1, norm_type="l1"
            ) for tsr in time_relevance_score
        )
        
        if input_is_tuple:
            return time_relevance_score
        else: return time_relevance_score[0]
            
    def attribute(
        self, inputs:Union[torch.Tensor , Tuple[torch.Tensor]], 
        additional_forward_args: Tuple[torch.Tensor], 
        baselines,
        attributions_fn=None, threshold=0.55
    ):
        input_is_tuple = type(inputs) == tuple
        if not input_is_tuple:
            inputs = tuple([inputs])
            baselines = tuple([baselines])
            
        with torch.no_grad():
            # (batch_size x n_output) x seq_len
            time_relevance_score = self.get_time_relevance_score(
                inputs, additional_forward_args, baselines
            )
            
            is_above_threshold = tuple(
                score > torch.quantile(score, threshold, dim=1, keepdim=True) for score in time_relevance_score
            )
            
            # batch_size x n_output x seq_len x features
            attrs = compute_attr(
                'occlusion', inputs, baselines, 
                self.occluder, additional_forward_args, self.args
            )
            
            # (batch_size x n_output) x seq_len x features
            attrs = tuple([
                attr.reshape((-1, attr.shape[-2], attr.shape[-1])) for attr in attrs
            ])
            
            time_relevance_score = tuple(
                tsr.reshape(attr.shape[:2] + (1,) * len(attr.shape[2:]))
                for attr, tsr in zip(attrs, time_relevance_score)
            )

            is_above_threshold = tuple(
                is_above.reshape(attr.shape[:2] + (1,) * len(attr.shape[2:]))
                for attr, is_above in zip(attrs, is_above_threshold)
            )
            attributions = tuple(
                tsr * attr * is_above.float()
                for tsr, is_above, attr in zip(
                    time_relevance_score,
                    is_above_threshold,
                    attrs
                )
            )

            if input_is_tuple: return attributions
            else: return attributions[0]
    
    def get_name():
        return 'WinIT3'

class WinIT2:
    def __init__(self, model, data, args):
        self.model = model
        self.args = args
        self.seq_len = args.seq_len
        self.task_name = args.task_name
        self.data = data
        self.explainer = Occlusion(self.model)
        
        if self.task_name =='classification':
            self.metric = 'js'
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
            p_y_hat = torch.softmax(p_y_hat, dim=1)
            p_y_exp = torch.softmax(p_y_exp, dim=1)
            
            kl_loss = torch.nn.KLDivLoss(reduction="none")(torch.log(p_y_hat), p_y_exp)
            # return torch.sum(kl_loss, -1)
            return kl_loss
        if self.metric == "js":
            p_y_hat = torch.softmax(p_y_hat, dim=1)
            p_y_exp = torch.softmax(p_y_exp, dim=1)
            
            average = (p_y_hat + p_y_exp) / 2
            lhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_hat)
            rhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_exp)
            return (lhs + rhs) / 2
        if self.metric == "pd":
            # batch_size x output_horizon x num_states
            diff = torch.abs(p_y_hat - p_y_exp)
            # sum over all dimension except batch, output_horizon
            # multioutput multi-horizon not supported yet
            return torch.sum(diff, dim=-1)
        
        raise Exception(f"unknown metric. {self.metric}")
    
    def generate_counterfactuals(self, batch_size, input_index, feature_index):
        if input_index is None:
            n_features = self.data.shape[-1]
            if feature_index is None:
                # take all features
                choices = self.data.reshape((-1, n_features))
            else:
                # take one feature
                choices = self.data[:, :, feature_index].reshape(-1)
        else:
            n_features = self.data[input_index].shape[-1]
            if feature_index is None:
                choices = self.data[input_index][:].reshape((-1, n_features))    
            else:
                choices = self.data[input_index][:].reshape(-1)

        sampled_index = np.random.choice(range(len(choices)), size=(batch_size*self.seq_len))
        
        if feature_index is None:
            samples = choices[sampled_index].reshape((batch_size, self.seq_len, n_features))
        else:
            samples = choices[sampled_index].reshape((batch_size, self.seq_len))
        
        return samples
    
    def format_output(self, outputs):
        if self.task_name == 'classification':
            return torch.nn.functional.softmax(outputs, dim=1)
        else:
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            return outputs
        
    def get_time_relevance_score(self, inputs, additional_forward_args, baselines):
        tsr_sliding_window_shapes = tuple(
            (1,) + input.shape[2:] for input in inputs
        )
        time_relevance_score = self.explainer.attribute(
            inputs=inputs,
            sliding_window_shapes=tsr_sliding_window_shapes,
            additional_forward_args=additional_forward_args,
            attributions_fn=abs,
            baselines=baselines,
            #TODO: uncomment after new release
            # kwargs_run_forward=kwargs,
        )
    
        # time_relevance_score shape will be ((N x O) x seq_len) after summation
        time_relevance_score = tuple(
            tsr.sum((tuple(i for i in range(2, len(tsr.shape)))))
            for tsr in time_relevance_score
        )
        time_relevance_score = tuple(
            normalize_scale(
                tsr, dim=1, norm_type="l1"
            ) for tsr in time_relevance_score
        )
        return time_relevance_score
            
    def attribute(
        self, inputs:Union[torch.Tensor , Tuple[torch.Tensor]], 
        additional_forward_args: Tuple[torch.Tensor], 
        baselines,
        attributions_fn=None, threshold=0.55
    ):
        model = self.model
        if type(inputs) != tuple:
            inputs = tuple([inputs])
            baselines = tuple([baselines])
            
        attr = []
        with torch.no_grad():
            y_original = self.format_output(
                model(*inputs, *additional_forward_args)
            ) 
            
            time_relevance_score = self.get_time_relevance_score(
                inputs, additional_forward_args, baselines
            )
            
            above_threshold = tuple(
                score > torch.quantile(score, threshold, dim=1, keepdim=True) for score in time_relevance_score
            )
            # print(time_relevance_score)
            for input_index in range(len(inputs)):
                batch_size, seq_len, n_features = inputs[input_index].shape
            
                # batch_size, n_output, seq_len, n_features
                iS_array = torch.zeros(size=(
                    batch_size * y_original.shape[1], seq_len, n_features
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
                        # if not above_threshold[input_index][feature][t]: continue
                        prev_value = cloned[:, t, feature]
                        # mask last t timesteps
                        
                        cloned[:, t, feature] = counterfactuals[:, t] # baselines[input_index][:, t, feature] # 
                        
                        inputs_hat = []
                        for i in range(len(inputs)):
                            if i == input_index: inputs_hat.append(cloned)
                            else: inputs_hat.append(inputs[i])
                        
                        y_perturbed = self.format_output(
                            model(*tuple(inputs_hat), *additional_forward_args)
                        )
                        
                        iSab = self._compute_metric(y_original, y_perturbed)
                        iSab = torch.clip(iSab, -1e6, 1e6)
                        iS_array[:, t, feature] = abs(iSab).reshape(-1)
                        cloned[:, t, feature] = prev_value
                        
                        del y_perturbed, inputs_hat
                    del cloned, # counterfactuals
                    gc.collect()
            
                # print('1 ', iS_array)               
                # batch_size, n_output, seq_len, n_features        
                iS_array[:, 1:] -= iS_array[:, :-1] 
                
                # reverse order along the time axis
                iS_array = iS_array.flip(dims=(1,))
                iS_array = normalize_scale(
                    iS_array, dim=(1, 2), norm_type="minmax"
                )
                # print('2 ', iS_array)   
                iS_array = (iS_array.T * time_relevance_score[input_index].T * above_threshold[input_index].T).T
                # print('3 ', iS_array)   
                
                if attributions_fn is not None:
                    attr.append(attributions_fn(iS_array))
                else: attr.append(iS_array)
           
        if len(attr) == 1: return attr[0]
        else: return tuple(attr)
    
    def get_name():
        return 'WinIT2'

class Decoupled(Occlusion):
    def __init__(
        self,
        attribution_method: Attribution,
        metric='pd'
    ) -> None:
        self.attribution_method = attribution_method
        self.is_delta_supported = False
        self._multiply_by_inputs = self.attribution_method.multiplies_by_inputs
        self.is_gradient_method = isinstance(
            self.attribution_method, GradientAttribution
        )
        
        assert metric in ['kl', 'js', 'pd'], 'metric must be one of kl, js or pd'
        self.metric = metric
        
        Occlusion.__init__(self, self.attribution_method.forward_func)
        self.fa = FeatureAblation(self.attribution_method.forward_func)
        self.use_weights = False  # We do not use weights for this method

    @property
    def multiplies_by_inputs(self):
        return self._multiply_by_inputs
    
    def _compute_metric(
        self, p_y_exp: torch.Tensor, p_y_hat: torch.Tensor
    ) -> torch.Tensor:
        if self.metric == "kl":
            kl_loss = torch.nn.KLDivLoss(reduction="none")(torch.log(p_y_hat), p_y_exp)
            # return torch.sum(kl_loss, -1)
            return kl_loss
        if self.metric == "js":
            p_y_hat = torch.sigmoid(p_y_hat)
            p_y_exp = torch.sigmoid(p_y_exp)
            
            average = (p_y_hat + p_y_exp) / 2
            lhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_hat)
            rhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_exp)
            diff = (lhs + rhs) / 2
            return diff
            # return torch.sum((lhs + rhs) / 2, -1)
        if self.metric == "pd":
            # batch_size x output_horizon x num_states
            diff = torch.abs(p_y_hat - p_y_exp)
            
            # sum over all dimension except batch, output_horizon
            # multioutput multi-horizon not supported yet
            return diff
        
        raise Exception(f"unknown metric. {self.metric}")

    def has_convergence_delta(self) -> bool:
        return False

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        sliding_window_shapes: Union[
            Tuple[int, ...], Tuple[Tuple[int, ...], ...]
        ],
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        normalize: bool = True,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> TensorOrTupleOfTensorsGeneric:
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs, baselines = _format_input_baseline(inputs, baselines)

        assert all(
            x.shape[1] == inputs[0].shape[1] for x in inputs
        ), "All inputs must have the same time dimension. (dimension 1)"

        # Compute sliding window for the Time-Relevance Score
        # Only the time dimension (dim 1) has a sliding window of 1

        time_relevance_score = self.get_time_relevance_score(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=show_progress
        )
        feature_relevance_score = self.get_feature_relevance_score(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=show_progress
        )    
        
        # Normalize if required
        if normalize:
            # normalize the last dimension
            time_relevance_score = tuple(
                normalize_scale(
                    tsr, dim=-1, norm_type="l1"
                ) for tsr in time_relevance_score
            )
            
            feature_relevance_score = tuple(
                normalize_scale(
                    fsr, dim=-1, norm_type="l1"
                ) for fsr in feature_relevance_score
            )
        # Formatting sliding window shapes
        sliding_window_shapes = _format_and_verify_sliding_window_shapes(
            sliding_window_shapes, inputs
        )

        # Construct tensors from sliding window shapes
        sliding_window_tensors = tuple(
            torch.ones(window_shape, device=inputs[i].device)
            for i, window_shape in enumerate(sliding_window_shapes)
        )
        attr = self.fa.attribute(
            inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            sliding_window_tensors=sliding_window_tensors,
            attributions_fn=abs
            #TODO: uncomment after new release
            # kwargs_run_forward=kwargs,
        )
        
        # print([a.shape for a in attr], [i.shape for i in time_relevance_score], [i.shape for i in feature_relevance_score])
        
        time_relevance_score = tuple(
            tsr.reshape(a.shape[:2] + (1,) * len(a.shape[2:]))
            for a, tsr in zip(attr, time_relevance_score)
        )
        feature_relevance_score = tuple(
            f_imp.reshape(a.shape[0], 1, a.shape[2])
            for a, f_imp in zip(attr, feature_relevance_score)
        )
        
        # print(time_relevance_score[0])
        # print(feature_relevance_score[0])
        # print(attr[0])
        
        attributions = tuple(
            ((tsr + frs) * a)
            for tsr, frs, a in zip(
                time_relevance_score,
                feature_relevance_score,
                attr
            )
        )    
        return _format_output(is_inputs_tuple, attributions)
    
    def get_time_relevance_score(
            self, inputs: TensorOrTupleOfTensorsGeneric,
            baselines: BaselineType = None,
            target: TargetType = None,
            additional_forward_args: Any = None,
            perturbations_per_eval=1,
            show_progress=False
        ):
        tsr_sliding_window_shapes = tuple(
            (1,) + input.shape[2:] for input in inputs
        )
        time_relevance_score = super().attribute.__wrapped__(
            self,
            inputs=inputs,
            sliding_window_shapes=tsr_sliding_window_shapes,
            strides=None,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            attributions_fn=abs,
            show_progress=show_progress,
            #TODO: uncomment after new release
            # kwargs_run_forward=kwargs,
        )
        
        # time_relevance_score shape will be ((N x O) x seq_len) after summation
        time_relevance_score = tuple(
            tsr.sum((tuple(i for i in range(2, len(tsr.shape)))))
            for tsr in time_relevance_score
        )
        return time_relevance_score
    
    def get_feature_relevance_score(
            self, inputs: TensorOrTupleOfTensorsGeneric,
            baselines: BaselineType = None,
            target: TargetType = None,
            additional_forward_args: Any = None,
            perturbations_per_eval=1,
            show_progress=False
        ):
        fsr_sliding_window_shapes = tuple(
            (input.shape[1], 1) for input in inputs
        )
        feature_relevance_score = super().attribute.__wrapped__(
            self,
            inputs=inputs,
            sliding_window_shapes=fsr_sliding_window_shapes,
            strides=None,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            attributions_fn=abs,
            show_progress=show_progress,
            #TODO: uncomment after new release
            # kwargs_run_forward=kwargs,
        )
        
        # feature_relevance_score shape will be ((N x O) x seq_len) after summation
        feature_relevance_score = tuple(
            # batch x seq_len x feature, sum across the time dimension
            fsr.sum(dim=1)
            for fsr in feature_relevance_score
        )
        return feature_relevance_score

    def _construct_ablated_input(
        self,
        expanded_input: Tensor,
        input_mask: Union[None, Tensor],
        baseline: Union[Tensor, int, float],
        start_feature: int,
        end_feature: int,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Ablates given expanded_input tensor with given feature mask, feature range,
        and baselines, and any additional arguments.
        expanded_input shape is (num_features, num_examples, ...)
        with remaining dimensions corresponding to remaining original tensor
        dimensions and num_features = end_feature - start_feature.
        input_mask is None for occlusion, and the mask is constructed
        using sliding_window_tensors, strides, and shift counts, which are provided in
        kwargs. baseline is expected to
        be broadcastable to match expanded_input.
        This method returns the ablated input tensor, which has the same
        dimensionality as expanded_input as well as the corresponding mask with
        either the same dimensionality as expanded_input or second dimension
        being 1. This mask contains 1s in locations which have been ablated (and
        thus counted towards ablations for that feature) and 0s otherwise.
        """
        input_mask = torch.stack(
            [
                self._occlusion_mask(
                    expanded_input,
                    j,
                    kwargs["sliding_window_tensors"],
                    kwargs["strides"],
                    kwargs["shift_counts"],
                    kwargs.get("is_above_threshold", None),
                )
                for j in range(start_feature, end_feature)
            ],
            dim=0,
        ).long()
        # print('Expanded input ', expanded_input.shape, input_mask.shape)
        
        ablated_tensor = (
            expanded_input
            * (
                torch.ones(1, dtype=torch.long, device=expanded_input.device)
                - input_mask[:, :expanded_input.shape[1]]
            ).to(expanded_input.dtype)
        ) + (baseline * input_mask[:, :expanded_input.shape[1]].to(expanded_input.dtype))

        return ablated_tensor, input_mask

    def _occlusion_mask(
        self,
        expanded_input: Tensor,
        ablated_feature_num: int,
        sliding_window_tsr: Tensor,
        strides: Union[int, Tuple[int, ...]],
        shift_counts: Tuple[int, ...],
        is_above_threshold: Tensor = None,
    ) -> Tensor:
        """
        This constructs the current occlusion mask, which is the appropriate
        shift of the sliding window tensor based on the ablated feature number.
        The feature number ranges between 0 and the product of the shift counts
        (# of times the sliding window should be shifted in each dimension).
        First, the ablated feature number is converted to the number of steps in
        each dimension from the origin, based on shift counts. This procedure
        is similar to a base conversion, with the position values equal to shift
        counts. The feature number is first taken modulo shift_counts[0] to
        get the number of shifts in the first dimension (each shift
        by shift_count[0]), and then divided by shift_count[0].
        The procedure is then continued for each element of shift_count. This
        computes the total shift in each direction for the sliding window.
        We then need to compute the padding required after the window in each
        dimension, which is equal to the total input dimension minus the sliding
        window dimension minus the (left) shift amount. We construct the
        array pad_values which contains the left and right pad values for each
        dimension, in reverse order of dimensions, starting from the last one.
        Once these padding values are computed, we pad the sliding window tensor
        of 1s with 0s appropriately, which is the corresponding mask,
        and the result will match the input shape.
        """
        if is_above_threshold is None:
            return super()._occlusion_mask(
                expanded_input=expanded_input,
                ablated_feature_num=ablated_feature_num,
                sliding_window_tsr=sliding_window_tsr,
                strides=strides,
                shift_counts=shift_counts,
            )

        # We first compute the hyper-rectangle on the non-temporal dims
        padded_tensor = super()._occlusion_mask(
            expanded_input=expanded_input[:, :, 0],
            ablated_feature_num=ablated_feature_num,
            sliding_window_tsr=torch.ones(sliding_window_tsr.shape[1:]),
            strides=strides[1:] if isinstance(strides, tuple) else strides,
            shift_counts=shift_counts[1:],
        ).to(expanded_input.device)

        # We get the current index and batch size
        bsz = expanded_input.shape[1]
        shift_count = shift_counts[0]
        stride = strides[0] if isinstance(strides, tuple) else strides
        current_index = (ablated_feature_num % shift_count) * stride

        # On the temporal dim, the hyper-rectangle is only applied on
        # non-zeros elements
        is_above = is_above_threshold.clone()
        for batch_idx in range(bsz):
            nonzero = is_above_threshold[batch_idx].nonzero()[:, 0]
            is_above[
                batch_idx,
                nonzero[
                    current_index : current_index + sliding_window_tsr.shape[0]
                ],
            ] = 0

        current_mask = is_above.unsqueeze(-1) * padded_tensor.unsqueeze(0)
        # print('Mask shape ', current_mask.shape)
        return current_mask

    def _run_forward(
        self, forward_func: Callable, inputs: Any, **kwargs
    ) -> Tuple[Tuple[Tensor, ...], Tuple[Tuple[int]]]:
        attributions = self.attribution_method.attribute.__wrapped__(
            self.attribution_method, inputs, **kwargs
        )

        # Check if it needs to return convergence delta
        return_convergence_delta = (
            "return_convergence_delta" in kwargs
            and kwargs["return_convergence_delta"]
        )

        # If the method returns delta, we ignore it
        if self.is_delta_supported and return_convergence_delta:
            attributions, _ = attributions

        # Get attr shapes
        attributions_shape = tuple(tuple(attr.shape) for attr in attributions)

        return attributions, attributions_shape

    @staticmethod
    def _reshape_eval_diff(eval_diff: Tensor, shapes: tuple) -> Tensor:
        # For this method, we need to reshape eval_diff to the output shapes
        return eval_diff.reshape((len(eval_diff),) + shapes)
    
    @staticmethod
    def get_name():
        return 'WIP'
    
class WinTSR2(Occlusion):
    def __init__(
        self, model, args, metric='pd'
    ) -> None:
        self.model = model
        self.occluder = Occlusion(self.model)
        self.gradient = IntegratedGradients(self.model)
        
        assert metric in ['kl', 'js', 'pd'], 'metric must be one of kl, js or pd'
        self.metric = metric
        self.args = args

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
            # p_y_hat = torch.softmax(p_y_hat, dim=1)
            # p_y_exp = torch.softmax(p_y_exp, dim=1)
            
            kl_loss = torch.nn.KLDivLoss(reduction="none")(torch.log(p_y_hat), p_y_exp)
            # return torch.sum(kl_loss, -1)
            return kl_loss
        if self.metric == "js":
            p_y_hat = torch.softmax(p_y_hat, dim=1)
            p_y_exp = torch.softmax(p_y_exp, dim=1)
            
            average = (p_y_hat + p_y_exp) / 2
            lhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_hat)
            rhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_exp)
            return (lhs + rhs) / 2
        if self.metric == "pd":
            # batch_size x output_horizon x num_states
            diff = torch.abs(p_y_hat - p_y_exp)
            
            
            # sum over all dimension except batch, output_horizon
            # multioutput multi-horizon not supported yet
            # return torch.sum(diff, dim=-1)
            return diff
        
        raise Exception(f"unknown metric. {self.metric}")

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        sliding_window_shapes: Union[
            Tuple[int, ...], Tuple[Tuple[int, ...], ...]
        ],
        strides: Union[
            None, int, Tuple[int, ...], Tuple[Union[int, Tuple[int, ...]], ...]
        ] = None,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        threshold: float = 0.0,
        normalize: bool = True,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> TensorOrTupleOfTensorsGeneric:
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs, baselines = _format_input_baseline(inputs, baselines)

        assert all(
            x.shape[1] == inputs[0].shape[1] for x in inputs
        ), "All inputs must have the same time dimension. (dimension 1)"

        # Compute sliding window for the Time-Relevance Score
        # Only the time dimension (dim 1) has a sliding window of 1
        # shape (batch_size * n_output) x seq_len  
        time_relevance_score = self.get_time_relevance_score(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=show_progress
        )  
        
        # Normalize if required along the time axis
        if normalize:
            # normalize the last dimension
            time_relevance_score = tuple(
                normalize_scale(
                    tsr, dim=-1, norm_type="minmax"
                ) for tsr in time_relevance_score
            )
            
        # print([score for score in time_relevance_score])   
        # Get indexes where the Time-Relevance Score is
        # higher than the threshold
        is_above_threshold = tuple(
            score > torch.quantile(score, threshold, dim=-1, keepdim=True) for score in time_relevance_score
        )
        # is_above_threshold = tuple(
        #     is_above.reshape(f_imp.shape[:2] + (1,) * len(f_imp.shape[2:]))
        #     for f_imp, is_above in zip(features_relevance_score, is_above_threshold)
        # )
        # feature_relevance_score = self.occluder.attribute(
        #     inputs=inputs,
        #     sliding_window_shapes=sliding_window_shapes,
        #     strides=strides,
        #     baselines=baselines,
        #     target=target,
        #     additional_forward_args=additional_forward_args,
        #     attributions_fn=abs,
        # )
        
        # batch_size x output_horizon x seq_len x features
        feature_relevance_score = compute_attr(
            'integrated_gradients', inputs, baselines, 
            self.gradient, additional_forward_args,
            args=self.args
        )
        
        feature_relevance_score = tuple([
            score.reshape((-1,score.shape[-2], score.shape[-1])) for score in feature_relevance_score
        ])
        
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
            
        if is_inputs_tuple: return attributions
        else: return attributions[0]  
        
        with torch.no_grad():
            y_original = self.model(*inputs, *additional_forward_args)
            
            features_relevance_score = []
            
            for input_index in range(len(inputs)):
                batch_size, seq_len, n_features = inputs[input_index].shape
                cloned = inputs[input_index].clone()
            
                # batch_size, n_output, seq_len, n_features
                iS_array = torch.zeros(size=(
                    batch_size * y_original.shape[1], seq_len, n_features
                ), device=inputs[input_index].device) 
                
                for feature in range(n_features):
                    for t in range(seq_len):
                        if not is_above_threshold[input_index][t].any(): continue
                        
                        prev_value = cloned[:, t, feature]
                        cloned[:, t, feature] = baselines[input_index][:, t, feature] # 
                        
                        inputs_hat = []
                        for i in range(len(inputs)):
                            if i == input_index: inputs_hat.append(cloned)
                            else: inputs_hat.append(inputs[i])
                        
                        y_perturbed = self.model(*tuple(inputs_hat), *additional_forward_args)

                        iSab = self._compute_metric(y_original, y_perturbed) # abs(y_original - y_perturbed)
                        iSab = torch.clip(iSab, -1e6, 1e6)
                        
                        iS_array[:, t, feature] = iSab.flatten()
                        cloned[:, t, feature] = prev_value
                        
                        del y_perturbed, inputs_hat, iSab
                    gc.collect()
                del cloned
                
                # if normalize:
                #     iS_array = normalize_scale(
                #         iS_array, dim=0, norm_type="minmax"
                #     )

                # print('2 ', iS_array)   
                # iS_array = (iS_array.T * time_relevance_score[input_index].T * is_above_threshold[input_index].T).T
                iS_array = (iS_array.T * time_relevance_score[input_index].T ).T
                # print('3 ', iS_array)   
                
                features_relevance_score.append(iS_array)
          
        if is_inputs_tuple:
            return tuple(features_relevance_score)
        else:
            return features_relevance_score[0]
    
    def get_time_relevance_score(
            self, inputs: TensorOrTupleOfTensorsGeneric,
            baselines: BaselineType = None,
            target: TargetType = None,
            additional_forward_args: Any = None,
            perturbations_per_eval=1,
            show_progress=False
        ):
        tsr_sliding_window_shapes = tuple(
            (1,) + input.shape[2:] for input in inputs
        )
        time_relevance_score = self.occluder.attribute(
            inputs=inputs,
            sliding_window_shapes=tsr_sliding_window_shapes,
            strides=None,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            attributions_fn=abs,
            show_progress=show_progress,
            #TODO: uncomment after new release
            # kwargs_run_forward=kwargs,
        )
        
        # time_relevance_score shape will be ((N x O) x seq_len) after summation
        time_relevance_score = tuple(
            tsr.sum((tuple(i for i in range(2, len(tsr.shape)))))
            for tsr in time_relevance_score
        )
        return time_relevance_score
    
    @staticmethod
    def get_name():
        return 'WinTSR'
    
class OcclusionTSR(Occlusion):
    r"""
    Two-step temporal saliency rescaling Tunnel.
    Performs a two-step interpretation method:
    - Mask all features at each time and compute the difference in the
      resulting attribution.
    - Mask each feature at each time and compute the difference in the
      resulting attribution, if the result of the first step is higher
      than a threshold.
    By default, the masked features are replaced with zeros. However, a
    custom baseline can also be passed.
    Using the arguments ``sliding_window_shapes`` and ``strides``, different
    alternatives of TSR can be used:
    - If:
      - :attr:`sliding_window_shapes` = `(1, 1, ...)`
      - :attr:`strides` = `1`
      - :attr:`threshold` = :math:`\alpha`
      the Feature-Relevance Score is computed by masking each feature
      individually providing the Time-Relevance Score is above the threshold.
      This corresponds to the **Temporal Saliency Rescaling** (TSR) method
      (Algorithm 1).
    - If:
      - :attr:`sliding_window_shapes` = `(1, G, G, ...)`
      - :attr:`strides` = `(1, G, G, ...)`
      - :attr:`threshold` = :math:`\alpha`
      the Feature-Relevance Score is computed by masking each feature as a
      group of G features. This corresponds to the **Temporal Saliency
      Rescaling With Feature Grouping** method (Algorithm 2).
    - If:
      - :attr:`sliding_window_shapes` = `(inputs.shape[1], 1, 1, ...)`
      - :attr:`strides` = `1`
      - :attr:`threshold` = `0.0`
      the Feature-Relevance Score is computed by first masking each features
      individually at every time steps. This corresponds to the **Temporal
      Feature Saliency Rescaling** (TFSR) method (Algorithm 3).
    .. hint::
        The convergence delta is ignored by this method, even if explicitely
        required by the attribution method.
    .. warning::
        The attribution method used must output a tensor or tuple of tensor
        of the same size as the inputs.
    Args:
        attribution_method (Attribution): An instance of any attribution algorithm
                    of type `Attribution`. E.g. Integrated Gradients,
                    Conductance or Saliency.
    References:
        `Benchmarking Deep Learning Interpretability in Time Series Predictions <https://arxiv.org/abs/2010.13924>`_
    """

    def __init__(
        self,
        attribution_method: Attribution,
    ) -> None:
        self.attribution_method = attribution_method
        self.is_delta_supported = False
        self._multiply_by_inputs = self.attribution_method.multiplies_by_inputs
        self.is_gradient_method = isinstance(
            self.attribution_method, GradientAttribution
        )
        Occlusion.__init__(self, self.attribution_method.forward_func)
        self.use_weights = False  # We do not use weights for this method

    @property
    def multiplies_by_inputs(self):
        return self._multiply_by_inputs

    def has_convergence_delta(self) -> bool:
        return False
    
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        sliding_window_shapes: Union[
            Tuple[int, ...], Tuple[Tuple[int, ...], ...]
        ],
        strides: Union[
            None, int, Tuple[int, ...], Tuple[Union[int, Tuple[int, ...]], ...]
        ] = None,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        threshold: float = 0.0,
        normalize: bool = True,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> TensorOrTupleOfTensorsGeneric:
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs, baselines = _format_input_baseline(inputs, baselines)

        assert all(
            x.shape[1] == inputs[0].shape[1] for x in inputs
        ), "All inputs must have the same time dimension. (dimension 1)"

        # Compute sliding window for the Time-Relevance Score
        # Only the time dimension (dim 1) has a sliding window of 1
        tsr_sliding_window_shapes = tuple(
            (1,) + input.shape[2:] for input in inputs
        )

        # Compute the Time-Relevance Score (step 1)
        time_relevance_score = super().attribute.__wrapped__(
            self,
            inputs=inputs,
            sliding_window_shapes=tsr_sliding_window_shapes,
            strides=None,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            attributions_fn=abs,
            show_progress=show_progress,
            #TODO: uncomment after new release
            # kwargs_run_forward=kwargs,
        )

        # Reshape the Time-Relevance Score and sum along the diagonal
        assert all(
            tsr.shape == input.shape
            for input, tsr in zip(inputs, time_relevance_score)
        ), "The attribution method must return a tensor of the same shape as the inputs"
        time_relevance_score = tuple(
            tsr.sum((tuple(i for i in range(2, len(tsr.shape)))))
            for tsr in time_relevance_score
        )

        # Normalize if required
        if normalize:
            # min max scaling
            time_relevance_score = tuple(
                normalize_scale(
                    tsr, dim=1, norm_type="minmax"
                ) for tsr in time_relevance_score
            )
            
        # Get indexes where the Time-Relevance Score is
        # higher than the threshold
        is_above_threshold = tuple(
            score > torch.quantile(score, threshold, dim=1, keepdim=True) for score in time_relevance_score
        )
        
        # Formatting strides
        strides = _format_and_verify_strides(strides, inputs)

        # Formatting sliding window shapes
        sliding_window_shapes = _format_and_verify_sliding_window_shapes(
            sliding_window_shapes, inputs
        )

        # Construct tensors from sliding window shapes
        sliding_window_tensors = tuple(
            torch.ones(window_shape, device=inputs[i].device)
            for i, window_shape in enumerate(sliding_window_shapes)
        )

        # Construct number of steps taking the threshold into account
        shift_counts = []
        for i, inp in enumerate(inputs):
            current_shape = np.subtract(
                inp.shape[2:], sliding_window_shapes[i][1:]
            )

            # On the temporal dim, the count shift is the maximum number
            # of element above the threshold
            non_zero_count = torch.unique(
                is_above_threshold[i].nonzero()[:, 0], return_counts=True
            )[1]
            if non_zero_count.sum() == 0:
                shift_count_time_dim = np.array([0])
            else:
                shift_count_time_dim = np.subtract(
                    non_zero_count.max().item(), sliding_window_shapes[i][0]
                )
            current_shape = np.insert(current_shape, 0, shift_count_time_dim)

            shift_counts.append(
                tuple(
                    np.add(
                        np.ceil(np.divide(current_shape, strides[i])).astype(
                            int
                        ),
                        1,
                    )
                )
            )

        # Compute Feature-Relevance Score (step 2)
        features_relevance_score = FeatureAblation.attribute.__wrapped__(
            self,
            inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            sliding_window_tensors=sliding_window_tensors,
            shift_counts=tuple(shift_counts),
            is_above_threshold=is_above_threshold,
            strides=strides,
            attributions_fn=abs,
            show_progress=show_progress,
            #TODO: uncomment after new release
            # kwargs_run_forward=kwargs,
        )
        
        if normalize:
            features_relevance_score = tuple(
                normalize_scale(frs, dim=0, norm_type="minmax") for frs in features_relevance_score
            )

        # Reshape attributions before merge
        time_relevance_score = tuple(
            tsr.reshape(input.shape[:2] + (1,) * len(input.shape[2:]))
            for input, tsr in zip(inputs, time_relevance_score)
        )
        
        # is_above_threshold = tuple(
        #     is_above.reshape(input.shape[:2] + (1,) * len(input.shape[2:]))
        #     for input, is_above in zip(inputs, is_above_threshold)
        # )

        # Merge attributions:
        # Time-Relevance Score x Feature-Relevance Score x is above threshold
        # attributions = tuple(
        #     (tsr * frs) * is_above.float()
        #     for tsr, frs, is_above in zip(
        #         time_relevance_score,
        #         features_relevance_score,
        #         is_above_threshold,
        #     )
        # )
        for input_index in range(len(inputs)):
            length, seq_len = is_above_threshold[input_index].shape
            for b in range(length):
                for t in range(seq_len):
                    t_imp = time_relevance_score[input_index][b, t]
                    if is_above_threshold[input_index][b, t]:
                        features_relevance_score[input_index][b, t] *= t_imp
                    else:
                        features_relevance_score[input_index][b, t] = t_imp * 0.1

        return _format_output(is_inputs_tuple, features_relevance_score)

    def _construct_ablated_input(
        self,
        expanded_input: Tensor,
        input_mask: Union[None, Tensor],
        baseline: Union[Tensor, int, float],
        start_feature: int,
        end_feature: int,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Ablates given expanded_input tensor with given feature mask, feature range,
        and baselines, and any additional arguments.
        expanded_input shape is (num_features, num_examples, ...)
        with remaining dimensions corresponding to remaining original tensor
        dimensions and num_features = end_feature - start_feature.
        input_mask is None for occlusion, and the mask is constructed
        using sliding_window_tensors, strides, and shift counts, which are provided in
        kwargs. baseline is expected to
        be broadcastable to match expanded_input.
        This method returns the ablated input tensor, which has the same
        dimensionality as expanded_input as well as the corresponding mask with
        either the same dimensionality as expanded_input or second dimension
        being 1. This mask contains 1s in locations which have been ablated (and
        thus counted towards ablations for that feature) and 0s otherwise.
        """
        input_mask = torch.stack(
            [
                self._occlusion_mask(
                    expanded_input,
                    j,
                    kwargs["sliding_window_tensors"],
                    kwargs["strides"],
                    kwargs["shift_counts"],
                    kwargs.get("is_above_threshold", None),
                )
                for j in range(start_feature, end_feature)
            ],
            dim=0,
        ).long()
        ablated_tensor = (
            expanded_input
            * (
                torch.ones(1, dtype=torch.long, device=expanded_input.device)
                - input_mask
            ).to(expanded_input.dtype)
        ) + (baseline * input_mask.to(expanded_input.dtype))

        return ablated_tensor, input_mask

    def _occlusion_mask(
        self,
        expanded_input: Tensor,
        ablated_feature_num: int,
        sliding_window_tsr: Tensor,
        strides: Union[int, Tuple[int, ...]],
        shift_counts: Tuple[int, ...],
        is_above_threshold: Tensor = None,
    ) -> Tensor:
        """
        This constructs the current occlusion mask, which is the appropriate
        shift of the sliding window tensor based on the ablated feature number.
        The feature number ranges between 0 and the product of the shift counts
        (# of times the sliding window should be shifted in each dimension).
        First, the ablated feature number is converted to the number of steps in
        each dimension from the origin, based on shift counts. This procedure
        is similar to a base conversion, with the position values equal to shift
        counts. The feature number is first taken modulo shift_counts[0] to
        get the number of shifts in the first dimension (each shift
        by shift_count[0]), and then divided by shift_count[0].
        The procedure is then continued for each element of shift_count. This
        computes the total shift in each direction for the sliding window.
        We then need to compute the padding required after the window in each
        dimension, which is equal to the total input dimension minus the sliding
        window dimension minus the (left) shift amount. We construct the
        array pad_values which contains the left and right pad values for each
        dimension, in reverse order of dimensions, starting from the last one.
        Once these padding values are computed, we pad the sliding window tensor
        of 1s with 0s appropriately, which is the corresponding mask,
        and the result will match the input shape.
        """
        if is_above_threshold is None:
            return super()._occlusion_mask(
                expanded_input=expanded_input,
                ablated_feature_num=ablated_feature_num,
                sliding_window_tsr=sliding_window_tsr,
                strides=strides,
                shift_counts=shift_counts,
            )

        # We first compute the hyper-rectangle on the non-temporal dims
        padded_tensor = super()._occlusion_mask(
            expanded_input=expanded_input[:, :, 0],
            ablated_feature_num=ablated_feature_num,
            sliding_window_tsr=torch.ones(sliding_window_tsr.shape[1:]),
            strides=strides[1:] if isinstance(strides, tuple) else strides,
            shift_counts=shift_counts[1:],
        ).to(expanded_input.device)

        # We get the current index and batch size
        bsz = expanded_input.shape[1]
        shift_count = shift_counts[0]
        stride = strides[0] if isinstance(strides, tuple) else strides
        current_index = (ablated_feature_num % shift_count) * stride

        # On the temporal dim, the hyper-rectangle is only applied on
        # non-zeros elements
        is_above = is_above_threshold.clone()
        for batch_idx in range(bsz):
            nonzero = is_above_threshold[batch_idx].nonzero()[:, 0]
            is_above[
                batch_idx,
                nonzero[
                    current_index : current_index + sliding_window_tsr.shape[0]
                ],
            ] = 0

        return is_above.unsqueeze(-1) * padded_tensor.unsqueeze(0)

    def _run_forward(
        self, forward_func: Callable, inputs: Any, **kwargs
    ) -> Tuple[Tuple[Tensor, ...], Tuple[Tuple[int]]]:
        attributions = self.attribution_method.attribute.__wrapped__(
            self.attribution_method, inputs, **kwargs
        )

        # Check if it needs to return convergence delta
        return_convergence_delta = (
            "return_convergence_delta" in kwargs
            and kwargs["return_convergence_delta"]
        )

        # If the method returns delta, we ignore it
        if self.is_delta_supported and return_convergence_delta:
            attributions, _ = attributions

        # Get attr shapes
        attributions_shape = tuple(tuple(attr.shape) for attr in attributions)

        return attributions, attributions_shape

    @staticmethod
    def _reshape_eval_diff(eval_diff: Tensor, shapes: tuple) -> Tensor:
        # For this method, we need to reshape eval_diff to the output shapes
        return eval_diff.reshape((len(eval_diff),) + shapes)
    
    @staticmethod
    def get_name():
        return 'OcclusionTSR' 
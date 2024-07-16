import numpy as np
import torch, gc

from captum.attr._utils.attribution import Attribution, GradientAttribution
from captum.attr._utils.common import (
    _format_input_baseline,
    _format_and_verify_sliding_window_shapes,
    _format_and_verify_strides,
)
from captum.log import log_usage
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

class WinTSR(Occlusion):
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
        self.use_weights = False  # We do not use weights for this method

    @property
    def multiplies_by_inputs(self):
        return self._multiply_by_inputs
    
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
    
    def generate_counterfactual(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        input_index: int,
        timestep: int,
    ) -> TensorOrTupleOfTensorsGeneric:
        n_features = inputs[input_index].shape[-1]
        for f in range(n_features):
            features = inputs[input_index][:, :timestep+1, f].reshape(-1)
            mu = torch.mean(features)
            std = max(torch.std(features), 1e-5)
            
            # this assumes data is normalized during the dataloader step
            return torch.normal(mu, 2*std)
    
    def _get_time_relevance(
        self, inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None
    ):  
        if type(inputs) == Tensor:
            inputs = (inputs,)
            baselines = (baselines,)
        
        # Compute the Time-Relevance Score (step 1). 
        # Shape ((N x O) x seq_len x features) if target is None
        # Shape (N x seq_len) if target is not None
        with torch.no_grad():
            # returns (attr, attr_shape). attr_shape is (N x O) x seq_len x features
            y_original = self.forward_func(
                *inputs, *additional_forward_args
            )
            # y_original = self.get_time_relevance_score(
            #     inputs, baselines, target, additional_forward_args
            # )
            # print(f'y_original {y_original}')
            num_outputs = y_original.shape[1]
            
            time_relevance_score = []
            
            for input_index in range(len(inputs)):
                batch_size, seq_len, n_features = inputs[input_index].shape
                
                if target is None:
                    score = torch.zeros(size=(
                        batch_size * num_outputs, 
                        seq_len
                    ), device=inputs[input_index].device)
                else:
                    score = torch.zeros(size=(
                        batch_size, seq_len
                    ), device=inputs[input_index].device)
                
                cloned = inputs[input_index].clone()
                for t in range(seq_len):
                    # mask last t timesteps
                    prev_val = cloned[:, t]
                    cloned[:, t] = baselines[input_index][:, t] # baselines[input_index][:, t]
                    
                    inputs_hat = []
                    for i in range(len(inputs)):
                        if i == input_index: inputs_hat.append(cloned)
                        else: inputs_hat.append(inputs[i])
                    
                    # returns (attr, attr_shape)
                    y_perturbed = self.forward_func(
                        *tuple(inputs_hat), *additional_forward_args
                    )
                    # y_perturbed = self.get_time_relevance_score(
                    #     tuple(inputs_hat), baselines, target, additional_forward_args
                    # )
                    
                    if target is None:
                        eval_diff = self._compute_metric(
                            y_original, y_perturbed
                        )
                        # eval_diff = torch.sum(torch.abs(y_original[0] - y_perturbed[0]), dim=-1)
                    else:
                        eval_diff = self._compute_metric(
                            y_original[:, target], y_perturbed[:, target]
                        )
                    # eval_diff = abs(y_original - y_perturbed)# 
                    
                    # iSab = abs(initial_eval - modified_eval)
                    # print(f'y_perturbed {y_perturbed}')
                    # print(f'Eval diff {eval_diff}')
                    eval_diff = torch.clip(eval_diff, -1e6, 1e6)
                    
                    score[:, t] = eval_diff.reshape(-1)
                    cloned[:, t] = prev_val
                    del y_perturbed, inputs_hat
                del cloned
                gc.collect()
                
                # batch_size, n_output, seq_len, n_features        
                # score[:, 1:] -= score[:, :-1] 
                # print(f'Score {score}')
                
                # reverse order along the time axis
                # score = score.flip(dims=(1,))
                
                time_relevance_score.append(abs(score))
            time_relevance_score = tuple(time_relevance_score)
        
        return time_relevance_score

    @log_usage()
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
        # time_relevance_score = self._get_time_relevance(
        #     inputs=inputs,
        #     baselines=baselines,
        #     target=target,
        #     additional_forward_args=additional_forward_args
        # )
             
        time_relevance_score = self.get_time_relevance_score(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=show_progress
        )  
        
        # y_original = self.forward_func(*inputs, *additional_forward_args)
        # batch_size, num_output, _ = y_original.shape
        
        # time_relevance_score = tuple(
        #     score.reshape((batch_size, num_output, -1)).mean(dim=1) 
        #     for score in time_relevance_score
        # )
        
        # Normalize if required
        if normalize:
            # normalize the last dimension
            time_relevance_score = tuple(
                normalize_scale(
                    tsr, dim=-1, norm_type="l1"
                ) for tsr in time_relevance_score
            )
            
        # print('tsr ', time_relevance_score) 
        # Get indexes where the Time-Relevance Score is
        # higher than the threshold
        is_above_threshold = tuple(
            score > torch.quantile(score, threshold, dim=-1, keepdim=True) for score in time_relevance_score
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
        # print('frs shape ', [tsr.shape for tsr in features_relevance_score])
        # Reshape attributions before merge
        time_relevance_score = tuple(
            tsr.reshape(f_imp.shape[:2] + (1,) * len(f_imp.shape[2:]))
            for f_imp, tsr in zip(features_relevance_score, time_relevance_score)
        )
        is_above_threshold = tuple(
            is_above.reshape(f_imp.shape[:2] + (1,) * len(f_imp.shape[2:]))
            for f_imp, is_above in zip(features_relevance_score, is_above_threshold)
        )

        # Merge attributions:
        # Time-Relevance Score x Feature-Relevance Score x is above threshold
        attributions = tuple(
            (tsr * frs) * is_above.float()
            for tsr, frs, is_above in zip(
                time_relevance_score,
                features_relevance_score,
                is_above_threshold,
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
        # print('Expanded input ', expanded_input.shape, input_mask.shape, f' Start {start_feature}, end {end_feature}')
        
        # if input_mask.shape[1] > expanded_input.shape[1]:
        #     reduced_input_mask = input_mask.reshape((input_mask.shape[0], -1, *expanded_input.shape[1:]))
        #     reduced_input_mask = torch.round(torch.mean(reduced_input_mask, dim=1, dtype=torch.float), dtype=torch.long)
            
        #     ablated_tensor = (
        #         expanded_input
        #             * (
        #                 torch.ones(1, dtype=torch.long, device=expanded_input.device)
        #                 # - reduced_input_mask
        #                 # - input_mask[:, :expanded_input.shape[1]]
        #                 - reduced_input_mask
        #             ).to(expanded_input.dtype)
        #         # ) + (baseline * reduced_input_mask.to(expanded_input.dtype))
        #         # ) + (baseline * input_mask[:, :expanded_input.shape[1]].to(expanded_input.dtype))
        #         ) + (baseline * reduced_input_mask.to(expanded_input.dtype))
        # else:
        ablated_tensor = (
            expanded_input
            * (
                torch.ones(1, dtype=torch.long, device=expanded_input.device)
                # - reduced_input_mask
                - input_mask[:, :expanded_input.shape[1]]
                # - input_mask
            ).to(expanded_input.dtype)
        # ) + (baseline * reduced_input_mask.to(expanded_input.dtype))
        ) + (baseline * input_mask[:, :expanded_input.shape[1]].to(expanded_input.dtype))
        # ) + (baseline * input_mask.to(expanded_input.dtype))

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
        return 'WinTSR'
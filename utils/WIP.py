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
from utils.tools import normalize_scale, reshape_over_output_horizon
from tint.attr.occlusion import FeatureAblation, Occlusion

class Decoupled(Occlusion):
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

    @log_usage()
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
        return 'WinTSR'
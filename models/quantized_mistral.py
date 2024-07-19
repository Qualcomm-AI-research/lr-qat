# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
import math
import warnings
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from accelerate import logging
from datasets import tqdm
from torch import nn
from transformers import Cache, MistralForCausalLM
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralMLP,
    MistralModel,
    MistralRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)

from models.llm_quant_states import LLM_FP_heads
from quantization.autoquant_utils import quantize_model
from quantization.base_quantized_classes import QuantizedActivation
from quantization.base_quantized_model import QuantizedModel
from utils.quant_click_options import QuantSetup

logger = logging.get_logger(__name__)


class QuantizedMistralForCausalLM(QuantizedModel):
    forward = MistralForCausalLM.forward
    prepare_inputs_for_generation = MistralForCausalLM.prepare_inputs_for_generation
    _reorder_cache = MistralForCausalLM._reorder_cache

    def __init__(self, org_model: MistralForCausalLM, quant_setup, **quant_params):
        super().__init__()
        self.config = org_model.config
        self.model = QuantizedMistralModel(org_model.model, quant_setup, **quant_params)
        self.vocab_size = org_model.vocab_size

        if quant_setup in LLM_FP_heads:
            self.lm_head = org_model.lm_head
        elif quant_setup == QuantSetup.all:
            self.lm_head = quantize_model(org_model.lm_head, **quant_params)
        else:
            raise ValueError(f"Quantization setup '{quant_setup}' not supported.")

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # # # ADAROUND MIXIN IMPLEMENTATION # # #
    def get_attention_blocks(self):
        return self.model.layers

    def get_head(self):
        return self.lm_head

    # The method is needed by LM Eval
    def tie_weights(self):
        pass


class QuantizedMistralModel(QuantizedModel):
    """The bare Mistral Model outputting raw hidden-states without any specific head on top.
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    forward = MistralModel.forward

    def __init__(self, org_model: MistralModel, quant_setup, **quant_params):
        super().__init__()
        self.config = org_model.config
        self.padding_idx = org_model.padding_idx
        self.vocab_size = org_model.vocab_size

        layers = []
        for layer in tqdm(org_model.layers):
            layer = QuantizedMistralDecoderLayer(layer, quant_setup, **quant_params)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        self._attn_implementation = org_model._attn_implementation

        # LayerNorm
        if quant_setup == QuantSetup.FP_head_embd_norm:
            self.norm = org_model.norm
            self.embed_tokens = org_model.embed_tokens
        elif quant_setup == QuantSetup.FP_head_embd:
            self.norm = QuantizedMistralRMSNorm(self.norm, **quant_params)
            self.embed_tokens = org_model.embed_tokens
        elif quant_setup == QuantSetup.FP_head or quant_setup == QuantSetup.all:
            self.norm = QuantizedMistralRMSNorm(self.norm, **quant_params)
            self.embed_tokens = quantize_model(org_model.embed_tokens, **quant_params)
        else:
            raise ValueError(f"Quantization setup '{quant_setup}' not supported.")
        self.gradient_checkpointing = org_model.gradient_checkpointing

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


class QuantizedMistralDecoderLayer(QuantizedModel):

    def __init__(self, org_model: MistralDecoderLayer, quant_setup, **quant_config):
        super().__init__()
        self.hidden_size = org_model.hidden_size

        try:
            quantized_self_attn_class = QUANTIZED_MISTRAL_ATTENTION_CLASSES[
                org_model.self_attn.__class__
            ]
        except KeyError as e:
            raise TypeError(
                f"Attention class {org_model.self_attn.__class__} is not supported by QuantizedMistral.\n"
                f"Use one of the following: {list(QUANTIZED_MISTRAL_ATTENTION_CLASSES.keys())}"
            )
        self.self_attn = quantized_self_attn_class(org_model.self_attn, **quant_config)

        self.mlp = QuantizedMistralMLP(org_model.mlp, **quant_config)

        # LayerNorm
        if quant_setup == QuantSetup.FP_head_embd_norm:
            self.input_layernorm = org_model.input_layernorm
            self.post_attention_layernorm = org_model.post_attention_layernorm
        else:
            self.input_layernorm = QuantizedMistralRMSNorm(
                org_model.input_layernorm, **quant_config
            )
            self.post_attention_layernorm = QuantizedMistralRMSNorm(
                org_model.post_attention_layernorm, **quant_config
            )

        # Activation Quantizers:
        self.residual_quantizer_1 = QuantizedActivation(**quant_config)
        self.residual_quantizer_2 = QuantizedActivation(**quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        hidden_states = self.residual_quantizer_1(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = self.residual_quantizer_2(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class QuantizedMistralAttention(QuantizedModel):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, org_model: MistralAttention, **quant_params):
        super().__init__()

        # copy attributes
        self.config = org_model.config
        self.layer_idx = org_model.layer_idx
        self.hidden_size = org_model.config.hidden_size
        self.num_heads = org_model.config.num_attention_heads
        self.head_dim = org_model.head_dim
        self.num_key_value_heads = org_model.num_key_value_heads
        self.num_key_value_groups = org_model.num_key_value_groups
        self.max_position_embeddings = org_model.max_position_embeddings
        self.rope_theta = org_model.rope_theta
        self.is_causal = org_model.is_causal
        self.attention_dropout = org_model.attention_dropout

        # copy not-quantizable modules
        self.rotary_emb = org_model.rotary_emb

        # quantized modules
        self.q_proj = quantize_model(org_model.q_proj, **quant_params)
        self.k_proj = quantize_model(org_model.k_proj, **quant_params)
        self.v_proj = quantize_model(org_model.v_proj, **quant_params)
        self.o_proj = quantize_model(org_model.o_proj, **quant_params)

        # activation quantizers
        self.attn_key_act_quantizer = QuantizedActivation(**quant_params)
        self.attn_query_act_quantizer = QuantizedActivation(**quant_params)
        self.attn_softmax_inp_quantizer = QuantizedActivation(**quant_params)
        self.attn_probs_act_quantizer = QuantizedActivation(**quant_params)
        self.attn_act_re_quantizer = QuantizedActivation(**quant_params)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        query_states = self.attn_query_act_quantizer(query_states)
        key_states = self.attn_key_act_quantizer(key_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )
        attn_weights = self.attn_softmax_inp_quantizer(attn_weights)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

            # Avoid numeric issues (inf) with certain dtypes like fp16
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device),
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = self.attn_probs_act_quantizer(attn_weights)

        # it is important to convert the FP32 softmax output to FP16 ONLY AFTER QUANTIZATION,
        # otherwise for spiked softmax (1.0s fp16 value) the division by a small scale in the quantizer will
        # output +INF (this does not happen with fp32), and the clamp will not cut to 1.0 -> NaNs appear.
        attn_weights = attn_weights.to(query_states.dtype)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = self.attn_act_re_quantizer(attn_output)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class QuantizedMistralMLP(QuantizedModel):

    def __init__(self, org_model: MistralMLP, **quant_params):
        super().__init__()
        self.config = org_model.config
        self.hidden_size = org_model.hidden_size
        self.intermediate_size = org_model.intermediate_size
        self.gate_proj_act = quantize_model(
            nn.Sequential(org_model.gate_proj, org_model.act_fn), **quant_params
        )[0]
        self.up_proj = quantize_model(org_model.up_proj, **quant_params)
        self.down_proj = quantize_model(org_model.down_proj, **quant_params)

        # activation quantizers
        self.up_mul_gate_requantizer = QuantizedActivation(**quant_params)

    def forward(self, x):
        out_gate = self.gate_proj_act(x)
        out_up_proj = self.up_proj(x)
        out_mul = self.up_mul_gate_requantizer(out_gate * out_up_proj)
        return self.down_proj(out_mul)


class QuantizedMistralRMSNorm(QuantizedModel, MistralRMSNorm):
    def __init__(self, org_model: MistralRMSNorm, **quant_params):
        raise NotImplementedError()


QUANTIZED_MISTRAL_ATTENTION_CLASSES = {
    MistralAttention: QuantizedMistralAttention,
}

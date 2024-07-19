# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import Cache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.llama.modeling_llama import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    LlamaAttention,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    apply_rotary_pos_emb,
)

from models.llm_quant_states import LLM_FP_heads
from quantization.autoquant_utils import quantize_model
from quantization.base_quantized_classes import FP32Acts, QuantizedActivation
from quantization.base_quantized_model import QuantizedModel
from utils.quant_click_options import QuantSetup


class QuantizedLlamaForCausalLM(QuantizedModel):
    def __init__(self, org_model: LlamaForCausalLM, quant_setup=None, **quant_params):
        super().__init__()

        # copy attributes
        self.config = org_model.config

        self.model = QuantizedLlamaModel(org_model.model, quant_setup, **quant_params)

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
    def get_embeddings(self):
        return (self.model.embed_tokens,)

    def get_attention_blocks(self):
        return self.model.layers

    def apply_block(self, block, hidden_states, **kw):
        block_out = block(hidden_states, **kw)
        hidden_states = block_out[0]
        return hidden_states

    def get_head(self):
        return self.lm_head

    # # # ADAROUND MIXIN IMPLEMENTATION -- END # # #

    # The method is needed by LM Eval
    def tie_weights(self):
        pass

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM
        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


class QuantizedLlamaModel(QuantizedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    Args:
        config: LlamaConfig
    """

    def __init__(self, org_model: LlamaModel, quant_setup, **quant_params):
        super().__init__()

        self.config = org_model.config
        self.padding_idx = org_model.padding_idx
        self.vocab_size = org_model.vocab_size
        self.embed_tokens = org_model.embed_tokens

        # LayerNorm
        if quant_setup == QuantSetup.FP_head_embd_norm:
            self.norm = org_model.norm
        else:
            self.norm = quantize_model(org_model.norm, **quant_params)

        # activation quantizers
        self.embed_act_quantizer = QuantizedActivation(**quant_params)

        self.gradient_checkpointing = org_model.gradient_checkpointing

        self.layers = nn.ModuleList(
            [
                QuantizedLlamaDecoderLayer(org_block, quant_setup, **quant_params)
                for org_block in org_model.layers
            ]
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        hidden_states = self.embed_act_quantizer(hidden_states)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class QuantizedLlamaMLP(QuantizedModel):
    def __init__(self, org_model: LlamaMLP, **quant_params):
        super().__init__()
        self.gate_proj_act = quantize_model(
            nn.Sequential(org_model.gate_proj, org_model.act_fn), **quant_params
        )[0]
        self.down_proj = quantize_model(org_model.down_proj, **quant_params)
        self.up_proj = quantize_model(org_model.up_proj, **quant_params)
        self.mlp_mul_act_quantizer = QuantizedActivation(**quant_params)

    def forward(self, x):
        out_gate = self.gate_proj_act(x)
        out_up_proj = self.up_proj(x)
        out_mul = self.mlp_mul_act_quantizer(out_gate * out_up_proj)
        return self.down_proj(out_mul)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class QuantizedLlamaAttention(QuantizedModel):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, org_model: LlamaAttention, **quant_params):
        super().__init__()
        self.config = org_model.config
        self.layer_idx = org_model.layer_idx

        self.attention_dropout = org_model.attention_dropout
        self.hidden_size = org_model.hidden_size
        self.num_heads = org_model.num_heads
        self.head_dim = org_model.head_dim
        self.num_key_value_heads = org_model.num_key_value_heads
        self.num_key_value_groups = org_model.num_key_value_groups
        self.max_position_embeddings = org_model.max_position_embeddings
        self.rope_theta = org_model.rope_theta
        self.is_causal = org_model.is_causal

        self.q_proj = quantize_model(org_model.q_proj, **quant_params)
        self.k_proj = quantize_model(org_model.k_proj, **quant_params)
        self.v_proj = quantize_model(org_model.v_proj, **quant_params)
        self.o_proj = quantize_model(org_model.o_proj, **quant_params)

        self.rotary_emb = org_model.rotary_emb

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
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            raise NotImplementedError(
                "We never tested and do not support pretraining_tp > 1 with QuantizedLlama"
            )
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
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

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = self.attn_query_act_quantizer(query_states)
        key_states = self.attn_key_act_quantizer(key_states)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )
        attn_weights = self.attn_softmax_inp_quantizer(attn_weights)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights_softmax_output = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        )
        # it is important to cast the FP32 softmax output to FP16 only after quantization, otherwise NaNs appear here
        attn_weights = self.attn_probs_act_quantizer(attn_weights_softmax_output).to(
            query_states.dtype
        )
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

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            raise NotImplementedError(
                "We never tested and do not support pretraining_tp > 1 with QuantizedLlama"
            )
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class QuantizedLlamaDecoderLayer(QuantizedModel):
    def __init__(self, org_model, quant_setup, **quant_params):
        super().__init__()
        self.hidden_size = org_model.hidden_size

        self.self_attn = QuantizedLlamaAttention(org_model.self_attn, **quant_params)
        self.mlp = QuantizedLlamaMLP(org_model.mlp, **quant_params)

        # LayerNorm
        if quant_setup == QuantSetup.FP_head_embd_norm:
            self.input_layernorm = org_model.input_layernorm
            self.post_attention_layernorm = org_model.post_attention_layernorm
            self.attn_ln_act_quantizer = FP32Acts()
            self.mlp_ln_act_quantizer = FP32Acts()
        else:
            self.input_layernorm = quantize_model(org_model.input_layernorm, **quant_params)
            self.post_attention_layernorm = quantize_model(
                org_model.post_attention_layernorm, **quant_params
            )
            self.attn_ln_act_quantizer = QuantizedActivation(**quant_params)
            self.mlp_ln_act_quantizer = QuantizedActivation(**quant_params)

        # activation quantizers
        self.attn_res_act_quantizer = QuantizedActivation(**quant_params)
        self.mlp_res_act_quantizer = QuantizedActivation(**quant_params)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
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
        hidden_states = self.attn_ln_act_quantizer(hidden_states)

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
        hidden_states = self.attn_res_act_quantizer(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp_ln_act_quantizer(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.mlp_res_act_quantizer(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

import math 
import torch 
from torch import nn 
import torch.nn.functional as F

from transformers.activations import ACT2FN 
from transformers.modeling_outputs import CausalLMOutputWithPast 
from transformers.modeling_utils import PreTrainedModel 


class LlamaRoPE(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim 
        self.max_position_embeddings = max_position_embeddings
        self.base = base 
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).to(device) / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device, self.inv_freq.dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len 
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=dtype)

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def _rotate_half(self, x):
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len, position_ids):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, k.device, k.dtype)

        cos = self.cos_cached[:, :, :seq_len, ...].to(dtype=k.dtype)
        sin = self.sin_cached[:, :, :seq_len, ...].to(dtype=k.dtype)
        cos = cos.squeeze(1).squeeze(0)[position_ids].unsqueeze(1)
        sin = sin.squeeze(1).squeeze(0)[position_ids].unsqueeze(1)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed 


class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size 
        self.num_heads = config.num_attention_heads 
        self.head_dim = self.hidden_size // self.num_heads 
        self.num_kv_heads = config.num_kv_heads 
        self.num_kv_groups = self.num_heads // self.num_kv_heads 
        self.max_position_embeddings = config.max_position_embeddings 
        self.pretraining_tp = config.pretraining_tp
        self.use_cache = config.use_cache

        assert self.head_dim * self.num_heads == self.hidden_size

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = LlamaRoPE(self.head_dim, max_position_embeddings=self.max_position_embeddings)
    
    def _repeat_kv(self, hidden_states, n_rep):
        if n_rep == 1: return hidden_states

        batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape 
        hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
        return hidden_states.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        batch_size, seq_len, _ = hidden_states.shape

        if self.pretraining_tp > 1:
            query_slicing = (self.num_heads * self.head_dim) // self.pretraining_tp
            key_value_slicing = (self.num_kv_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split(query_slicing, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2) 

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None: 
            kv_seq_len += past_key_value[0].shape[-2] 

        query_states, key_states = self.rotary_emb(query_states, key_states, kv_seq_len, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if self.use_cache else None

        key_states = self._repeat_kv(key_states, self.num_kv_groups)
        value_states = self._repeat_kv(value_states, self.num_kv_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        assert attn_weights.shape == (batch_size, self.num_heads, seq_len, kv_seq_len)

        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, 1, seq_len, kv_seq_len)
            attn_weights = attn_weights + attention_mask

        attn_weights = attn_weights.softmax(dim=-1).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        assert attn_output.shape == (batch_size, self.num_heads, seq_len, self.head_dim)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size 
        self.intermediate_size = config.intermediate_size 
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.pretraining_tp = config.pretraining_tp

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp 
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj 
    

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps 

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype 
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size 
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config=config)
        self.norm1 = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,
                hidden_states, 
                attention_mask=None,
                position_ids=None, 
                past_key_value=None):
        residual = hidden_states 
        hidden_states = self.norm1(hidden_states) 
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask, 
            position_ids=position_ids,
            past_key_value=past_key_value
        )
        hidden_states = residual + hidden_states 

        residual = hidden_states 
        hidden_states = self.norm2(hidden_states) 
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states 

        return hidden_states, present_key_value


class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_cache = config.use_cache
        
    def _expand_mask(self, mask, dtype, seq_len):
        batch_size, src_len = mask.shape
        expanded_mask = mask[:, None, None, :].expand(batch_size, 1, seq_len, src_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask 

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    
    def _make_causal_mask(self, inputs_embeds, past_kv_len=0):
        batch_size, seq_len, _ = inputs_embeds.shape 
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device 

        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.shape[-1], device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.shape[-1], 1), 0)
        mask = mask.to(dtype)

        if past_kv_len > 0:
            mask = torch.cat([torch.zeros(seq_len, past_kv_len, dtype=dtype, device=device), mask], dim=-1)

        return mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len + past_kv_len)
    
    def forward(self, input_ids, attention_mask=None, past_key_values=None):
        batch_size, seq_len = input_ids.shape

        past_kv_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]

        device = input_ids.device
        position_ids = torch.arange(past_kv_len, past_kv_len + seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_len)

        inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, past_kv_len + seq_len), 
                dtype=torch.bool, 
                device=inputs_embeds.device
            )
        attention_mask = self._expand_mask(attention_mask, inputs_embeds.dtype, seq_len)

        if seq_len > 1:
            causal_attention_mask = self._make_causal_mask(inputs_embeds, past_kv_len)
            attention_mask += causal_attention_mask

        hidden_states = inputs_embeds 
        present_key_values = () if self.use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None 

            hidden_states, present_key_value = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value
            )

            if self.use_cache: present_key_values += (present_key_value,)

        hidden_states = self.norm(hidden_states)
        
        return hidden_states, present_key_values


class LlamaCausalModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def _init_weights(self, module):
        std = self.config.initializer_range 
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self,
                input_ids=None,
                labels=None,
                attention_mask=None,
                past_key_values=None,
                use_cache=None,
                return_dict=None,
                output_attentions=None,
                output_hidden_states=None):
        hidden_states, present_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        if self.config.pretraining_tp > 1:
            split_num = (self.config.vocab_size + 1) // self.config.pretraining_tp
            lm_head_slices = self.lm_head.weight.split(split_num, dim=0)
            logtis = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logtis, dim=-1)
        else:
            logits = self.lm_head(hidden_states)

        loss = None 
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=present_key_values, 
            hidden_states=hidden_states,
        )
    
    def prepare_inputs_for_generation(self, 
                                      input_ids, 
                                      attention_mask=None, 
                                      past_key_values=None,
                                      use_cache=None):
        if past_key_values: 
            input_ids = input_ids[:, -1:]

        next_model_inputs = {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'past_key_values': past_key_values,
            'use_cache': use_cache
        } 
        return next_model_inputs 
    
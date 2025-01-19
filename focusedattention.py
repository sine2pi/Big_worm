
import base64
import gzip
import math
import os
import functools
import warnings
import numpy as np
import torch
import transformers
import aiohttp
import torch.nn.functional as F
from torch import Tensor, amp, optim, nn
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from threading import Thread
from typing import Dict, Optional, Tuple, Union, List, Any
from transformers.modeling_utils import PreTrainedModel
from dataclasses import dataclass
from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments, PretrainedConfig, TrainerCallback,
    WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizerFast
)

import evaluate
from evaluate import module
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import KFold, train_test_split
from datasets import load_dataset, Dataset, concatenate_datasets, IterableDatasetDict, Audio, DatasetDict
from torch.nn.functional import scaled_dot_product_attention

transformers.utils.logging.set_verbosity_error()
warnings.filterwarnings(action="ignore")
warnings.warn = lambda *args, **kwargs: None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch_dtype = torch.float32
torch.set_default_dtype(dtype)


class CustomEmbedding(nn.Module):
    def __init__(self, initial_value, learnable=True):
        super(CustomEmbedding, self).__init__()
        if learnable:
            self.value = nn.Parameter(torch.tensor(initial_value))
        else:
            self.register_buffer('value', torch.tensor(initial_value))
    def forward(self):
        return self.value

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)

class CombinedRotaryEmbedding(nn.Module):
    def __init__(self, base, n_state, n_head, num_rotations=None, checkpointing=False, loss=None):
        super().__init__()
        self.base = base
        self.n_state = n_state
        self.n_head = n_head
        self.h_dim = n_state // n_head
        self.num_rotations = self.h_dim // 2 if num_rotations is None else num_rotations
        self.checkpointing = checkpointing
        self.loss = loss
        
        self.thetas = nn.Parameter(torch.zeros(self.num_rotations)) 
        self.rotation_pairs = nn.Parameter(data=torch.rand(self.num_rotations, 2) * self.h_dim)
        self.theta_scale = nn.Parameter(data=torch.ones(1))
        self.rotation_matrix = nn.Parameter(data=torch.eye(n=self.h_dim))
        self.inv_freq = nn.Parameter(data=1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim)))
        self.num_rotations_scale = nn.Parameter(data=torch.ones(1))

    def givens_rotation_matrix(self, n_state, i, j, theta):
        G = torch.eye(n_state, device=theta.device)
        G[i, i] = math.cos(theta)
        G[i, j] = -math.sin(theta)
        G[j, i] = math.sin(theta)
        G[j, j] = math.cos(theta)
        return G
      
    def update_base(self, new_base, loss=None):
        if loss is not None and loss != self.loss: 
            self.loss = loss 
            if new_base != self.base:
                self.base = float(new_base)
                inv_freq = 1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim)) 
                self.register_buffer('inv_freq', inv_freq) 

    def reset_parameters(self):
        nn.init.orthogonal_(tensor=self.rotation_matrix)
        nn.init.zeros_(tensor=self.thetas)

    def forward(self, x, new_base=None, loss=None): 
        self.update_base(new_base, loss) 
        return self._forward(x)

    def _forward(self, x):
        if x.dim() not in [3, 4]:
            raise ValueError(f"Expected input tensor to be 3D or 4D, but got {x.dim()}D")

        batch_size, seq_len, *rest = x.size()

        if x.dim() == 3:
            n_state = rest[0]
            if n_state != self.n_head * self.h_dim:
                raise ValueError(f"Expected n_state ({n_state}) to be compatible with n_head ({self.n_head}) * h_dim ({self.h_dim} = {self.n_head * self.h_dim})") #added informative message
        else: 
            n_head, h_dim = rest
            if n_head != self.n_head or h_dim != self.h_dim:
                raise ValueError(f"For 4D input, expected n_head {self.n_head} and h_dim {self.h_dim}, but got n_head {n_head} and h_dim {h_dim}")

        x = x.view(batch_size, seq_len, self.n_head, self.h_dim) 
        x = x.reshape(-1, self.h_dim)
        adjusted_num_rotations = int(torch.round(self.num_rotations * self.num_rotations_scale))

        for k in range(adjusted_num_rotations):
            i, j = self.rotation_pairs[k].long()
            theta = self.thetas[k] * self.theta_scale
            G = self.givens_rotation_matrix(n_state=self.h_dim, i=i, j=j, theta=theta)
            x = torch.matmul(input=x, other=G)

        x = torch.matmul(input=x, other=self.rotation_matrix)
        x = x.view(batch_size, seq_len, self.n_head, self.h_dim)

        sinusoid_inp = torch.einsum('i, j -> i j', torch.arange(end=seq_len, device=x.device), self.inv_freq.to(device=x.device))
        sin = sinusoid_inp.sin()[None, :, None, :]
        cos = sinusoid_inp.cos()[None, :, None, :]

        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.cat(tensors=[x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        x = x.view(batch_size, seq_len, self.n_state)
        return x

class LearnedSinusoidalEmbeddings(nn.Module):
    def __init__(self, n_ctx, n_state, checkpointing=False):
        super().__init__()
        self.n_ctx = n_ctx
        self.n_state = n_state
        self.checkpointing = checkpointing

        position = torch.arange(start=0, end=self.n_ctx, dtype=torch.float).unsqueeze(dim=1)
        div_term = torch.exp(input=torch.arange(start=0, end=self.n_state, step=2).float() * -(math.log(10000.0) / self.n_state))
        features = torch.zeros(self.n_ctx, self.n_state)
        features[:, 0::2] = torch.sin(input=position * div_term)
        features[:, 1::2] = torch.cos(input=position * div_term)
        self.register_buffer('my_big_toe', tensor=features)
        self.positional_embeddings = nn.Parameter(self.my_big_toe.clone())

    def forward(self, positions):
        if self.checkpointing:
            position_embeddings = checkpoint(lambda x: self.positional_embeddings[x], positions)
        else:
            position_embeddings = self.positional_embeddings[positions]
        position_embeddings = torch.nn.functional.normalize(input=position_embeddings, p=2, dim=-1)
        return position_embeddings

class MultiheadAttention(nn.Module):
    def __init__(self, base, n_state, n_head, max_rel_dist, checkpointing=False, loss=None):
        super().__init__()
        self.base = base
        self.n_state = n_state
        self.n_head = n_head
        self.max_rel_dist = max_rel_dist
        self.checkpointing = checkpointing
        self.loss = loss

        assert self.n_state % self.n_head == 0, "n_state must be divisible by n_head"
        self.h_dim = self.n_state // self.n_head
        assert self.h_dim % 2 == 0, "Head dimension must be even for rotary embeddings"

        self.query = nn.Linear(self.n_state, self.n_state)
        self.key = nn.Linear(self.n_state, self.n_state, bias=False)
        self.value = nn.Linear(self.n_state, self.n_state)
        self.out = nn.Linear(self.n_state, self.n_state)

        self.combined_rotary = CombinedRotaryEmbedding(
            base=self.base,
            n_head=self.n_head,
            n_state=self.n_state,
            checkpointing=self.checkpointing
        )
        self.kv_cache = {}
        
        self.positional_scaling = nn.Parameter(torch.ones(1))
        self.rel_pos_bias = nn.Parameter(torch.zeros((2 * int(self.max_rel_dist) - 1, self.n_head))).to(device)

    def update_base(self, new_base, loss=None):
        if loss is not None and loss != self.loss: 
            self.loss = loss 
            if new_base != self.base:
                self.base = float(new_base)
                inv_freq = 1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim)) 
                self.register_buffer('inv_freq', inv_freq) 
                self.combined_rotary.update_base(self.base)

    def update_dist(self, new_dist, loss=None):
        if loss is not None and loss != self.loss: 
            self.loss = loss
            if new_dist != self.max_rel_dist:
                self.max_rel_dist = int(new_dist)
                self.rel_pos_bias = nn.Parameter(torch.zeros((2 * int(self.max_rel_dist) - 1, self.n_head), device=self.rel_pos_bias.device))
        
    def forward(self, x: Tensor, xa: Optional[Tensor] = None, 
                mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None, new_dist=None, new_base=None, loss=None):
        
        self.update_base(new_base, loss) 
        self.update_dist(new_dist, loss)

        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:

            k = kv_cache[self.key]
            v = kv_cache[self.value]

        q = self.combined_rotary(q) * self.positional_scaling
        k = self.combined_rotary(k) * self.positional_scaling

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape

        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        a = scaled_dot_product_attention(q, k, v, is_causal=mask is not None and n_ctx > 1)
        out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
        qk = None
        qk = qk if qk is not None else 0

        seq_len_q = q.size(2)
        seq_len_k = k.size(2)

        positions = torch.arange(end=seq_len_q, device=q.device).unsqueeze(dim=1) - torch.arange(end=seq_len_k, device=q.device).unsqueeze(dim=0)
        positions = positions.clamp(min=-self.max_rel_dist + 1, max=self.max_rel_dist - 1) + self.max_rel_dist - 1
        rel_bias = self.rel_pos_bias[positions]
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)
  
        qk = qk + rel_bias
        qk = qk.float()
        return out, qk
    
class AdaptiveSpanAttention(nn.Module):
    def __init__(self, base, n_state, n_head, max_rel_dist, window_size, max_span, checkpointing=False, temperature_scale_factor=0.1):
        super().__init__()
        self.base = base
        self.n_state = n_state
        self.n_head = n_head
        self.max_rel_dist = max_rel_dist
        self.window_size = window_size
        self.max_span = max_span
        self.checkpointing = checkpointing
        self.temperature_scale_factor = temperature_scale_factor
        self.multihead_attn = MultiheadAttention(self.base, self.n_state, self.n_head, self.max_rel_dist, self.checkpointing)
        self.span_scale = nn.Parameter(torch.tensor(1.0))

        
    def forward(self, query, key, value, span_scale):
        span_length = int(self.max_span * span_scale.mean().item())
        span_length = min(span_length, query.shape[1], key.shape[1], value.shape[1])

        effective_span = min(span_length, self.max_rel_dist)
        query_span = query[:, :effective_span, :]
        key_span = key[:, :effective_span, :]
        value_span = value[:, :effective_span, :]

        attn_output, attn_weights = self.multihead_attn(query_span, key_span, value_span)
        temperature = 1.0 - self.temperature_scale_factor * span_scale  

        n_batch, n_ctx, n_state = query_span.shape
        scale = (n_state // self.multihead_attn.n_head) ** -0.25

        q = query_span.view(*query_span.shape[:2], self.multihead_attn.n_head, -1).permute(0, 2, 1, 3)
        k = key_span.view(*key_span.shape[:2], self.multihead_attn.n_head, -1).permute(0, 2, 1, 3)
        v = value_span.view(*value_span.shape[:2], self.multihead_attn.n_head, -1).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = torch.softmax((attn_scores / temperature) * scale, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).flatten(start_dim=2)
        attn_weights = attn_weights * (1.0 / span_scale)     
       
        attn_output = torch.bmm(attn_weights.view(-1, *attn_weights.shape[2:]), v.view(-1, *v.shape[2:]))
        attn_output = attn_output.view(query.size(0), query.size(1), -1)
        attn_output = attn_output.permute(0, 2, 1).contiguous().view(query.size(0), -1, query.size(2))    
        return attn_output, attn_weights

class SpanPredictor(nn.Module):
    def __init__(self, n_state):
        super().__init__()
        self.linear = nn.Linear(n_state, 1)

    def forward(self, global_attention_output):
        scale = torch.sigmoid(self.linear(global_attention_output))
        return scale

class HybridAttention(nn.Module):
    def __init__(self, base, n_state, n_head, max_rel_dist, window_size, max_span=128, sliding_window=16):
        super().__init__()
        self.sliding_window = sliding_window
        self.base = base
        self.n_state = n_state
        self.n_head = n_head
        self.max_rel_dist = max_rel_dist
        self.window_size = window_size
        self.max_span = max_span
        self.sliding_window = sliding_window
        self.span_predictor = SpanPredictor(n_state)
        self.local_max_rel_dist = max_rel_dist  
        self.global_max_rel_dist = max_rel_dist
        self.local_attn = AdaptiveSpanAttention(self.base, self.n_state, self.n_head, self.local_max_rel_dist, self.window_size, self.max_span)
        self.global_attn = MultiheadAttention(self.base, self.n_state, self.n_head, self.global_max_rel_dist, checkpointing=False)
        self.ln_local = nn.LayerNorm(self.n_state)
        self.ln_global = nn.LayerNorm(self.n_state)
        self.projection = nn.Linear(2 * self.n_state, self.n_state)

    def forward(self, x, loss=None): 
 
        x_local = self.ln_local(x)
        x_global = self.ln_global(x)

        global_out, _ = self.global_attn(x_global, x_global, x_global)
        span_scale = self.span_predictor(global_out.mean(dim=1)) 

        window_size = max(1, int(self.sliding_window * span_scale.mean().item()))
        span_length = max(1, int(self.max_span * span_scale.mean().item()))

        effective_max_rel_dist = min(self.max_rel_dist, x_local.size(1))
        local_effective_max_rel_dist = min(self.local_max_rel_dist, span_length, window_size)
        global_effective_max_rel_dist = effective_max_rel_dist
        self.local_attn.max_rel_dist = local_effective_max_rel_dist
        self.global_attn.max_rel_dist = global_effective_max_rel_dist

        local_out = self.sliding_window_attention(x_local, window_size, span_length, span_scale)
        combined = torch.cat([local_out.permute(1, 0, 2), global_out.permute(1, 0, 2)], dim=-1)
        combined_out = self.projection(combined)
        return combined_out

    def sliding_window_attention(self, x, window_size, span_length, span_scale):
        batch_size, seq_len, n_state = x.size()
        output = torch.zeros_like(x, device=x.device)

        for i in range(0, seq_len, window_size):
            end = min(i + window_size, seq_len)
            query = x[:, i:end, :]
            start = max(0, i - span_length + window_size)
            key = x[:, start:i + span_length, :]
            value = x[:, start:i + span_length, :]

            attn_output, _ = self.local_attn(query, key, value, span_scale)
            output[:, i:end, :] = attn_output
        return output

class ResidualAttentionBlock(nn.Module):
    def __init__(self, base, n_state, n_head, max_rel_dist, hybrid_attention=False, loss=None):
        super().__init__()
        self.base = base
        self.n_state = n_state
        self.n_head = n_head
        self.max_rel_dist = max_rel_dist
        self.hybrid_attention = hybrid_attention
        self.loss = loss

        if self.hybrid_attention:
            self.attn = HybridAttention(self.base, self.n_state, self.n_head, self.max_rel_dist, self.loss)
        else:
            self.attn = MultiheadAttention(self.base, self.n_state, self.n_head, self.max_rel_dist)
        self.attn_ln = LayerNorm(self.n_state)

        n_mlp = self.n_state * 4
        self.mlp = nn.Sequential(nn.Linear(self.n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, self.n_state))
        self.mlp_ln = LayerNorm(self.n_state)

    def forward(self, x, mask=None, loss=None, kv_cache=None):
        x = self._attn_forward(x, mask, loss, kv_cache)
        x = self._mlp_forward(x)
        return x

    def _attn_forward(self, x, mask=None, loss=None, kv_cache=None):
        residual = x
        x = self.attn_ln(x)
        if isinstance(self.attn, HybridAttention):
            x = residual + self.attn(x, loss)[0]
        else:
            x = residual + self.attn(x, mask=mask, kv_cache=kv_cache)[0]
        return x

    def _mlp_forward(self, x):
        residual = x
        x = self.mlp_ln(x)
        x = residual + self.mlp(x)
        return x

class AudioEncoder(nn.Module):
    def __init__(self, base, n_mels, n_state, n_head, n_layer, n_ctx, max_rel_dist, checkpointing=False, cross_attention=False, hybrid_attention=False, loss=None): 
        super().__init__()
        self.base = base
        self.n_mels = n_mels
        self.n_state = n_state
        self.n_head = n_head
        self.n_layer = n_layer 
        self.n_ctx = n_ctx
        self.max_rel_dist = max_rel_dist
        self.checkpointing = checkpointing
        self.cross_attention = cross_attention 
        self.hybrid_attention = hybrid_attention
        self.loss = loss
        self.h_dim = self.n_state // self.n_head
        self.conv1 = nn.Conv1d(self.n_mels, self.n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.n_state, self.n_state, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = LearnedSinusoidalEmbeddings(self.n_ctx, self.n_state, checkpointing=self.checkpointing)

        self.combined_rotary = CombinedRotaryEmbedding(
            base=self.base,
            n_head=self.n_head,
            n_state=self.n_state,
            checkpointing=self.checkpointing,
        )

        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(
                base=self.base,
                n_state=self.n_state,
                n_head=self.n_head,
                max_rel_dist=self.max_rel_dist,              
                hybrid_attention=self.hybrid_attention,
                loss=self.loss,
            ) for _ in range(self.n_layer)
        ])

        self.ln_post = LayerNorm(self.n_state)

    def update_base(self, new_base, loss=None):
        if loss is not None and loss != self.loss: 
            self.loss = loss 
            if new_base != self.base:
                self.base = float(new_base)
                inv_freq = 1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim)) 
                self.register_buffer('inv_freq', inv_freq) 
                self.combined_rotary.update_base(self.base)

    def forward(self, x, new_base=None, loss=None):
        self.update_base(new_base, loss)

        if self.checkpointing:
            x = checkpoint(self._conv_forward, x)
        else:
            x = self._conv_forward(x=x)
        for block in self.blocks:
            if self.checkpointing:
                x = checkpoint(block, x)
            else:
                x = block(x)
        x = self.ln_post(x)
        return x

    def _conv_forward(self, x):
        x = F.gelu(input=self.conv1(x))
        x = F.gelu(input=self.conv2(x))
        x = x.permute(0, 2, 1)
        x = self.combined_rotary(x)
        pos_emb = self.positional_embedding(torch.arange(end=x.size(1), device=x.device)).unsqueeze(0)
        x = x + pos_emb
        return x

class TextDecoder(nn.Module):
    def __init__(self, base, vocab_size, n_state, n_head, n_layer, n_ctx, max_rel_dist, checkpointing=False, cross_attention=False, hybrid_attention=False, loss=None):  
        super().__init__()
        self.base = base
        self.vocab_size = vocab_size
        self.n_state = n_state
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_ctx = n_ctx
        self.max_rel_dist = max_rel_dist
        self.checkpointing = checkpointing
        self.cross_attention = cross_attention
        self.hybrid_attention = hybrid_attention
        self.loss = loss
        self.h_dim = self.n_state // self.n_head

        self.token_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.n_state)
        self.positional_embedding = LearnedSinusoidalEmbeddings(self.n_ctx, self.n_state, checkpointing=self.checkpointing)
        
        self.combined_rotary = CombinedRotaryEmbedding(
            base=self.base,
            n_head=self.n_head,
            n_state=self.n_state,
            checkpointing=self.checkpointing,
        )

        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(
                base=self.base,
                n_state=self.n_state,
                n_head=self.n_head,
                max_rel_dist=self.max_rel_dist,                
                hybrid_attention=self.hybrid_attention,
                loss=self.loss,
            ) for _ in range(self.n_layer)
        ])
        
        self.ln_post = LayerNorm(self.n_state)
        self.ln = LayerNorm(self.n_state)
        mask = torch.empty(self.n_ctx, self.n_ctx).fill_(value=-np.inf).triu_(diagonal=1)
        self.register_buffer(name="mask", tensor=mask, persistent=False)

    def update_base(self, new_base, loss=None):
        if loss is not None and loss != self.loss: 
            self.loss = loss 
            if new_base != self.base:
                self.base = float(new_base)
                inv_freq = 1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim)) 
                self.register_buffer('inv_freq', inv_freq) 
                self.combined_rotary.update_base(self.base)

    def forward(self, x, xa, kv_cache=None, new_base=None, loss=None):

        self.update_base(new_base, loss)
        x = self._embedding_forward(x, xa, kv_cache)

        for block in self.blocks:
            x = block(x, xa, loss, kv_cache)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(dtype=x.dtype), 0, 1)).float()
        return logits

    def _embedding_forward(self, x, xa, kv_cache):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        positions = torch.arange(end=x.shape[1], device=x.device) + offset
        pos_emb = self.positional_embedding(positions).unsqueeze(0)
        x = self.token_embedding(x) + pos_emb
        x = x.to(xa.dtype)
        batch_size, seq_length, embedding_dim = x.shape
        num_heads = self.n_head
        head_dim = embedding_dim // num_heads
        x = x.view(batch_size, seq_length, num_heads, head_dim)
        x = self.combined_rotary(x)
        x = x.view(batch_size, seq_length, embedding_dim)
        return x

class EchoConfig(PretrainedConfig):
    model_type = "Echo"
    def __init__(
        self,
        base=10000,
        bos_token_id=50257,
        checkpointing=False,
        cross_attention=False,
        decoder_start_token_id=50258,
        eos_token_id=50257,
        hybrid_attention=True,
        init_std=0.03,
        max_rel_dist=128,
        n_audio_ctx=1500,
        n_audio_head=16,
        n_audio_layer=24,
        n_audio_state=1024,
        n_mels=128,
        n_text_ctx=448,
        n_text_head=16,
        n_text_layer=16,
        n_text_state=1024,
        pad_token_id=50257,
        unk_token_id=50257,
        vocab_size=51865,
        **kwargs,
    ):
        super(EchoConfig, self).__init__(**kwargs)
        self.base = base
        self.bos_token_id = bos_token_id
        self.checkpointing = checkpointing
        self.cross_attention = cross_attention
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = eos_token_id
        self.hybrid_attention = hybrid_attention
        self.init_std = init_std
        self.max_rel_dist = max_rel_dist
        self.n_audio_ctx = n_audio_ctx
        self.n_audio_head = n_audio_head
        self.n_audio_layer = n_audio_layer
        self.n_audio_state = n_audio_state
        self.n_mels = n_mels
        self.n_text_ctx = n_text_ctx
        self.n_text_head = n_text_head
        self.n_text_layer = n_text_layer
        self.n_text_state = n_text_state
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.vocab_size = vocab_size

class Echo(PreTrainedModel):
    config_class = EchoConfig

    def __init__(self, config: EchoConfig):
        super().__init__(config)
        self.config = config

        self.encoder = AudioEncoder(
            base=self.config.base,
            n_mels=self.config.n_mels,
            n_state=self.config.n_audio_state, 
            n_head=self.config.n_audio_head,
            n_layer=self.config.n_audio_layer,
            n_ctx=self.config.n_audio_ctx,
            max_rel_dist=self.config.max_rel_dist,


            checkpointing=self.config.checkpointing,
            cross_attention=self.config.cross_attention,
            hybrid_attention=self.config.hybrid_attention,
            loss=getattr(self.config, 'loss', None),  
        )

        self.decoder = TextDecoder(
            base=self.config.base,
            vocab_size=self.config.vocab_size,
            n_state=self.config.n_text_state, 
            n_head=self.config.n_text_head,
            n_layer=self.config.n_text_layer,
            n_ctx=self.config.n_text_ctx,
            max_rel_dist=self.config.max_rel_dist,

            checkpointing=self.config.checkpointing,
            cross_attention=self.config.cross_attention,
            hybrid_attention=self.config.hybrid_attention,
            loss=getattr(self.config, 'loss', None),  
        )

        all_heads = torch.zeros(self.config.n_text_layer, self.config.n_text_head, dtype=torch.bool) 
        all_heads[self.config.n_text_layer // 2:] = True 
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

        self.base = self.config.base
        self.max_rel_dist= self.config.max_rel_dist
        self.adjust_counter = 0
        self.best_loss = float('inf')
        self.kv_cache = {}

    def update_dist(self, new_dist):
        self.new_dist = new_dist
        for name, module in self.encoder.named_modules():
            if isinstance(module, MultiheadAttention):
                module.update_dist(self.new_dist)
        for name, module in self.decoder.named_modules():
            if isinstance(module, MultiheadAttention):
                module.update_dist(self.new_dist)

    def adjust_max_rel_dist(self, loss, step_size=1, threshold=0.0005):
        if self.adjust_counter % 25 == 0:
            if loss < self.best_loss:
                potential_new_dist = self.max_rel_dist + step_size
            else:
                potential_new_dist = max(1, self.max_rel_dist - step_size)
            if abs(potential_new_dist - self.max_rel_dist) >= threshold:
                new_dist = potential_new_dist
                self.update_dist(new_dist)
                self.max_rel_dist = new_dist
                self.best_loss = loss

        self.adjust_counter += 1
        return self.max_rel_dist

    def adjust_base(self, loss, factor=1.0025):
        if self.adjust_counter % 25 == 0:
            if loss < self.best_loss:
                new_base = self.base * factor
            else:
                new_base = self.base / factor
            self.update_base(new_base)
            self.base = new_base
            self.best_loss = loss
   
        self.adjust_counter += 1
        return self.base

    def update_base(self, new_base):
        self.new_base=new_base
        for name, module in self.encoder.named_modules():
            if isinstance(module, (MultiheadAttention, CombinedRotaryEmbedding, AudioEncoder)):
                module.update_base(self.new_base)
        for name, module in self.decoder.named_modules():
            if isinstance(module, (MultiheadAttention, CombinedRotaryEmbedding, TextDecoder)):
                module.update_base(self.new_base)

    @staticmethod
    def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone() 
        shifted_input_ids[:, 0] = decoder_start_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    def forward(self, input_features, labels=None, dec_input_ids=None):
        if labels is not None:
            if dec_input_ids is None:
                dec_input_ids = self.shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        encoded_features = self.encoder(input_features).to(self.device)  
        logits = self.decoder(dec_input_ids, encoded_features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(logits.device).long()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

            self.adjust_base(loss.item())
            self.adjust_max_rel_dist(loss.item())

        return {"loss": loss, "logits": logits}

    def _initialize_weights(self, module):
            nn.init.normal_(self.decoder.token_embedding.weight, mean=0.0, std=self.config.init_std)
            if hasattr(self.decoder.positional_embedding, 'weight'):
                nn.init.normal_(self.decoder.positional_embedding.weight, mean=0.0, std=self.config.init_std)
            for block in self.decoder.blocks:
                for layer in block.children():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

            nn.init.constant_(self.decoder.ln.weight, 1)
            if self.decoder.ln.bias is not None:
                nn.init.constant_(self.decoder.ln.bias, 0)

            nn.init.xavier_normal_(self.encoder.conv1.weight)
            if self.encoder.conv1.bias is not None:
                nn.init.zeros_(self.encoder.conv1.bias)

            nn.init.kaiming_normal_(self.encoder.conv2.weight, mode='fan_out', nonlinearity='relu')
            if self.encoder.conv2.bias is not None:
                nn.init.zeros_(self.encoder.conv2.bias)

            nn.init.constant_(self.encoder.ln_post.weight, 1)
            if self.encoder.ln_post.bias is not None:
                nn.init.constant_(self.encoder.ln_post.bias, 0)

    def apply_initialization(self, module):
        self._initialize_weights( module )


    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.config.n_text_layer, self.config.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel):
        return self.encoder(mel)

    def logits(self, labels, input_features):
        return self.decoder(labels, input_features)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def supports_gradient_checkpointing(self):
        return True

    def install_kv_cache_hooks(self, cache = None):
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.config.n_text_ctx:
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiheadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {'input_features': input_ids}

    def _prepare_decoder_input_ids_for_generation(self, batch_size, decoder_start_token_id=None, bos_token_id=None):
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * decoder_start_token_id

    def can_generate(self):
        return True

    def generate(self, inputs, **kwargs):
        encoder_outputs = self.encoder(inputs)
        decoder_input_ids = torch.zeros((inputs.size(0), 1), dtype=torch.long, device=inputs.device)
        outputs = self.decoder(decoder_input_ids, encoder_outputs)
        return outputs.argmax(dim=-1)

    def generate_beam_search(self, inputs, **kwargs):
        encoder_outputs = self.encoder(inputs)
        decoder_input_ids = torch.zeros((inputs.size(0), 1), dtype=torch.long, device=inputs.device)
        outputs = self.decoder(decoder_input_ids, encoder_outputs)
        return outputs.argmax(dim=-1)

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=checkpoint):
        self.checkpointing = enable
        self.gradient_checkpointing_func = gradient_checkpointing_func
        for module in self.modules():
            if hasattr(module, 'checkpointing'):
                module.checkpointing = enable
                module.gradient_checkpointing_func = gradient_checkpointing_func

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": True}
        gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)

config = EchoConfig(
    base=10000,
    bos_token_id=50257,
    checkpointing=False,
    cross_attention=False,
    decoder_start_token_id=50258,
    eos_token_id=50257,
    hybrid_attention=True,
    init_std=0.02,
    max_rel_dist=128,
    n_audio_ctx=1500,
    n_audio_head=16,
    n_audio_layer=24,
    n_audio_state=1024,
    n_mels=128,
    n_text_ctx=448,
    n_text_head=16,
    n_text_layer=16,
    n_text_state=1024,
    pad_token_id=50257,
    unk_token_id=50257,
    vocab_size=51865,
    
)

model = Echo(config=config).to('cuda')
model.apply_initialization(module)


from datetime import datetime
log_dir = os.path.join('./output/', datetime.now().strftime('%Y-%m-%d_%H'))
os.makedirs(log_dir, exist_ok=True)

name="/echo_test20/"
config.save_pretrained(log_dir+name)
model.save_pretrained(log_dir+name, safe_serialization=False)
torch.save(model.state_dict(), log_dir+name+"state_dict.pt")
model = Echo.from_pretrained(pretrained_model_name_or_path=(log_dir+name)).to('cuda')


feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model_name_or_path="openai/whisper-small", feature_size=128)
tokenizer = WhisperTokenizerFast.from_pretrained(pretrained_model_name_or_path="openai/whisper-small", language="en", task="transcribe")
processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path="openai/whisper-small")

class GradientClippingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.nn.utils.clip_grad_norm_(parameters=kwargs["model"].parameters(), max_norm=0.98)

class MetricsCallback(TrainerCallback):
    def __init__(self, tb_writer, tokenizer, metric, log_every_n_steps=1):
        super().__init__()
        self.tb_writer = tb_writer
        self.tokenizer = tokenizer
        self.metric = metric
        self.log_every_n_steps = log_every_n_steps
        self.predictions = None
        self.label_ids = None

    def compute_wer(self, pred_str, label_str):
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        return wer

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        if metrics is not None:
            eval_loss = metrics.get('eval_loss')

            if state.global_step % self.log_every_n_steps == 0:
                for key, value in metrics.items():
                    if key.startswith("eval_"):
                        self.tb_writer.add_scalar(key, value, state.global_step)

        if self.predictions is not None and self.label_ids is not None:
            pred_str = self.tokenizer.batch_decode(self.predictions, skip_special_tokens=True)
            label_str = self.tokenizer.batch_decode(self.label_ids, skip_special_tokens=True)

            if state.global_step % self.log_every_n_steps == 0:
                sample_index = 0
                self.tb_writer.add_text(f"Prediction", pred_str[sample_index], state.global_step)
                self.tb_writer.add_text(f"Label", label_str[sample_index], state.global_step)
                print(f"Evaluation: - Step {state.global_step} - Loss: {eval_loss:.4f}")
                print(f"Prediction: {pred_str[sample_index]}")
                print(f"Label: {label_str[sample_index]}")
                print("-" * 10)

        self.predictions = None
        self.label_ids = None

def create_compute_metrics(callback_instance):
    def compute_metrics(eval_pred):
        pred_logits = eval_pred.predictions
        label_ids = eval_pred.label_ids

        if isinstance(pred_logits, tuple):
            pred_ids = pred_logits[0]
        else:
            pred_ids = pred_logits
        if pred_ids.ndim == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)

        label_ids[label_ids == -100] = callback_instance.tokenizer.pad_token_id
        callback_instance.predictions = pred_ids
        callback_instance.label_ids = label_ids

        pred_str = callback_instance.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = callback_instance.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * callback_instance.metric.compute(predictions=pred_str, references=label_str)

        pred_flat = pred_ids.flatten()
        labels_flat = label_ids.flatten()
        mask = labels_flat != callback_instance.tokenizer.pad_token_id

        accuracy = accuracy_score(y_true=labels_flat[mask], y_pred=pred_flat[mask])
        precision = precision_score(y_true=labels_flat[mask], y_pred=pred_flat[mask], average='weighted', zero_division=0)
        recall = recall_score(y_true=labels_flat[mask], y_pred=pred_flat[mask], average='weighted', zero_division=0)
        f1 = f1_score(y_true=labels_flat[mask], y_pred=pred_flat[mask], average='weighted', zero_division=0)

        return {"wer": wer, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    return compute_metrics

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    tokenizer: Any
    feature_extractor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

def prepare_dataset(batch):
    batch["input_features"] = feature_extractor(batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

def get_length_of_dataset(dataset):
    length = 0
    for item in dataset:
        length += len(item["audio"]["array"]) / item["audio"]["sampling_rate"]
    return length / 3600

def prepare_dataset(batch):
    batch["input_features"] = feature_extractor(batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

dataset = DatasetDict.load_from_disk("H:\ds\mapped")
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, tokenizer=tokenizer, feature_extractor=feature_extractor)
metric = evaluate.load(path="wer")
tb_writer = SummaryWriter(log_dir=log_dir)
metrics_callback = MetricsCallback(tb_writer=tb_writer, tokenizer=tokenizer, metric=metric, log_every_n_steps=20)
compute_metrics = create_compute_metrics(callback_instance=metrics_callback)

training_args = Seq2SeqTrainingArguments(
    output_dir=log_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    eval_accumulation_steps=1,
    tf32=True,
    bf16=True,
    evaluation_strategy="steps",
    max_steps=1000,
    save_steps=500,
    eval_steps=20,
    logging_steps=5,
    logging_dir=log_dir + "/logs_hf",
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    optim="adafactor",
    weight_decay=0.0025,
    disable_tqdm=False,
    save_total_limit=2,
    save_strategy="steps",
    remove_unused_columns=False,
    label_names=["labels"],
    gradient_checkpointing=False,
    eval_on_start=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
    callbacks=[metrics_callback]
)

trainer.train(resume_from_checkpoint=False)
eval_results = trainer.evaluate()
# model.save_pretrained(log_dir+name+"_b", safe_serialization=False)
# import tensorboard




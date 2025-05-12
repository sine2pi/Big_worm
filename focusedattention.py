# %%
import os
import warnings
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import numpy as np
from typing import Optional, Dict, Union, List, Tuple
from functools import partial
import gzip
import base64
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from datetime import datetime
from datasets import load_dataset, Audio
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperFeatureExtractor, WhisperTokenizer
import evaluate
import transformers
from dataclasses import dataclass
from itertools import chain
import random
from torch.amp import autocast
from torch.nn.functional import scaled_dot_product_attention

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
transformers.utils.logging.set_verbosity_error()
device = torch.device(device="cuda:0")
dtype = torch.float32

torch.set_default_dtype(dtype)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
tox = {"device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), "dtype": torch.float32}

# %xmode Minimal
# %xmode Plain
# %xmode Context
# %xmode Verbose

extractor = None
tokenizer = None
optimizer = None
scheduler = None
model = None

def set_model(model_):
    global model
    if isinstance(model_, str):
        model = torch.hub.load(model_, 'model')
    elif isinstance(model_, nn.Module):
        model = model_
    else:
        raise ValueError(f"Invalid model type: {type(model_)}. Use a string or nn.Module.")
    model.to(device)
    model.eval()

def set_device_and_dtype(device_: str, dtype_: str):
    global device, dtype
    if device_ == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif device_ == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Invalid device: {device_}. Use 'cuda' or 'cpu'.")
    if dtype_ == "float32":
        dtype = torch.float32
    elif dtype_ == "float16":
        dtype = torch.float16
    else:
        raise ValueError(f"Invalid dtype: {dtype_}. Use 'float32' or 'float16'.")
    torch.set_default_dtype(dtype)

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_extractor_and_tokenizer(extractor_, tokenizer_):
    global extractor, tokenizer
    extractor = extractor_
    tokenizer = tokenizer_

@dataclass
class Dimensions:
    vocab: int
    text_ctx: int
    text_dims: int
    text_head: int
    decoder_idx: int
    mels: int
    audio_ctx: int
    audio_dims: int
    audio_head: int
    encoder_idx: int
    pad_token_id: int
    eos_token_id: int
    decoder_start_token_id: int
    act: str
    debug: bool
    cross_attention: bool

# %%
def get_tracked_parameters(model, param_paths=None):
    if param_paths is None:
        param_paths = {
            "sw": "encoder.sw",
        }
    result = {}
    for name, path in param_paths.items():
        parts = path.split('.')
        param = model
        for part in parts:
            param = getattr(param, part)
        try:
            if isinstance(param, torch.Tensor):
                if param.numel() == 1:
                    result[name] = param if not param.requires_grad else param
                else:
                    result[name] = param.sum()
            else:
                result[name] = float(param) if hasattr(param, "__float__") else str(param)
        except Exception as e:
            result[name] = f"Error: {str(e)}"
    return result

def plot_waveform_and_spectrogram(x=None, w=None, sample_idx=0, sr=16000, title="Waveform and Spectrogram"):
    if x is not None and w is not None:
        x_np = x[sample_idx].detach().cpu().numpy()
        if x_np.shape[0] < x_np.shape[1]:
            x_np = x_np.T

        w_np = w[sample_idx].detach().cpu().numpy()
        if w_np.ndim > 1:
            w_np = w_np.squeeze()
        t = np.arange(len(w_np)) / sr
        fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
        axs[0].plot(t, w_np, color="tab:blue")
        axs[0].set_title("Waveform")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Amplitude")
        axs[1].imshow(x_np.T, aspect="auto", origin="lower", cmap="magma")
        axs[1].set_title("Spectrogram")
        axs[1].set_xlabel("Frame")
        axs[1].set_ylabel("Mel Bin")
        plt.tight_layout()
        plt.show()

    elif x is not None:
        x_np = x[sample_idx].detach().cpu().numpy()
        if x_np.shape[0] < x_np.shape[1]:
            x_np = x_np.T
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        ax.imshow(x_np.T, aspect="auto", origin="lower", cmap="magma")
        ax.set_title("Spectrogram")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mel Bin")
        plt.tight_layout()
        plt.show()

    elif w is not None:
        w_np = w[sample_idx].detach().cpu().numpy()
        if w_np.ndim > 1:
            w_np = w_np.squeeze()
        t = np.arange(len(w_np)) / sr
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        ax.plot(t, w_np, color="tab:blue")
        ax.set_title("Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        plt.tight_layout()
        plt.show()
    else:
        raise ValueError("No data to plot. Please provide at least one input tensor.")

def shift_with_zeros(labels: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    input_ids = labels.new_zeros(labels.shape)
    input_ids[:, 1:] = labels[:, :-1].clone()
    return input_ids

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)

class RMSNorm(nn.RMSNorm):
    def __init__(self, dims: Union[int, Tensor, List, Tuple], eps = 1e-8, elementwise_affine = True, device=torch.device(device="cuda:0"), dtype=torch.float32):
        tox = {"device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), "dtype": torch.float32}
        if isinstance(dims, int):
            self.normalized_shape = (dims,)
        else:
            self.normalized_shape = tuple(dims)
        super().__init__(normalized_shape=dims, eps=eps, elementwise_affine=elementwise_affine)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, **tox))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()
    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.device, x.dtype), None if self.bias is None else self.bias.to(x.device, x.dtype))

class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias) -> Tensor:
        return super()._conv_forward(x, weight.to(x.device, x.dtype), None if bias is None else bias.to(x.device, x.dtype))

class Conv2d(nn.Conv2d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias) -> Tensor:
        return super()._conv_forward(x, weight.to(x.device, x.dtype), None if bias is None else bias.to(x.device, x.dtype))

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

class Rotary(nn.Module):
    def __init__(self, dims, ctx=1500, learned_freq=True, variable_radius=False,
                 learned_radius=False, use_xpos=False, xpos_scale_base=1.0, debug=0):
        super().__init__()

        self._counter = 0
        self.debug = debug
        self.dims = dims
        max_ctx = ctx
        self.max_ctx = max_ctx
        self.variable_radius = variable_radius
        self.inv_freq = nn.Parameter(
            1.0 / (10000 ** (torch.arange(0, dims, 2) / dims)),
            requires_grad=learned_freq
        )

        if variable_radius:
            self.radius = nn.Parameter(
                torch.ones(dims // 2),
                requires_grad=learned_radius
            )

        if use_xpos:
            scale = (torch.arange(0, dims, 2) + 0.4 * dims) / (1.4 * dims)
            self.scale_base = xpos_scale_base
            self.register_buffer('scale', scale, persistent=False)

        if use_xpos:
            self.register_buffer('cached_scales', torch.zeros(ctx, dims), persistent=False)
            self.cached_scales_ctx = 0

        self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2))

    def get_seq_pos(self, ctx, device, dtype, offset=0):
        return (torch.arange(ctx, device=device, dtype=dtype) + offset) / self.interpolate_Factor

    def get_scale(self, t, ctx=None, offset=0):
        from einops import repeat
        assert self.use_xpos
        should_cache = (self.cache_if_possible and
            exists(ctx) and (offset + ctx) <= self.max_ctx)

        if (should_cache and exists(self.cached_scales) and
            (ctx + offset) <= self.cached_scales_ctx):
            return self.cached_scales[offset:(offset + ctx)]

        power = (t - len(t) // 2) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = repeat(scale, 'n d -> n (d r)', r=2)

        if should_cache and offset == 0:
            self.cached_scales[:ctx] = scale.detach()
            self.cached_scales_ctx = ctx

        return scale

    def forward(self, x = None) -> Tensor:
        if isinstance(x, int):
            t = torch.arange(x, device=self.inv_freq.device).float()
        else:
            t = x.float().to(self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs = freqs + self.bias[:freqs.shape[0]]

        if self.variable_radius:
            radius = F.softplus(self.radius)
            freqs = torch.polar(radius.unsqueeze(0).expand_as(freqs), freqs)
        else:
            freqs = torch.polar(torch.ones_like(freqs), freqs)
        freqs = freqs.unsqueeze(0)

        if self.debug:
            if self._counter == 1:
                print(f'ROTA -- freqs: {freqs.shape}, x: {x.shape if x is not None else None}', freqs.shape, x.shape)
            self._counter += 1

        return freqs

    def _reshape_for_multihead(self, freqs, head, head_dim):
        ctx = freqs.shape[0]
        complex_per_head = head_dim // 2
        if complex_per_head * head > freqs.shape[1]:
            freqs = freqs[:, :complex_per_head * head]
        elif complex_per_head * head < freqs.shape[1]:
            padding = torch.zeros(
                (ctx, complex_per_head * head - freqs.shape[1]),
                device=freqs.device,
                dtype=freqs.dtype
            )
            freqs = torch.cat([freqs, padding], dim=1)
        freqs = freqs.view(ctx, head, complex_per_head)
        return freqs.permute(2, 1, 0, 2).unsqueeze(0)

    @staticmethod
    def apply_rotary(x, freqs):
        multihead_format = len(freqs.shape) == 4

        if multihead_format:
            x1 = x[..., :freqs.shape[-1]*2]
            x2 = x[..., freqs.shape[-1]*2:]
            x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
            x1 = torch.view_as_complex(x1)
            x1 = x1 * freqs
            x1 = torch.view_as_real(x1).flatten(-2)
            return torch.cat([x1.type_as(x), x2], dim=-1)
        else:
            x1 = x[..., :freqs.shape[-1]*2]
            x2 = x[..., freqs.shape[-1]*2:]
            x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
            x1 = torch.view_as_complex(x1)
            x1 = x1 * freqs
            x1 = torch.view_as_real(x1).flatten(-2)
            return torch.cat([x1.type_as(x), x2], dim=-1)

# %%
class Multihead(nn.Module):
    use_sdpa = False
    def __init__(self, dims: int, head: int, ctx, debug=None):
        super().__init__()
        self._counter = 0
        self.debug = debug
        self.head = head
        self.dims = dims
        self.ctx = ctx
        self.head_dim = dims // head
        self.q = Linear(dims, dims)
        self.k = Linear(dims, dims, bias=False)
        self.v = Linear(dims, dims)
        self.out = Linear(dims, dims)

        self.Rotary = Rotary(dims=self.head_dim, use_xpos=True, learned_freq=False)
        self.Factor = nn.Parameter(torch.tensor(0.0005))

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        decoder: Optional[bool] = False):

        q = self.q(x)
        if kv_cache is None or xa is None or self.k not in kv_cache:
            k = self.k(x if xa is None else xa)
            v = self.v(x if xa is None else xa)
        else:
            k = kv_cache[self.k]
            v = kv_cache[self.v]

        wv, qk = self._forward(q, k, v, mask)
        return self.out(wv), qk

    def _forward(self, q: Tensor, k: Tensor, v: Tensor, mask = None, freq=None, Freq=None, temperature=1.0):

        batch, ctx, dims = q.shape
        head = self.head
        head_dim = self.head_dim
        scale = (dims // self.head) ** -0.25

        q = q.view(batch, ctx, head, head_dim).transpose(1, 2).contiguous()
        k = k.view(batch, ctx, head, head_dim).transpose(1, 2).contiguous()
        v = v.view(batch, ctx, head, head_dim).transpose(1, 2).contiguous()

        if Multihead.use_sdpa:
            try:
                is_causal = mask is not None and ctx > 1
                attn_output = scaled_dot_product_attention(q, k, v, is_causal=is_causal)
                return attn_output, None
            except RuntimeError:
                pass

        token_ids = k[:, :, :, 0]
        scaled_zero = torch.ones_like(token_ids)
        zero_Factor = torch.clamp(F.softplus(self.Factor), min=0.0, max=0.001)
        scaled_zero[token_ids.float() == 0] = zero_Factor.to(q.device, q.dtype)
        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        if mask is not None:
            qk = prepare_mask_for_attention(qk, mask)

        # if mask is not None:
        #     mask = mask[:ctx, :ctx].to(q.device, q.dtype)

            qk = qk * mask.unsqueeze(0).unsqueeze(0)
        qk = qk * scaled_zero.unsqueeze(-2)
        qk = qk.float()
        w = F.softmax(qk / temperature, dim=-1)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.out(wv), qk.detach()

class ProjectionModule(nn.Module):
    def __init__(self, dims: int, head: int, proj_type: str = "query", use_bias: bool = True):
        super().__init__()
        assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.proj_type = proj_type
        self.scale = self.head_dim ** -0.25 if proj_type != "value" else 1.0
        self.proj = Linear(in_features=dims, out_features=dims, bias=use_bias)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(tensor=self.proj.weight, std=0.02)
        if hasattr(self.proj, 'bias') and self.proj.bias is not None:
            nn.init.zeros_(tensor=self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        batch, ctx = x.shape[:2]
        proj = self.proj(x)

        proj = proj.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        if self.proj_type in ["query", "key"]:
            proj = proj * self.scale
        return proj

def prepare_mask_for_attention(attn, mask):
    batch, head, q_ctx, k_ctx = attn.shape

    if mask.dim() == 2:
        mask_ctx = min(mask.size(0), q_ctx)
        mask_to_apply = mask[:mask_ctx, :mask_ctx]
        attn[:, :, :mask_ctx, :mask_ctx] = attn[:, :, :mask_ctx, :mask_ctx] + mask_to_apply
    elif mask.dim() == 3:
        mask_ctx = min(mask.size(1), q_ctx)
        mask_to_apply = mask[:, :mask_ctx, :mask_ctx]
        attn[:, :, :mask_ctx, :mask_ctx] = attn[:, :, :mask_ctx, :mask_ctx] + mask_to_apply.unsqueeze(1)
    elif mask.dim() == 4:
        mask_q_ctx = min(mask.size(2), q_ctx)
        mask_k_ctx = min(mask.size(3), k_ctx)
        attn[:, :, :mask_q_ctx, :mask_k_ctx] = attn[:, :, :mask_q_ctx, :mask_k_ctx] + mask[:, :, :mask_q_ctx, :mask_k_ctx]
    return attn

def calculate_attention(q, k, v, mask=None, temperature=1.0, use_sdpa=True):
    if use_sdpa:
        try:
            is_causal = mask is not None and q.size(2) > 1
            attn_output = scaled_dot_product_attention(q, k, v, is_causal=is_causal)
            return attn_output, None
        except RuntimeError:
            pass

    attn = torch.matmul(q, k.transpose(-1, -2))
    if mask is not None:
        attn = prepare_mask_for_attention(attn, mask)

    attn = F.softmax(attn / temperature, dim=-1)
    attn_output = torch.matmul(attn, v)

    return attn_output, attn

class BaseAttention(nn.Module):
    use_sdpa = True

    def __init__(self, dims: int, head: int, ctx, debug=False):
        super().__init__()
        assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.ctx = ctx
        self.scale = self.head_dim ** -0.25
        self.debug = debug
        self.q = Linear(dims, dims)
        self.k = Linear(dims, dims, bias=False)
        self.v = Linear(dims, dims)
        self.out = Linear(dims, dims)

        self.Rotary = Rotary(dims=self.head_dim, use_xpos=True, learned_freq=False)
        self.Factor = nn.Parameter(torch.tensor(0.0005))

    def _reshape_to_output(self, attn_output, batch, ctx):
        return attn_output.permute(0, 2, 1, 3).reshape(batch, ctx, self.dims)

class MultiheadD(BaseAttention):
    def __init__(self, dims: int, head: int, ctx):
        super().__init__(dims, head, ctx)
        self.out = Linear(in_features=dims, out_features=dims)
        nn.init.normal_(tensor=self.out.weight, std=0.02)
        nn.init.zeros_(tensor=self.out.bias)

    @autocast('cuda', enabled=True)
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if q.dim() == 3:
            batch, ctx, _ = q.shape
            q = q.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(batch, k.size(1), self.head, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(batch, v.size(1), self.head, self.head_dim).permute(0, 2, 1, 3)
        else:
            batch = q.size(0)
            ctx = q.size(2)

        attn_output, _ = calculate_attention(q, k, v, mask, 1.0, BaseAttention.use_sdpa)
        output = self._reshape_to_output(attn_output, batch, ctx)
        return self.out(output)

class MultiheadC(BaseAttention):
    def __init__(self, dims: int, head: int, ctx):
        super().__init__(dims, head, ctx)

        self.query_module = ProjectionModule(dims, head, "query")
        self.key_module = ProjectionModule(dims, head, "key")
        self.value_module = ProjectionModule(dims, head, "value")

        self.combiner = MultiheadD(dims, head, ctx)
        self.key_update_predictor = nn.Sequential(
            Linear(dims, dims // 4), nn.ReLU(), Linear(dims // 4, 1), nn.Sigmoid()
        )
        self.value_update_predictor = nn.Sequential(
            Linear(dims, dims // 4), nn.ReLU(), Linear(dims // 4, 1), nn.Sigmoid()
        )
        self.update_threshold = 0.5
        self.stored_key_cache = None
        self.stored_value_cache = None

    def should_update_key(self, x: torch.Tensor) -> torch.Tensor:
        avg_rep = x.mean(dim=1)
        return self.key_update_predictor(avg_rep) > self.update_threshold

    def should_update_value(self, x: torch.Tensor) -> torch.Tensor:
        avg_rep = x.mean(dim=1)
        return self.value_update_predictor(avg_rep) > self.update_threshold

    def forward(self, x, xa=None, mask=None, kv_cache=None):
        batch, ctx, _ = x.shape
        q = self.query_module(x)
        kv_input = xa if xa is not None else x
        device = kv_input.device

        if kv_cache is None:
            k = self.key_module(kv_input)
            v = self.value_module(kv_input)

            self.stored_key_cache = k
            self.stored_value_cache = v
        else:
            update_k = self.should_update_key(kv_input)
            update_v = self.should_update_value(kv_input)

            if update_k.any():
                new_k = self.key_module(kv_input)
                if self.stored_key_cache is not None:
                    update_mask = update_k.view(-1, 1, 1, 1).expand_as(self.stored_key_cache)
                    k = torch.where(update_mask, new_k, self.stored_key_cache)
                else:
                    k = new_k
            else:
                k = self.stored_key_cache if self.stored_key_cache is not None else self.key_module(kv_input)

            if update_v.any():
                new_v = self.value_module(kv_input)
                if self.stored_value_cache is not None:
                    update_mask = update_v.view(-1, 1, 1, 1).expand_as(self.stored_value_cache)
                    v = torch.where(update_mask, new_v, self.stored_value_cache)
                else:
                    v = new_v
            else:
                v = self.stored_value_cache if self.stored_value_cache is not None else self.value_module(kv_input)

            self.stored_key_cache = k
            self.stored_value_cache = v
        output = self.combiner(q, k, v, mask=mask)
        return output

class Refiner:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.states = states
        self.actions = actions
        self.R = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.default_value = 0.0

    def get_value(self, state, action):
        return self.R.get((state, action), self.default_value)

    def set_value(self, state, action, value):
        self.R[(state, action)] = value

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.actions)
        else:
            action_values = [self.get_value(state, a) for a in range(self.actions)]
            return np.argmax(action_values)

    def update(self, state, action, reward, next_state):
        next_values = [self.get_value(next_state, a) for a in range(self.actions)]
        best_next_value = max(next_values)
        old_value = self.get_value(state, action)
        td_target = reward + self.gamma * best_next_value
        td_error = td_target - old_value
        new_value = old_value + self.alpha * td_error
        self.set_value(state, action, new_value)

class Predictor(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.linear = Linear(in_features=dims, out_features=1)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, global_out):
        if global_out.dim() > 2:
            global_out = global_out.mean(dim=1)
        scale = torch.sigmoid(self.linear(global_out))
        return scale

class MultiheadB(BaseAttention):
    def __init__(self, dims, head, ctx, sharpen=True, temp_scale=0.01):
        super().__init__(dims, head, ctx)
        self.sharpen = sharpen
        self.temp_scale = temp_scale
        self.span_scale = nn.Parameter(torch.tensor(1.0))
        self.counter = True

    @autocast('cuda', enabled=True)
    def forward(self, q, k, c, ctx=None, max_span=None, span_scale=None):
        batch, ctx, _ = q.shape
        if ctx is None:
            ctx = q.shape[1]
        if max_span is None:
            max_span = q.shape[1]
        if span_scale is None:
            span_scale = self.span_scale

        span_mean = span_scale.mean().item()
        span_len = min(int(max_span * span_mean), q.shape[1], k.shape[1], c.shape[1])
        eff_span = min(span_len, ctx)

        if eff_span == 0:
            batch = q.shape[0]
            return (torch.zeros(batch, eff_span, self.dims, device=q.device), None)

        q_span = q[:, :eff_span, :]
        k_span = k[:, :eff_span, :]
        v_span = c[:, :eff_span, :]

        batch = q_span.shape[0]

        reshape_dims = (batch, -1, self.head, self.head_dim)
        q = q_span.view(*reshape_dims).permute(0, 2, 1, 3)
        k = k_span.view(*reshape_dims).permute(0, 2, 1, 3)
        v = v_span.view(*reshape_dims).permute(0, 2, 1, 3)

        temperature = (
            1.0 + self.temp_scale * (1.0 - span_mean)
            if self.sharpen
            else 0.5 + self.temp_scale * span_mean
        )

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            attn_output, weights = calculate_attention(q, k, v, None, temperature, BaseAttention.use_sdpa)
            out = self._reshape_to_output(attn_output, batch, eff_span)

        if self.counter:
            print(f"AdaptiveSpan - out: {out}, weights: {weights}")
        self.counter = False
        return out, weights

class MultiheadA(nn.Module):
    def __init__(self, dims, head, ctx, win_size=256, max_span=512, temp_scale=0.01):
        super().__init__()
        self.head = head
        self.ctx = ctx
        self.dims = dims
        self.max_span = max_span
        self.sliding_window = win_size
        self.temp_scale = temp_scale
        self.sharpen = True
        self.head_dim = dims // head
        self.batch = None
        self.s_factor = nn.Parameter(torch.tensor(1.0))
        self.refiner = Refiner(states=10000, actions=10, alpha=0.1, gamma=0.9, epsilon=0.1
        )
        self.span_pred = Predictor(dims=dims)
        self.attn_local = MultiheadB(dims=dims, head=head, ctx=ctx, sharpen=True, temp_scale=temp_scale
        )
        self.attn_global = MultiheadC(dims=dims, head=head, ctx=ctx)
        self.projection = Linear(in_features=2 * dims, out_features=dims)
        self.ln_a = LayerNorm(normalized_shape=dims)
        self.ln_b = LayerNorm(normalized_shape=dims)
        mask_buffer = torch.empty(self.ctx, self.ctx).fill_(float("-inf")).triu_(diagonal=1)
        self.register_buffer("mask", mask_buffer, persistent=False)
        self.register_buffer("window_mask", None, persistent=False)
        self.register_buffer("threshold", torch.tensor(1e-4), persistent=False)

    def forward(self, x, xa=None, mask=None, kv_cache=None, decoder=False):
        batch, q_ctx, dims = x.shape

        local_normed = self.ln_a(x)
        globe_normed = self.ln_b(x)

        local_attention_mask_to_use = None
        if decoder:
            if hasattr(self, 'mask') and self.mask is not None:
                 local_attention_mask_to_use = self.mask[:q_ctx, :q_ctx]
        global_kv_source = xa
        global_attention_input_mask_to_use = None

        if xa is not None:
            global_attention_input_mask_to_use = mask

            globe_out = self.attn_global(globe_normed, xa=global_kv_source, mask=global_attention_input_mask_to_use, kv_cache=kv_cache)
        else:
            if decoder:
                if hasattr(self, 'mask') and self.mask is not None:
                    global_attention_input_mask_to_use = self.mask[:q_ctx, :q_ctx]
            globe_out = self.attn_global(globe_normed, xa=None, mask=global_attention_input_mask_to_use, kv_cache=kv_cache)

        raw_freq_scale = self.span_pred(globe_out)
        freq_scale = torch.nan_to_num(raw_freq_scale, nan=0.0)
        state = self.extract(local_normed)
        action = self.refiner.choose_action(state=state)
        refine = self.action_scale(action=action)
        span_scale = torch.clamp(freq_scale * refine, min=0.0, max=1.0)
        span_mean = span_scale.mean().item()
        current_win_size = max(1, int(self.sliding_window * span_mean))
        current_span_len = max(1, int(self.max_span * span_mean))
        local_dynamic_ctx = min(self.ctx, q_ctx, current_span_len, current_win_size)
        self.attn_local.ctx = local_dynamic_ctx
        local_out = self.slide_win(
            x=local_normed,
            win_size=current_win_size,
            span_len=current_span_len,
            span_scale=span_scale,
            mask=local_attention_mask_to_use,
        )
        with torch.no_grad():
            quality = self.quality(output=local_out)
            next_state = self.extract(local_out)
            self.refiner.update(
                state=state, action=action, reward=quality, next_state=next_state)

        combined = torch.cat([local_out, globe_out], dim=-1)
        output = self.projection(combined)
        return output

    def quality(self, output):
        with torch.no_grad():
            safe_output = output.clamp(min=1e-10)
            entropy = -(safe_output * torch.log(safe_output)).sum(-1).mean()
            coverage = (output > 0.01).float().mean()
            return float(coverage - 0.1 * entropy)

    def extract(self, x):
        with torch.no_grad():
            meadims = x.mean(dim=(0, 1))
            var_state = x.var(dim=(0, 1), unbiased=False)
            state = torch.cat([meadims, var_state])
            state_id = self.discretize(state.cpu().numpy())
        return state_id

    def discretize(self, state):
        bins = np.linspace(-1, 1, num=10)
        state_discrete = np.digitize(state, bins)
        state_hash = hash(tuple(state_discrete))
        state_id = state_hash % (self.refiner.states - 1)
        return state_id

    def action_scale(self, action):
        span_value = action / (self.refiner.actions - 1)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        span_scale = torch.tensor([span_value], device=device, dtype=dtype)
        return span_scale

    @autocast('cuda', enabled=True)
    def _focus(self, query, key, value, span_scale, mask):

        max_iterations = 10
        iteration = 0
        prev_attn = torch.zeros_like(input=query)
        attn_out = torch.zeros_like(input=query)

        threshold = self.threshold.item()
        s_factor = self.s_factor

        while iteration < max_iterations:
            span_len = int(self.max_span * span_scale.mean().item())
            span_len = min(span_len, query.size(1), key.size(1), value.size(1))
            eff_span = min(span_len, self.ctx)

            if eff_span == 0:
                break
            q_span = query[:, :eff_span, :]
            k_span = key[:, :eff_span, :]
            v_span = value[:, :eff_span, :]

            batch, ctx, dims = q_span.size()

            q = q_span.view(batch, ctx, self.head, -1).transpose(1, 2)
            k = k_span.view(batch, ctx, self.head, -1).transpose(1, 2)
            v = v_span.view(batch, ctx, self.head, -1).transpose(1, 2)

            if self.sharpen:
                temperature = 1.0 + self.temp_scale * (1.0 - span_scale.mean().item())
            else:
                temperature = 0.5 + self.temp_scale * span_scale.mean().item()

            if mask is not None and (mask.size(-2) != q.size(-2) or mask.size(-1) != k.size(-2)):
                mask_q_ctx = min(mask.size(-2), q.size(-2))
                mask_k_ctx = min(mask.size(-1), k.size(-2))
                resized_mask = torch.ones(
                    (batch, self.head, q.size(-2), k.size(-2)),
                    device=mask.device,
                    dtype=mask.dtype,
                )
                resized_mask[:, :, :mask_q_ctx, :mask_k_ctx] = mask[:, :, :mask_q_ctx, :mask_k_ctx]
                mask_to_use = resized_mask
            else:
                mask_to_use = mask

            attn_output, weights = calculate_attention(q, k, v, mask_to_use, temperature, BaseAttention.use_sdpa)
            attn_out = attn_output.transpose(1, 2).contiguous().view(batch, ctx, -1)

            diff = torch.abs(attn_out - prev_attn).mean()
            dynamic_threshold = threshold + s_factor * diff

            if diff < dynamic_threshold:
                break

            prev_attn = attn_out
            query = query + attn_out
            iteration += 1
        return attn_out

    @autocast('cuda', enabled=True)
    def slide_win(self, x, win_size, span_len, span_scale, mask):
        batch, ctx, dims = x.size()
        self.batch = batch
        num_windows = (ctx + win_size - 1) // win_size
        output = torch.zeros_like(x)
        device = x.device

        window_mask_for_focus = None

        for i in range(num_windows):
            start_idx = i * win_size
            end_idx = min((i + 1) * win_size, ctx)
            current_window_q_ctx = end_idx - start_idx

            key_val_start_idx = max(0, start_idx - span_len + current_window_q_ctx)
            key_val_end_idx = min(ctx, start_idx + span_len)
            current_window_kv_len = key_val_end_idx - key_val_start_idx

            query_window = x[:, start_idx:end_idx, :]
            key_window = x[:, key_val_start_idx:key_val_end_idx, :]
            value_window = key_window

            if current_window_q_ctx == 0 or current_window_kv_len == 0:
                continue

            if mask is not None:
                if mask.dim() == 2:

                    sliced_2d_mask = mask[start_idx:end_idx, key_val_start_idx:key_val_end_idx]

                    window_mask_for_focus = sliced_2d_mask.unsqueeze(0).unsqueeze(0).expand(
                        batch, self.head, current_window_q_ctx, current_window_kv_len
                    ).to(device=device, dtype=torch.float32)
                elif mask.dim() == 4:

                    window_mask_for_focus = mask[:, :, start_idx:end_idx, key_val_start_idx:key_val_end_idx]
                    if window_mask_for_focus.size(1) == 1 and self.head > 1:
                        window_mask_for_focus = window_mask_for_focus.expand(-1, self.head, -1, -1)
                else:
                    window_mask_for_focus = None
            else:
                window_mask_for_focus = None

            attn_out = self._focus(
                query=query_window,
                key=key_window,
                value=value_window,
                span_scale=span_scale,
                mask=window_mask_for_focus,
            )

            if attn_out.shape[1] == output[:, start_idx:end_idx, :].shape[1]:
                 output[:, start_idx:end_idx, :] = attn_out
            else:

                output[:, start_idx:start_idx+attn_out.shape[1], :] = attn_out
        return output

# %%
class Residual(nn.Module):
    def __init__(self, dims: int, head: int, ctx, act, cross_attention=False, debug=False):
        super().__init__()
        ctx = ctx
        self._counter = 0
        self.dropout = 0.1
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.cross_attention = cross_attention
        self.debug = debug

        self.blend_xa = nn.Parameter(torch.tensor(0.5))
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(),
                   "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        self.act = act_map.get(act, nn.GELU())

        self.attna = MultiheadA(dims, head, ctx=ctx)
        self.attnb = (MultiheadA(dims, head, ctx=ctx) if cross_attention else None)

        mlp = dims * 4
        self.mlp = nn.Sequential(Linear(dims, mlp), self.act, Linear(mlp, dims))
        self.lna = RMSNorm(dims)
        self.lnb = RMSNorm(dims) if cross_attention else None
        self.lnc = RMSNorm(dims)

    def forward(self, x, xa=None, mask=None, kv_cache=None, decoder=False):

        r = x
        x = x + self.attna(self.lna(x), mask=mask, kv_cache=kv_cache, decoder=decoder)[0]
        if self.attnb and xa is not None:
            cross_out = self.attnb(self.lnb(x), xa, kv_cache=kv_cache, decoder=decoder)[0]
            blend = torch.sigmoid(self.blend_xa)
            x = blend * x + (1 - blend) * cross_out
        x = x + self.mlp(self.lnc(x))
        x = x + r

        if self._counter < 1 and self.debug:
            print("--------")
            print("RESIDUAL")
            print(f"Is decoder: {decoder}")
            print(f"kv_cache: {kv_cache is not None}")
            print(f"Input x shape: {x.shape}")
            print(f"Input xa shape: {xa.shape if xa is not None else None}")
            print(f"Mask: {mask.shape if mask is not None else None}")

        self._counter += 1
        return x

# %%
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class SWave(nn.Module):
    def __init__(self, channels, reduction=1, kernel_sizes=None):
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        super().__init__()
        self.temporal_convs = nn.ModuleList(
            [Conv1d(2, 1, kernel_size=k, padding=k - 1, padding_mode="zeros", bias=True) for k in kernel_sizes]
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels),
        )
        self.temporal_combine = nn.Conv1d(len(kernel_sizes), 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t = x.size()
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        channel_out = self.sigmoid(avg_out + max_out).view(b, c, 1)
        x = x * channel_out
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        temporal_outputs = []
        for conv in self.temporal_convs:
            conv_out = conv(y)
            temporal_outputs.append(conv_out[:, :, :t])
        if len(self.temporal_convs) > 1:
            temporal_combined = torch.cat(temporal_outputs, dim=1)
            temporal_att = self.sigmoid(self.temporal_combine(temporal_combined))
        else:
            temporal_att = self.sigmoid(temporal_outputs[0])
        return x * temporal_att

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=1, kernel_size=7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
        )
        self.conv = nn.Conv1d(
            2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t = x.size()
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        channel_out = self.sigmoid(avg_out + max_out).view(b, c, 1)
        x = x * channel_out
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)[:, :, :t]
        temporal_att = self.sigmoid(y)
        return x * temporal_att

class AudioEncoder(nn.Module):
    def __init__(self, mels: int, layer: int, dims: int, head: int, ctx: int, act: str, debug):
        super().__init__()
        self._counter = 0
        self.debug = debug
        self.dims = dims
        self.head = head
        self.ctx = ctx
        self.head_dim = dims // head
        self.dropout = 0.1

        self.Rotary = Rotary(dims=self.dims, ctx=ctx, learned_freq=False, variable_radius=False, learned_radius=False)

        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(),
                   "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(),
                   "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        self.act = act_map.get(act, nn.GELU())

        self.sw = nn.Parameter(torch.tensor(0.5))
        self.ln_enc = RMSNorm(dims, **tox)
        self.positional_embedding = lambda length: sinusoids(length, dims)

        self.se = nn.Sequential(
            Conv1d(mels, dims, kernel_size=3, padding=1), self.act,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=2, dilation=2),
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims),
            Conv1d(dims, dims, kernel_size=1),
            CBAMBlock(dims, reduction=1), self.act,
            nn.Dropout(p=self.dropout),
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1))

        self.we = nn.Sequential(
            nn.Conv1d(1, dims, kernel_size=11, stride=5, padding=5),
            self.act,
            nn.Conv1d(dims, dims, kernel_size=5, stride=2, padding=2),
            self.act,
            SWave(dims, reduction=1, kernel_sizes=None),
            # nn.AdaptiveAvgPool1d(1),
            )

        self.blockA = (nn.ModuleList([
            Residual(dims=dims, head=head, ctx=ctx, act=act, debug=debug) for _ in range(layer)]) if layer > 0 else None)

    def forward(self, x, w, decoder=False) -> Tensor:
        """ batch, ctx, dims = B, L, D  -  x is spectrogram (B, Mels, L) - w is waveform (B, 1, L) """
        blend = torch.sigmoid(self.sw)

        if self._counter < 1 and self.debug:
            plot_waveform_and_spectrogram(x=x, w=w)
            print(f"Initial Spectrogram tensor shape: {x.shape if x is not None else None}, Initial Waveform tensor shape: {w.shape if w is not None else None}")

        if x is not None:
            x = x.to(device=device)
            x = self.se(x).permute(0, 2, 1) # (B, L, D)
            x = x + self.positional_embedding(x.shape[1]).to(x.device, x.dtype)

        if w is not None: # self.we output shape: (B, D, L)
            w = w.to(device=device)
            w = self.we(w).permute(0, 2, 1) # (B, L, D)

        if x is not None:
            if w is not None:
                if w.shape[1] != x.shape[1]:
                    w = w.permute(0, 2, 1) # (B, L, D) -> (B, D, L)
                    w = F.interpolate(w, size=x.shape[1], mode='linear', align_corners=False) #  w_L to x_L
                    w = w.permute(0, 2, 1) # (B, D, L) -> (B, L, D)
                else:
                    w = w
                w = w + self.positional_embedding(w.shape[1]).to(w.device, w.dtype)
                x = blend * x + (1 - blend) * w
            else:
                x = x
        elif w is not None:
            w = w + self.positional_embedding(w.shape[1]).to(w.device, w.dtype)
            x = w
        else:
            raise ValueError("You have to provide either x (spectrogram) or w (waveform)")

        ctx = x.shape[1]
        freqs = self.Rotary(ctx)
        x = self.Rotary.apply_rotary(x, freqs)

        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        for block in chain(self.blockA or []):
            x = block(x, decoder=decoder)

        if self._counter < 1 and self.debug:
            print("-------")
            print("ENCODER")
            print(f"Features to Residual Blocks shape: {x.shape}")
            print(f"Original input x shape: {x.shape if x is not None else None}")
            print(f"Original input w shape: {w.shape if w is not None else None}")
            print(f"Positional embedding for a length of {x.shape[1]} would be: {self.positional_embedding(x.shape[1]).shape}")

        self._counter += 1
        return self.ln_enc(x)

# %%
class TextDecoder(nn.Module):
    def __init__(self, vocab: int, layer: int, dims: int, head: int, ctx: int, cross_attention: bool, debug):
        super().__init__()
        self._counter = 0
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.debug = debug
        self.dropout = 0.1

        self.Rotary = Rotary(dims=dims, learned_freq=False, variable_radius=False, learned_radius=False)
        self.token_embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=dims)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()

        self.positional_embedding = nn.Parameter(data=torch.empty(ctx, dims), requires_grad=True)
        self.ln_dec = RMSNorm(dims=dims)

        self.blockA = (nn.ModuleList([
            Residual(dims=dims, head=head, ctx=ctx, act="gelu", cross_attention=cross_attention, debug=debug) for _ in range(layer)]) if layer > 0 else None)

        mask = (torch.triu(torch.ones(ctx, ctx), diagonal=1))
        mask = torch.where(mask == 1, torch.tensor(0.0), torch.tensor(1.0))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, xa, kv_cache=None, decoder=True) -> Tensor:

        mask=self.mask
        x = x.to(device=device)
        ctx = x.shape[1]
        freqs = self.Rotary(ctx)

        # if ctx != mask.shape[0]:
        #     mask = mask[:ctx, :ctx].to(x.device, x.dtype)
        # mask = sliding_mask(ctx, mask)

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (self.token_embedding(x) + self.positional_embedding[offset: offset + x.shape[-1]])
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.Rotary.apply_rotary(x, freqs)

        for block in chain(self.blockA or []):
            x = block(x, xa=xa, mask=mask, kv_cache=kv_cache, decoder=decoder)

        x = self.ln_dec(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()
        return logits

# %%
class Echo(nn.Module):
    def __init__(self, param: Dimensions):
        super().__init__()
        self.param = param

        self.shared = nn.ModuleDict({
            # "rotary": Rotary(dims=param.audio_dims // param.audio_head, ctx=param.audio_ctx),
            "rotary_encoder": Rotary(dims=param.audio_dims // param.audio_head, ctx=param.audio_ctx),
            "rotary_decoder": Rotary(dims=param.text_dims // param.text_head, ctx=param.text_ctx),

        })

        self.param_tracking_paths = {
            "RotationA": "encoder.blockA.0.attna.Rotary.inv_freq",
            "RotationB": "decoder.Rotary.inv_freq",
            "Silence": "encoder.blockA.0.attna.Factor",
        }

        self.encoder = AudioEncoder(
            debug=param.debug,
            mels=param.mels,
            ctx=param.audio_ctx,
            dims=param.audio_dims,
            head=param.audio_head,
            layer=param.encoder_idx,
            act=param.act,
        )

        self.decoder = TextDecoder(
            debug=param.debug,
            vocab=param.vocab,
            ctx=param.text_ctx,
            dims=param.text_dims,
            head=param.text_head,
            layer=param.decoder_idx,
            cross_attention=param.cross_attention,

            # rotary=param.rotary,
            # Factor=param.Factor,
            # rotary_config=param.rotary_config,
        )

        all_head = torch.zeros(self.param.decoder_idx, self.param.text_head, dtype=torch.bool)
        all_head[self.param.decoder_idx // 2 :] = True
        self.register_buffer("alignment_head", all_head.to_sparse(), persistent=False)

    def set_alignment_head(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool).copy()
        mask = torch.from_numpy(array).reshape(
            self.param.decoder_idx, self.param.text_head)
        self.register_buffer("alignment_head", mask.to_sparse(), persistent=False)

    def embed_audio(self, input_features: torch.Tensor):
        return self.encoder(input_features)

    def logits(self,input_ids: torch.Tensor, encoder_output: torch.Tensor):
        return self.decoder(input_ids, encoder_output)

    def forward(self,
        decoder_input_ids=None,
        labels=None,
        input_features: torch.Tensor=None,
        waveform: Optional[torch.Tensor]=None,
        input_ids=None,
    ) -> Dict[str, torch.Tensor]:

        if labels is not None:
            if input_ids is None:
                input_ids = shift_with_zeros(
                    labels, self.param.pad_token_id, self.param.decoder_start_token_id).to('cuda')
            decoder_input_ids = input_ids
        if input_features is not None:
            if waveform is not None:
                encoder_output = self.encoder(x=input_features, w=waveform)
            else:
                encoder_output = self.encoder(x=input_features, w=None)
        elif waveform is not None:
            encoder_output = self.encoder(x=None, w=waveform)
        else:
            raise ValueError("You have to provide either input_features or waveform")
        logits = self.decoder(input_ids, encoder_output)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=0)

        return {
            "logits": logits,
            "loss": loss,
            "labels": labels,
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "encoder_output": encoder_output
        }

    @property
    def device(self):
        return next(self.parameters()).device

    def _init_weights(self, module):
        std = 0.02
        self.init_counts = {"Linear": 0, "Conv1d": 0, "LayerNorm": 0, "RMSNorm": 0,
            "Conv2d": 0, "SEBlock": 0, "TextDecoder": 0, "AudioEncoder": 0, "Residual": 0,
                            "MultiheadA": 0, "MultiheadB": 0, "MultiheadC": 0, "MultiheadD": 0}

        for name, module in self.named_modules():
            if isinstance(module, Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Linear"] += 1
            elif isinstance(module, Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv1d"] += 1
            elif isinstance(module, LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                self.init_counts["LayerNorm"] += 1
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
                self.init_counts["RMSNorm"] += 1
            elif isinstance(module, MultiheadA):
                self.init_counts["MultiheadA"] += 1
            elif isinstance(module, MultiheadB):
                self.init_counts["MultiheadB"] += 1
            elif isinstance(module, MultiheadC):
                self.init_counts["MultiheadC"] += 1
            elif isinstance(module, MultiheadD):
                self.init_counts["MultiheadD"] += 1
            elif isinstance(module, Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv2d"] += 1
            elif isinstance(module, SEBlock):
                nn.init.ones_(module.fc[0].weight)
                nn.init.zeros_(module.fc[0].bias)
                nn.init.ones_(module.fc[2].weight)
                nn.init.zeros_(module.fc[2].bias)
                self.init_counts["SEBlock"] += 1
            elif isinstance(module, TextDecoder):
                self.init_counts["TextDecoder"] += 1
            elif isinstance(module, AudioEncoder):
                nn.init.xavier_uniform_(module.se[0].weight)
                nn.init.zeros_(module.se[0].bias)
                nn.init.xavier_uniform_(module.se[2].weight)
                nn.init.zeros_(module.se[2].bias)
                nn.init.xavier_uniform_(module.se[4].weight)
                nn.init.zeros_(module.se[4].bias)
                self.init_counts["AudioEncoder"] += 1
            elif isinstance(module, Residual):
                self.init_counts["Residual"] += 1

    def init_weights(self):
        print("Initializing all weights")
        self.apply(self._init_weights)
        print("Initialization summary:")
        for module_type, count in self.init_counts.items():
            print(f"{module_type}: {count}")

# %%

metric = evaluate.load(path="wer")

@dataclass
class DataCollator:

    def __call__(self, features: List[Dict[str, Union[List[int], Tensor]]]) -> Dict[str, Tensor]:
        global extractor, tokenizer
        decoder_start_token_id = tokenizer.bos_token_id
        pad_token_id = tokenizer.pad_token_id
        input_feature_pad_value = 0.0
        waveform_pad_value = 0.0

        if decoder_start_token_id is None:
            raise ValueError("The tokenizer does not have a bos_token_id. Please set it manually.")
        batch = {}

        if "input_features" in features[0] and features[0]["input_features"] is not None:
            input_features_list = [f["input_features"] for f in features]
            max_len_feat = max(f.shape[1] for f in input_features_list)

            padded_input_features = []
            for feat in input_features_list:
                num_mels, current_len = feat.shape
                padding_needed = max_len_feat - current_len
                if padding_needed > 0:
                    padded_feat = F.pad(feat, (0, padding_needed), mode='constant', value=input_feature_pad_value)
                else:
                    padded_feat = feat
                padded_input_features.append(padded_feat)
            batch["input_features"] = torch.stack(padded_input_features)

        if "waveform" in features[0] and features[0]["waveform"] is not None:
            waveform_list = [f["waveform"] for f in features]
            max_len_wav = max(w.shape[1] for w in waveform_list)

            padded_waveforms = []
            for wav in waveform_list:
                _, current_len = wav.shape
                padding_needed = max_len_wav - current_len
                if padding_needed > 0:
                    padded_wav = F.pad(wav, (0, padding_needed), mode='constant', value=waveform_pad_value)
                else:
                    padded_wav = wav
                padded_waveforms.append(padded_wav)
            batch["waveform"] = torch.stack(padded_waveforms)

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = tokenizer.pad(label_features, return_tensors="pt", padding="longest", pad_to_multiple_of=None)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), 0)
        if (labels[:, 0] == decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        batch["input_ids"] = shift_with_zeros(labels, pad_token_id, decoder_start_token_id)
        return batch

def prepare_dataset(batch, input_features=True, waveform=True):
    global extractor, tokenizer
    audio = batch["audio"]
    wav = torch.tensor(audio["array"]).float()
    if waveform:
        batch["waveform"] = wav.unsqueeze(0)
    if input_features:
        features_np = extractor(wav, sampling_rate=audio["sampling_rate"], padding="longest", return_tensors="pt").input_features[0]
        current_features = torch.tensor(features_np)
        pad_val = current_features.min().item()
        current_features = torch.where(current_features == pad_val, torch.tensor(0.0, dtype=current_features.dtype), current_features)
        batch["input_features"] = current_features
    batch["labels"] = tokenizer(batch["transcription"], add_special_tokens=False).input_ids
    return batch

def compute_metrics(eval_pred, compute_result: bool = True):
    global extractor, tokenizer, model, optimizer, scheduler

    pred_logits = eval_pred.predictions
    label_ids = eval_pred.label_ids
    if hasattr(pred_logits, "cpu"):
        pred_logits = pred_logits.cpu()
    if hasattr(label_ids, "cpu"):
        label_ids = label_ids.cpu()
    if isinstance(pred_logits, tuple):
        pred_ids = pred_logits[0]
    else:
        pred_ids = pred_logits
    if hasattr(pred_ids, "ndim") and pred_ids.ndim == 3:
        if not isinstance(pred_ids, torch.Tensor):
            pred_ids = torch.tensor(pred_ids)
        pred_ids = pred_ids.argmax(dim=-1)
        pred_ids = pred_ids.tolist()
    elif hasattr(pred_ids, "tolist"):
        pred_ids = pred_ids.tolist()
    if hasattr(label_ids, "tolist"):
        label_ids = label_ids.tolist()
    label_ids = [
        [tokenizer.pad_token_id if token == -100 else token for token in seq]
        for seq in label_ids
    ]
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # print("--------------------------------")
    # print(f"Prediction: {pred_str[0]}")
    # print(f"Label: {label_str[0]}")

    pred_flat = list(chain.from_iterable(pred_ids))
    labels_flat = list(chain.from_iterable(label_ids))
    mask = [i != tokenizer.pad_token_id for i in labels_flat]

    acc = accuracy_score(
        [i for i, m in zip(labels_flat, mask) if m],
        [p for p, m in zip(pred_flat, mask) if m]
    )
    pre = precision_score(
        [i for i, m in zip(labels_flat, mask) if m],
        [p for p, m in zip(pred_flat, mask) if m],
        average='weighted', zero_division=0
    )
    rec = recall_score(
        [i for i, m in zip(labels_flat, mask) if m],
        [p for p, m in zip(pred_flat, mask) if m],
        average='weighted', zero_division=0
    )
    f1 = f1_score(
        [i for i, m in zip(labels_flat, mask) if m],
        [p for p, m in zip(pred_flat, mask) if m],
        average='weighted', zero_division=0
    )
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    # tracked_params = get_tracked_parameters(model, model.param_tracking_paths)

    # RotationA = tracked_params["RotationA"]
    # RotationB = tracked_params["RotationB"]
    # Silence = tracked_params["Silence"]

    metrics = {
        "wer": wer,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        # "RotationA": RotationA,
        # "RotationB": RotationB,
        # "Silence": Silence,
    }
        # metrics.update(tracked_params)
    return metrics

def create_model(param):
    model = Echo(param).to('cuda')
    model.init_weights()
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    return model

def setup_tokenizers(token):
    global extractor, tokenizer

    extractor = WhisperFeatureExtractor.from_pretrained(
        "openai/whisper-small",
        token=token,
        feature_size=128,
        sampling_rate=16000,
        do_normalize=True,
        return_tensors="pt",
        chunk_ctxgth=15,
        padding_value=0.0,
        padding=False,
    )

    tokenizer = WhisperTokenizer.from_pretrained(
        "./tokenizer",
        pad_token="0",
        bos_token="0",
        eos_token="0",
        unk_token="0",
        token=token,
        local_files_only=True,
        pad_token_id=0
    )

    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 0
    tokenizer.eos_token_id = 0
    tokenizer.decoder_start_token_id = 0

def prepare_datasets(token):
    dataset = load_dataset("google/fleurs", "en_us", token=token, trust_remote_code=True, streaming=False)
    dataset = dataset.cast_column(column="audio", feature=Audio(sampling_rate=16000))
    def filter_func(x):
        return (0 < len(x["transcription"]) < 512 and
                len(x["audio"]["array"]) > 0 and
                len(x["audio"]["array"]) < 1500 * 160)
    dataset = dataset.filter(filter_func).shuffle(seed=4)
    print("Dataset size:", dataset["train"].num_rows, dataset["test"].num_rows)
    prepare_fn = partial(prepare_dataset, input_features=True, waveform=True)
    dataset = dataset.map(function=prepare_fn, remove_columns=list(next(iter(dataset.values())).features)
    ).with_format(type="torch")
    train_dataset = dataset["train"].shuffle(seed=4).flatten_indices()
    test_dataset = dataset["test"].shuffle(seed=4).take(200).flatten_indices()
    return train_dataset, test_dataset

def get_training_args(log_dir):
    return Seq2SeqTrainingArguments(
        output_dir=log_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        tf32=True,
        bf16=True,
        eval_strategy="steps",
        save_strategy="steps",
        max_steps=10000,
        save_steps=10001,
        eval_steps=1000,
        warmup_steps=100,
        num_train_epochs=1,
        logging_steps=100,
        logging_dir=log_dir,
        report_to=["tensorboard"],
        push_to_hub=False,
        disable_tqdm=False,
        save_total_limit=1,
        label_names=["labels"],
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        learning_rate=0.0025,
        weight_decay=0.25,
        save_safetensors=False,
        eval_on_start=False,
        batch_eval_metrics=True,
    )


# %%

if __name__ == "__main__":

    param = Dimensions(
        mels=128,
        audio_ctx=1500,
        audio_head=4,
        encoder_idx=4,
        audio_dims=512,
        vocab=51865,
        text_ctx=256,
        text_head=4,
        decoder_idx=4,
        text_dims=512,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=0,
        act="swish",
        debug=None,
        cross_attention=False,

    )

    token = ""
    log_dir = os.path.join('./output/logs', datetime.now().strftime(format='%m-%d_%H'))
    os.makedirs(name=log_dir, exist_ok=True)

    setup_tokenizers(token)
    model = create_model(param)
    train_dataset, test_dataset = prepare_datasets(token)
    training_args = get_training_args(log_dir)


    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollator(),
        compute_metrics=compute_metrics,
        processing_class=extractor,
    )

    trainer.train()



# from tensorboard import program
# log_dir = "./output/logs"
# tb = program.TensorBoard()
# tb.configure(argv=[None, '--logdir', log_dir])
# url = tb.launch()
# print(f"TensorBoard started at {url}")



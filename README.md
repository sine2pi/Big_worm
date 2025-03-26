#### IntegratedAttention: Dynamic Multi-Scale Attention with adaptive focus

###### Usage: attn = IntegratedAttention(ctx, dims, head)

##### IntegratedAttention combines adaptive local and global attention mechanisms with reinforcement learning to dynamically adjust attention spans based on content.

##### Core Components
```
IntegratedAttention
├── Refiner (Q-learning agent)
├── AdaptiveSpan (Local attention)
├── AdaptiveUpdateAttention (Global attention)
├── Span Predictor (Neural frequency estimator)
└── Window/span adaptation mechanisms
```
**RL + Heuristics**
- ✓ Computationally efficient
- ✓ No higher-order gradients needed
- ✓ Well-defined discrete states and actions
- ✓ Stable training dynamics


##### How It Works

1. **Dual Processing Paths**:
   - Local path with sliding window attention
   - Global path with content-dependent update frequencies

2. **Dynamic Span Control**:
   - Neural predictor generates initial span scale
   - RL agent selects action to refine span scale
   - Combined scale determines attention window size

3. **Reinforcement Learning Loop**:
   - Extract state from input representations
   - Choose action (span adjustment) via epsilon-greedy
   - Compute quality metrics as reward signal
   - Update Q-values for future decisions

4. **Adaptive Window Sliding**:
   - Dynamically sized windows based on content complexity
   - Adjustable overlap between windows
   - Iterative focusing within each window

5. **Integration Mechanism**:
   - Combine local and global attention outputs
   - Project to original dimension space


- **Content-dependent span adaptation**
- **Sliding window with variable sizes**
- **Q-learning for attention optimization**
- **Quality-driven feedback loop**
- **Iterative attention refinement**

This mechanism allows the model to dynamically focus on relevant context while maintaining computational efficiency by adapting to the content complexity.


``` python
import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn import functional as F
from typing import Tuple, Optional
from torch.nn.functional import scaled_dot_product_attention

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype))
        
class ProjectionModule(nn.Module):
    """Unified projection module that handles query, key, and value transformations."""
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
    """Prepares and applies mask to attention scores."""
    batch, head, q_len, k_len = attn.shape
    
    if mask.dim() == 2:
        mask_len = min(mask.size(0), q_len)
        mask_to_apply = mask[:mask_len, :mask_len]
        attn[:, :, :mask_len, :mask_len] = attn[:, :, :mask_len, :mask_len] + mask_to_apply
    elif mask.dim() == 3:
        mask_len = min(mask.size(1), q_len)
        mask_to_apply = mask[:, :mask_len, :mask_len]
        attn[:, :, :mask_len, :mask_len] = attn[:, :, :mask_len, :mask_len] + mask_to_apply.unsqueeze(1)
    elif mask.dim() == 4:
        mask_q_len = min(mask.size(2), q_len)
        mask_k_len = min(mask.size(3), k_len)
        attn[:, :, :mask_q_len, :mask_k_len] = attn[:, :, :mask_q_len, :mask_k_len] + mask[:, :, :mask_q_len, :mask_k_len]
    
    return attn

def calculate_attention(q, k, v, mask=None, temperature=1.0, use_sdpa=True):
    """Shared attention calculation logic."""
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
    """Base class for attention mechanisms with common functionality."""
    use_sdpa = True
    
    def __init__(self, dims: int, head: int, max_dist: int = 512):
        super().__init__()
        assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.max_dist = max_dist
        self.scale = self.head_dim ** -0.25
        
    def _reshape_to_output(self, attn_output, batch, seq_len):
        """Reshape attention output to original dimensions."""
        return attn_output.permute(0, 2, 1, 3).reshape(batch, seq_len, self.dims)

class AttentionCombiner(BaseAttention):
    """Combines separate Q and KV representations for attention computation."""
    def __init__(self, dims: int, head: int):
        super().__init__(dims, head)
        self.out = Linear(in_features=dims, out_features=dims)
        nn.init.normal_(tensor=self.out.weight, std=0.02)
        nn.init.zeros_(tensor=self.out.bias)

    @autocast('cuda', enabled=True)
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Processes and combines attention inputs."""
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

class AdaptiveUpdateAttention(BaseAttention):
    """Attention implementation with content-dependent update frequencies."""
    def __init__(self, dims: int, head: int, max_dist=512):
        super().__init__(dims, head, max_dist)
        
        self.query_module = ProjectionModule(dims, head, "query")
        self.key_module = ProjectionModule(dims, head, "key")
        self.value_module = ProjectionModule(dims, head, "value")
        
        self.combiner = AttentionCombiner(dims, head)
        
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
        """Predict whether the key should be updated based on content."""
        avg_rep = x.mean(dim=1)
        return self.key_update_predictor(avg_rep) > self.update_threshold

    def should_update_value(self, x: torch.Tensor) -> torch.Tensor:
        """Predict whether the value should be updated based on content."""
        avg_rep = x.mean(dim=1)
        return self.value_update_predictor(avg_rep) > self.update_threshold

    def forward(self, x, xa=None, mask=None, kv_cache=None):
        """Process inputs with adaptive update mechanism."""
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
    """Q-learning based refiner for adaptive attention span."""
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
    """Neural predictor for span scale estimation."""
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

class AdaptiveSpan(BaseAttention):
    """Attention with adaptive span size."""
    def __init__(self, dims, head, max_dist, sharpen=True, temp_scale=0.01):
        super().__init__(dims, head, max_dist)
        self.sharpen = sharpen
        self.temp_scale = temp_scale
        self.span_scale = nn.Parameter(torch.tensor(1.0))

    @autocast('cuda', enabled=True)
    def forward(self, query, key, value, max_dist=None, max_span=None, span_scale=None):
        if max_dist is None:
            max_dist = self.max_dist
        if max_span is None:
            max_span = query.shape[1]
        if span_scale is None:
            span_scale = self.span_scale
            
        span_mean = span_scale.mean().item()
        span_len = min(int(max_span * span_mean), query.shape[1], key.shape[1], value.shape[1])
        eff_span = min(span_len, max_dist)
        
        if eff_span == 0:
            batch = query.shape[0]
            return (torch.zeros(batch, eff_span, self.dims, device=query.device), None)
            
        q_span = query[:, :eff_span, :]
        k_span = key[:, :eff_span, :]
        v_span = value[:, :eff_span, :]

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

        return out, weights

class IntegratedAttention(nn.Module):
    """Combines local adaptive span and global content-dependent attention with RL-based adaptation."""
    def __init__(self, ctx, dims, head, max_dist=512, win_size=256, max_span=384, temp_scale=0.01):
        super().__init__()
        self.head = head
        self.max_dist = max_dist
        self.ctx = ctx
        self.dims = dims
        self.max_span = max_span
        self.sliding_window = win_size
        self.temp_scale = temp_scale
        self.sharpen = True
        self.head_dim = dims // head
        self.batch = None

        self.refiner = Refiner(
            states=10000, actions=10, alpha=0.1, gamma=0.9, epsilon=0.1
        )
        
        self.span_pred = Predictor(dims=dims)
        self.attn_local = AdaptiveSpan(
            dims=dims, head=head, max_dist=max_dist, sharpen=True, temp_scale=temp_scale
        )
        self.attn_global = AdaptiveUpdateAttention(dims=dims, head=head, max_dist=max_dist)
        
        self.projection = Linear(in_features=2 * dims, out_features=dims)
        self.ln_a = LayerNorm(normalized_shape=dims)
        self.ln_b = LayerNorm(normalized_shape=dims)

        mask = torch.empty(max_span, max_span).fill_(float("-inf")).triu_(diagonal=1)
        self.register_buffer("mask", mask, persistent=False)
        self.register_buffer("window_mask", None, persistent=False)
        self.register_buffer("threshold", torch.tensor(1e-4), persistent=False)
        self.register_buffer("s_factor", torch.tensor(0.1), persistent=False)

    def forward(self, x, xa=None, mask=None, kv_cache=None):
        """Process input with integrated local and global attention."""
        if mask is None:
            mask = self.mask
            
        local = self.ln_a(x)
        globe = self.ln_b(x)

        globe_out = self.attn_global(globe, globe, globe)
        
        freq_scale = self.span_pred(globe_out)
        
        state = self.extract(local)

        action = self.refiner.choose_action(state=state)
        refine = self.action_scale(action=action)

        span_scale = torch.clamp(freq_scale * refine, min=0.0, max=1.0)
        span_mean = span_scale.mean().item()

        with torch.no_grad():
            current_win_size = max(1, int(self.sliding_window * span_mean))
            current_span_len = max(1, int(self.max_span * span_mean))

            effective_max = min(self.max_dist, local.size(1))
            local_max = min(self.max_dist, current_span_len, current_win_size)
            globe_max = effective_max

        self.attn_local.max_dist = local_max
        self.attn_global.max_dist = globe_max

        local_out = self.slide_win(
            x=local,
            win_size=current_win_size,
            span_len=current_span_len,
            span_scale=span_scale,
            mask=mask,
        )
        
        with torch.no_grad():
            quality = self.quality(output=local_out)
            next_state = self.extract(local_out)
            self.refiner.update(
                state=state, action=action, reward=quality, next_state=next_state)
        
        combined = torch.cat([local_out, globe_out], dim=-1)
        x = self.projection(combined)
        return x

    def quality(self, output):
        """Calculate quality metric for reinforcement learning."""
        with torch.no_grad():
            safe_output = output.clamp(min=1e-10)
            entropy = -(safe_output * torch.log(safe_output)).sum(-1).mean()
            coverage = (output > 0.01).float().mean()
            return float(coverage - 0.1 * entropy)

    def extract(self, x):
        """Extract state features for RL agent."""
        with torch.no_grad():
            meadims = x.mean(dim=(0, 1))
            var_state = x.var(dim=(0, 1), unbiased=False)
            state = torch.cat([meadims, var_state])
            state_id = self.discretize(state.cpu().numpy())
        return state_id

    def discretize(self, state):
        """Convert continuous state to discrete state ID."""
        bins = np.linspace(-1, 1, num=10)
        state_discrete = np.digitize(state, bins)
        state_hash = hash(tuple(state_discrete))
        state_id = state_hash % (self.refiner.states - 1)
        return state_id

    def action_scale(self, action):
        """Convert discrete action to continuous scale factor."""
        span_value = action / (self.refiner.actions - 1)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        span_scale = torch.tensor([span_value], device=device, dtype=dtype)
        return span_scale
    
    @autocast('cuda', enabled=True)
    def _focus(self, query, key, value, span_scale, mask):
        """Iterative attention refinement."""
        max_iterations = 10
        iteration = 0
        prev_attn = torch.zeros_like(input=query)
        attn_out = torch.zeros_like(input=query)
        attn_weights = None

        threshold = self.threshold.item()
        s_factor = self.s_factor.item()

        while iteration < max_iterations:
            span_len = int(self.max_span * span_scale.mean().item())
            span_len = min(span_len, query.size(1), key.size(1), value.size(1))
            eff_span = min(span_len, self.max_dist)

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
                mask_q_len = min(mask.size(-2), q.size(-2))
                mask_k_len = min(mask.size(-1), k.size(-2))
                resized_mask = torch.ones(
                    (batch, self.head, q.size(-2), k.size(-2)),
                    device=mask.device,
                    dtype=mask.dtype,
                )
                resized_mask[:, :, :mask_q_len, :mask_k_len] = mask[:, :, :mask_q_len, :mask_k_len]
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
            
        return attn_out, attn_weights
    
    @autocast('cuda', enabled=True)
    def slide_win(self, x, win_size, span_len, span_scale, mask):
        """Process input with sliding window attention."""
        batch, ctx, dims = x.size()
        self.batch = batch
        num_windows = (ctx + win_size - 1) // win_size
        output = torch.zeros_like(x)
        device = x.device
        default_mask = None

        for i in range(num_windows):
            start_idx = i * win_size
            end_idx = min((i + 1) * win_size, ctx)
            window_size = end_idx - start_idx

            key_start = max(0, start_idx - span_len + win_size)
            key_end = min(start_idx + span_len, ctx)
            span_size = key_end - key_start

            query = x[:, start_idx:end_idx, :]
            key = x[:, key_start:key_end, :]
            value = key

            if mask is not None:
                if mask.dim() == 4:
                    window_mask = mask[:, :, start_idx:end_idx, key_start:key_end]
                    if window_mask.size(1) == 1:
                        window_mask = window_mask.expand(-1, self.head, -1, -1)
                else:
                    if (
                        default_mask is None
                        or default_mask.size(-2) != window_size
                        or default_mask.size(-1) != span_size
                    ):
                        default_mask = torch.ones(
                            (batch, self.head, window_size, span_size),
                            device=device,
                            dtype=torch.bool,
                        )
                    window_mask = default_mask
            else:
                if (
                    default_mask is None
                    or default_mask.size(-2) != window_size
                    or default_mask.size(-1) != span_size
                ):
                    default_mask = torch.ones(
                        (batch, self.head, window_size, span_size),
                        device=device,
                        dtype=torch.bool,
                    )
                window_mask = default_mask

            attn_out, _ = self._focus(
                query=query,
                key=key,
                value=value,
                span_scale=span_scale,
                mask=window_mask,
            )

            output[:, start_idx:end_idx, :] = attn_out

        return output

```



1. Long-Range Dependencies and Specificity

2. Avoiding "Attention Collapse" with Long Spans

3. Tasks Requiring Precise Identification within Broad Context
   
4. Hierarchical Reasoning
   
5. Sparsity Inducement
   


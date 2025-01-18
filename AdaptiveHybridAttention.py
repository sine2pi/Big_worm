
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
        self.multihead_attn = MultiheadAttention(self.base, self.n_state, self.n_head, self.max_rel_dist, self.window_size, checkpointing=self.checkpointing)
        self.span_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, key, value, span_scale):
        span_length = int(self.max_span * span_scale)
        span_length = min(span_length, query.shape[1], key.shape[1], value.shape[1])
        effective_span = min(span_length, self.max_rel_dist)

        query_span = query[:, :effective_span, :]
        key_span = key[:, :effective_span, :]
        value_span = value[:, :effective_span, :]

        attn_output, attn_weights = self.multihead_attn(query_span, key_span, value_span)
        temperature = 1.0 - self.temperature_scale_factor * span_scale  # Dynamic temperature

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
        attn_output = torch.bmm(attn_weights.permute(0, 2, 1, 3).flatten(end_dim=2), value_span)  
        return attn_output, attn_weights

class SpanPredictor(nn.Module):
    def __init__(self, n_state):
        super().__init__()
        self.linear = nn.Linear(n_state, 1)

    def forward(self, global_attention_output):
        scale = torch.sigmoid(self.linear(global_attention_output))
        return scale

class HybridAttention(nn.Module):
    def __init__(self, base, n_state, n_head, max_rel_dist, window_size, max_span, sliding_window=16):
        super().__init__()
        self.base = base
        self.n_state = n_state
        self.n_head = n_head
        self.max_rel_dist = max_rel_dist
        self.window_size = window_size
        self.max_span = max_span
        self.sliding_window = sliding_window
        self.span_predictor = SpanPredictor(n_state)
        assert self.n_state % self.n_head == 0, "n_state must be divisible by n_head"

        self.local_max_rel_dist = max_rel_dist  
        self.global_max_rel_dist = max_rel_dist

        self.local_attn = AdaptiveSpanAttention(self.base, self.n_state, self.n_head, self.local_max_rel_dist, self.window_size, self.max_span)
        self.global_attn = MultiheadAttention(self.base, self.n_state, self.n_head, self.global_max_rel_dist, self.window_size, checkpointing=False)

        self.ln_local = nn.LayerNorm(self.n_state)
        self.ln_global = nn.LayerNorm(self.n_state)
        self.projection = nn.Linear(2 * self.n_state, self.n_state)

    def forward(self, x, loss=None): 
        x_local = self.ln_local(x)
        x_global = self.ln_global(x)
        x_local = x_local.permute(1, 0, 2)  
        x_global = x_global.permute(1, 0, 2)  
        global_out, _ = self.global_attn(x_global, x_global, x_global)

        span_scale = self.span_predictor(global_out.mean(dim=1)) 
        window_size = max(1, int(self.sliding_window * span_scale)) 
        span_length = max(1, int(self.max_span * span_scale))   

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
    
class MultiheadAttention(nn.Module):
    def __init__(self, base, n_state, n_head, max_rel_dist: int, window_size: float, checkpointing=False, loss=None):
        super().__init__()
        self.base = base
        self.n_state = n_state
        self.n_head = n_head
        self.max_rel_dist = max_rel_dist
        self.window_size = window_size
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
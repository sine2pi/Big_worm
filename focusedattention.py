
class AdaptiveSpanAttention(nn.Module):
    def __init__(self, n_state, n_head, initial_max_dist, max_max_dist, n_freq, win_size, max_span, temp_scale=0.01):
        super().__init__()

        self.max_span = max_span
        self.temp_scale = temp_scale
        self.multi_attn = MultiheadAttention(n_state, n_head, n_freq) 
        self.relative_bias = RelativePositionalBias(initial_max_dist, max_max_dist, n_head)
        self.current_max_dist = initial_max_dist
        self.max_max_dist = max_max_dist
        self.span_scale = nn.Parameter(torch.tensor(1.0))

    def update_max_dist(self, new_dist, device):
        if new_dist is not None and new_dist != self.current_max_dist:
            if new_dist > self.max_max_dist:
                raise ValueError(f"new_dist ({new_dist}) exceeds maximum allowed max_dist ({self.max_max_dist})")
            self.current_max_dist = new_dist
            self.relative_bias.update_dist(new_dist, device)

    def forward(self, query, key, value, span_scale, new_dist=None, temperature=1.0):
        
        self.update_max_dist(new_dist, device=query.device)

        span_len = int(self.max_span * span_scale.mean().item())
        span_len = min(span_len, query.shape[1], key.shape[1], value.shape[1])

        eff_span = min(span_len, self.current_max_dist)
        q_span = query[:, :eff_span, :]
        k_span = key[:, :eff_span, :]
        v_span = value[:, :eff_span, :]

        batch_size, seq_len_q = q_span.size(0), q_span.size(1)
        seq_len_k = k_span.size(1)
        rel_pos_bias = self.relative_bias(seq_len_q, seq_len_k, device=query.device)

        attn_out, attn_weights = self.multi_attn(q_span, k_span, v_span, rel_pos_bias=rel_pos_bias, temperature=temperature, span_scale=span_scale)
        
        attn_out = attn_out.view(query.size(0), -1, query.size(2))

        return attn_out, attn_weights

class SpanPredictor(nn.Module):
    def __init__(self, n_state):
        super().__init__()
        self.linear = nn.Linear(n_state, 1)

    def forward(self, global_out):
        scale = torch.sigmoid(self.linear(global_out))
        return scale

class HybridAttention(nn.Module):
    def __init__(self, n_state, n_head, initial_max_dist, max_max_dist, n_freq, win_size=32, max_span=32, slid_win=32, temp_scale=0.01):
        super().__init__()
        self.win_size = win_size
        self.max_span = max_span
        self.slid_win = slid_win
        self.temp_scale = temp_scale

        self.span_pred = SpanPredictor(n_state)
        self.initial_max_dist = initial_max_dist
        self.max_max_dist = max_max_dist
        self.dist_local = initial_max_dist
        self.dist_global = initial_max_dist
        self.attn_local = AdaptiveSpanAttention(n_state, n_head, self.dist_local, max_max_dist, n_freq, win_size, max_span, temp_scale)
        self.attn_global = MultiheadAttention(n_state, n_head, n_freq)
        self.relative_bias_global = RelativePositionalBias(initial_max_dist, max_max_dist, n_head)
        self.ln_local = nn.LayerNorm(n_state)
        self.ln_global = nn.LayerNorm(n_state)
        self.projection = nn.Linear(2 * n_state, n_state)

    def update_max_dist_local(self, new_dist, device):
        if new_dist is not None and new_dist != self.dist_local:
            if new_dist > self.max_max_dist:
                raise ValueError(f"new_dist ({new_dist}) exceeds maximum allowed max_dist ({self.max_max_dist})")
            self.dist_local = new_dist
            self.attn_local.update_max_dist(new_dist, device)

    def update_max_dist_global(self, new_dist, device):
        if new_dist is not None and new_dist != self.dist_global:
            if new_dist > self.max_max_dist:
                raise ValueError(f"new_dist ({new_dist}) exceeds maximum allowed max_dist ({self.max_max_dist})")
            self.dist_global = new_dist
            self.relative_bias_global.update_dist(new_dist, device)

    def forward(self, x, new_dist=None, new_base=None, xa=None, mask=None, kv_cache=None):
        local = self.ln_local(x)
        globe = self.ln_global(x)

        self.update_max_dist_global(new_dist, device=x.device)
        batch_size, seq_len_q = globe.size(0), globe.size(1)
        seq_len_k = seq_len_q if xa is None else xa.size(1)
        rel_pos_bias_global = self.relative_bias_global(seq_len_q, seq_len_k, device=x.device)
        globe_out, _ = self.attn_global(globe, xa, mask, kv_cache, rel_pos_bias=rel_pos_bias_global)

        span_scale = self.span_pred(globe_out.mean(dim=1))

        win_size = max(1, int(self.slid_win * span_scale.mean().item()))
        span_len = max(1, int(self.max_span * span_scale.mean().item()))

        local_n_dist = min(self.dist_local, span_len, win_size)
        self.update_max_dist_local(local_n_dist, device=x.device)

        temperature = 1.0 - self.temp_scale * span_scale

        local_out = self.slid_win_attention(local, win_size, span_len, span_scale, new_dist, temperature)
        combined = torch.cat([local_out, globe_out], dim=-1)
        x = self.projection(combined)
        return x

    def slid_win_attention(self, x, win_size, span_len, span_scale, new_dist, temperature):
        batch_size, seq_len, n_state = x.size()
        out = torch.zeros_like(x, device=x.device)

        for i in range(0, seq_len, win_size):
            end = min(i + win_size, seq_len)
            query = x[:, i:end, :]
            start = max(0, i - span_len + win_size)
            key = x[:, start:i + span_len, :]
            value = x[:, start:i + span_len, :]

            attn_out, _ = self.attn_local(query, key, value, span_scale, new_dist, temperature)
            out[:, i:end, :] = attn_out

        return out

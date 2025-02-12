
class FocusedAttention(nn.Module):
    def __init__(self, base, dims, head, max_dist, sharpen, win_size=32, max_span=32, slid_win=32, temp_scale=0.01):
        super().__init__()
        self.max_dist = max_dist
        self.win_size = win_size
        self.max_span = max_span
        self.slid_win = slid_win
        self.temp_scale = temp_scale

        self.span_scale = nn.Parameter(torch.tensor(1.0))
        self.sharpen = sharpen

        self.multihead_attn = MultiheadAttention(base=base, dims=dims, head=head, max_dist=max_dist)
        self.dist_local = max_dist
        self.dist_global = max_dist

        self.attn_global = MultiheadAttention(base=base, dims=dims, head=head, max_dist=self.dist_global)
        self.ln_local = LayerNorm(normalized_shape=dims)
        self.ln_global = LayerNorm(normalized_shape=dims)
        self.projection = Linear(in_features=2 * dims, out_features=dims)

        self.linear = nn.Linear(in_features=dims, out_features=1)

    def forward(self, x, new_dist=None, new_base=None, xa=None, mask=None, kv_cache=None):
        local = self.ln_local(x)
        globe = self.ln_global(x)

        globe_out, _ = self.attn_global(globe, globe, globe)

        span_scale = torch.sigmoid(self.linear(globe_out.mean(dim=1)))

        win_size = max(1, int(self.slid_win * span_scale.mean().item()))
        span_len = max(1, int(self.max_span * span_scale.mean().item()))

        effective_max_dist = min(self.max_dist, local.size(1))
        local_max_dist = min(self.dist_local, span_len, win_size)
        globe_max_dist = effective_max_dist

        local_out = self.slide_win(x=local, win_size=win_size, span_len=span_len, span_scale=span_scale)

        combined = torch.cat(tensors=[local_out, globe_out], dim=-1)
        x = self.projection(combined)

        return x

    def slide_win(self, x, win_size, span_len, span_scale):
        batch_size, seq_len, dims = x.size()
        out = torch.zeros_like(x, device=x.device)

        for i in range(0, seq_len, win_size):
            end = min(i + win_size, seq_len)
            query = x[:, i:end, :]

            start = max(0, i - span_len + win_size)
            key = x[:, start:i + span_len, :]
            value = x[:, start:i + span_len, :]

            span_len_attn = int(self.max_span * span_scale.mean().item())
            span_len_attn = min(span_len_attn, query.shape[1], key.shape[1], value.shape[1])
            eff_span = min(span_len_attn, self.max_dist)

            q_span = query[:, :eff_span, :]
            k_span = key[:, :eff_span, :]
            v_span = value[:, :eff_span, :]

            batch_size, _, dims = query.shape
            scale = (dims // self.multihead_attn.head) ** -0.25

            q = q_span.view(q_span.shape[0], q_span.shape[1], self.multihead_attn.head, -1).permute(0, 2, 1, 3)
            k = k_span.view(k_span.shape[0], k_span.shape[1], self.multihead_attn.head, -1).permute(0, 2, 1, 3)
            v = v_span.view(v_span.shape[0], v_span.shape[1], self.multihead_attn.head, -1).permute(0, 2, 1, 3)

            if self.sharpen:
                temperature = 1.0 + self.temp_scale * (1.0 - span_scale.mean().item())
            else:
                temperature = 0.5 + self.temp_scale * span_scale.mean().item()

            attn_scores = torch.matmul(q, k.transpose(-2, -1))
            attn_weights = torch.softmax((attn_scores / temperature) * scale, dim=-1)
            attn_out = torch.matmul(attn_weights, v)
            attn_out = attn_out.permute(0, 2, 1, 3).flatten(start_dim=2)
            attn_out = attn_out.contiguous().view(batch_size, eff_span, dims)

            out[:, i:end, :] = attn_out

        return out

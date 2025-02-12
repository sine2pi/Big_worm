

class FocusedAttention(nn.Module):
    def __init__(self, base, dims, head, max_dist, sharpen, win_size=32, max_span=32, slid_win=32, temp_scale=0.01):
        super().__init__()
        self.base = base
        self.dims = dims
        self.head = head
        self.max_dist = max_dist
        self.sharpen = sharpen
        self.win_size = win_size
        self.max_span = max_span
        self.slid_win = slid_win
        self.temp_scale = temp_scale

        self.span_scale_param = nn.Parameter(torch.tensor(1.0))
        self.span_predictor = nn.Linear(in_features=dims, out_features=1)

        self.multihead_attn_local = MultiheadAttention(base=base, dims=dims, head=head, max_dist=max_dist)
        self.multihead_attn_global = MultiheadAttention(base=base, dims=dims, head=head, max_dist=max_dist)

        self.ln_local = LayerNorm(normalized_shape=dims)
        self.ln_global = LayerNorm(normalized_shape=dims)
        self.projection = Linear(in_features=2 * dims, out_features=dims)

    def forward(self, x):

        local = self.ln_local(x)
        global_ = self.ln_global(x)

        globe_out, _ = self.multihead_attn_global(global_, global_, global_)

        span_scale = torch.sigmoid(self.span_predictor(globe_out.mean(dim=1)))

        win_size = max(1, int(self.slid_win * span_scale.mean().item()))
        span_len = max(1, int(self.max_span * span_scale.mean().item()))

        effective_max_dist = min(self.max_dist, local.size(1))
        local_max_dist = min(self.max_dist, span_len, win_size)
        globe_max_dist = effective_max_dist

        self.multihead_attn_local.max_dist = local_max_dist
        self.multihead_attn_global.max_dist = globe_max_dist

        local_out = self._window(local, win_size, span_len, span_scale)

        combined = torch.cat([local_out, globe_out], dim=-1)
        x = self.projection(combined)

        return x

    def _window(self, x, win_size, span_len, span_scale):
        batch_size, seq_len, dims = x.size()
        output = torch.zeros_like(x, device=x.device)

        for i in range(0, seq_len, win_size):
            end = min(i + win_size, seq_len)
            query = x[:, i:end, :]

            start = max(0, i - span_len + win_size)
            key = x[:, start:i + span_len, :]
            value = x[:, start:i + span_len, :]

            attn_out, _ = self._focus(query, key, value, span_scale)
            output[:, i:end, :] = attn_out

        return output

    def _focus(self, query, key, value, span_scale):
        span_len = int(self.max_span * span_scale.mean().item())
        span_len = min(span_len, query.size(1), key.size(1), value.size(1))
        eff_span = min(span_len, self.max_dist)

        q_span = query[:, :eff_span, :]
        k_span = key[:, :eff_span, :]
        v_span = value[:, :eff_span, :]

        batch_size, seq_len, dims = q_span.size()
        scale_factor = (dims // self.head) ** -0.25

        q = q_span.view(batch_size, seq_len, self.head, -1).permute(0, 2, 1, 3)
        k = k_span.view(batch_size, seq_len, self.head, -1).permute(0, 2, 1, 3)
        v = v_span.view(batch_size, seq_len, self.head, -1).permute(0, 2, 1, 3)

        if self.sharpen:
            temperature = 1.0 + self.temp_scale * (1.0 - span_scale.mean().item())
        else:
            temperature = 0.5 + self.temp_scale * span_scale.mean().item()

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = torch.softmax((attn_scores / temperature) * scale_factor, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)

        return attn_out, attn_weights


class SpanPredictor(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.linear = nn.Linear(in_features=dims, out_features=1)

    def forward(self, global_out):
        scale = torch.sigmoid(self.linear(global_out))
        return scale

class AdaptiveSpan(nn.Module):
    def __init__(self, dims, heads, max_dist, sharpen, temp_scale=0.01):
        super().__init__()
        self.head = heads
        self.max_dist = max_dist
        self.dims = dims
        self.temp_scale = temp_scale

        self.span_scale = nn.Parameter(torch.tensor(1.0))
        self.sharpen = sharpen

    def forward(self, query, key, value, max_dist, max_span, span_scale):
        span_len = int(max_span * span_scale.mean().item())
        span_len = min(span_len, query.shape[1], key.shape[1], value.shape[1])
        eff_span = min(span_len, max_dist)
        
        q_span = query[:, :eff_span, :]
        k_span = key[:, :eff_span, :]
        v_span = value[:, :eff_span, :]
            
        batch_size, _, dims = query.shape
        scale = (dims // self.head) ** -0.25
      
        q = q_span.view(q_span.shape[0], q_span.shape[1], self.head, -1).permute(0, 2, 1, 3)
        k = k_span.view(k_span.shape[0], k_span.shape[1], self.head, -1).permute(0, 2, 1, 3)
        v = v_span.view(v_span.shape[0], v_span.shape[1], self.head, -1).permute(0, 2, 1, 3)
        
        with torch.autocast(device_type="cuda"):
            if self.sharpen:
                temperature = 1.0 + self.temp_scale * (1.0 - span_scale.mean().item())
            else:
                temperature = 0.5 + self.temp_scale * span_scale.mean().item()

            scores = torch.matmul(q, k.transpose(-2, -1))
            weights = torch.softmax((scores / temperature) * scale, dim=-1)
            out = torch.matmul(weights, v)
            out = out.permute(0, 2, 1, 3).flatten(start_dim=2)
            out = out.contiguous().view(batch_size, eff_span, dims)
        
        return out, weights

class FocusA(nn.Module):
    def __init__(self, dims, heads, max_dist, q_agent, sharpen, win_size, max_span):
        super().__init__()
        self.heads = heads
        self.max_dist = max_dist
        self.dims = dims
        self.max_span = max_span
        self.win_size = win_size
        self.sliding_window = win_size  
        self.q_agent = q_agent
        self.sharpen = sharpen
        self.span_pred = SpanPredictor(dims=dims)
        self.attn_local = AdaptiveSpan(dims=dims, heads=heads, max_dist=max_dist, 
                                      sharpen=sharpen, temp_scale=0.01)
        self.attn_global = MultiheadAttention(dims=dims, heads=heads, max_dist=max_dist)
        self.projection = nn.Linear(in_features=2 * dims, out_features=dims)

        self.ln_a = nn.LayerNorm(normalized_shape=dims)
        self.ln_b = nn.LayerNorm(normalized_shape=dims)

    def forward(self, x):
        local = self.ln_a(x)
        globe = self.ln_b(x)
        state = self.extract_state(local)
        action = self.q_agent.choose_action(state)
        span_scale = self.action_to_span_scale(action)

        globe_out, _ = self.attn_global(globe, globe, globe)

        span_scale = self.span_pred(globe_out.mean(dim=1))
        span_mean = span_scale.mean().item()

        current_win_size = max(1, int(self.sliding_window * span_mean))
        current_span_len = max(1, int(self.max_span * span_mean))
        
        effective_max = int(min(self.max_dist, local.size(1)))
        local_max = int(min(self.max_dist, current_span_len, current_win_size))
        globe_max = effective_max

        self.attn_local.max_dist = local_max
        self.attn_global.max_dist = globe_max

        local_out = self.slide_win(
            x=local,
            win_size=current_win_size, 
            span_len=current_span_len, 
            span_scale=span_scale  
        )
        
        combined = torch.cat(tensors=[local_out, globe_out], dim=-1)
        x = self.projection(combined)
        return x
    
    def extract_state(self, x):
        with torch.no_grad():
            state = x.mean(dim=(0, 1)).cpu().numpy()
            state_id = self.discretize_state(state)
        return state_id

    def discretize_state(self, state):
        bins = np.linspace(-1, 1, num=10)
        state_discrete = np.digitize(state, bins)
        state_id = int("".join(map(str, state_discrete)))
        return state_id

    def action_to_span_scale(self, action):
        num_actions = self.q_agent.Q.shape[1]
        span_scale_value = action / (num_actions - 1)
        span_scale = torch.tensor([span_scale_value])
        return span_scale
        
    def _focus(self, query, key, value, span_scale):
        max_iterations = 10
        iteration = 0
        prev_attn_out = torch.zeros_like(input=query)
        attn_out = torch.zeros_like(input=query) 
        attn_weights = None  

        base_threshold = 1e-4
        scaling_factor = 0.1

        while iteration < max_iterations:
            span_len = int(self.max_span * span_scale.mean().item())
            span_len = min(span_len, query.size(1), key.size(1), value.size(1))
            eff_span = min(span_len, self.max_dist)

            if eff_span == 0:
                break

            q_span = query[:, :eff_span, :]
            k_span = key[:, :eff_span, :]
            v_span = value[:, :eff_span, :]

            batch_size, seq_len, dims = q_span.size()
            d_k = dims // self.head
            scale_factor = 1 / math.sqrt(d_k)

            q = q_span.view(batch_size, seq_len, self.head, -1).transpose(1, 2)
            k = k_span.view(batch_size, seq_len, self.head, -1).transpose(1, 2)
            v = v_span.view(batch_size, seq_len, self.head, -1).transpose(1, 2)

            if self.sharpen:
                temperature = 1.0 + self.temp_scale * (1.0 - span_scale.mean().item())
            else:
                temperature = 0.5 + self.temp_scale * span_scale.mean().item()

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor / temperature
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            diff = torch.abs(attn_out - prev_attn_out).mean()
            dynamic_threshold = base_threshold + scaling_factor * diff

            if diff < dynamic_threshold:
                break

            prev_attn_out = attn_out
            query = query + attn_out  
            iteration += 1 
        return attn_out, attn_weights
        
    def slide_win(self, x, win_size, span_len, span_scale):
        self.batch_size, seq_len, self.dims = x.size()
        num_windows = (seq_len + win_size - 1) // win_size

        output = torch.zeros_like(x)

        for i in range(num_windows):
            start_idx = i * win_size
            end_idx = min((i + 1) * win_size, seq_len)
            query = x[:, start_idx:end_idx, :]

            key_start = max(0, start_idx - span_len + win_size)
            key_end = min(start_idx + span_len, seq_len)
            key = x[:, key_start:key_end, :]
            value = x[:, key_start:key_end, :]

            attn_out, _ = self._focus(query=query, key=key, value=value, span_scale=span_scale)
            output[:, start_idx:end_idx, :] = attn_out

        return output
    

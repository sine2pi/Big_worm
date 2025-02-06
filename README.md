### What is it:

An approach to building an attention mechanism that can adapt its focus based on the input sequence. It combines the strengths of local and global attention while incorporating mechanisms for efficiency and dynamic adaptation. 


### Why it's a good idea: 
Addresses a common challenge in sequence modeling, i.e., balancing local and global dependencies. 


### How it works:

**Hybrid Attention with Adaptive, Sharpened Local Focus**

This model combines global self-attention with a novel local attention mechanism that features an *adaptive span* and *span-dependent sharpness*.  It achieves both fine-grained focus and broad contextual awareness efficiently.

1.  **Global Context for Span Prediction:** The `HybridAttention` module first performs global self-attention over the entire input sequence.  The *mean* of the global attention output (across the sequence length) is used as a condensed representation of the overall context. This context vector is fed into a `SpanPredictor` (a simple linear layer + sigmoid) to predict a `span_scale` value between 0 and 1.

2.  **Adaptive and Sharpened Local Attention (Sliding Window):**
    *   **Sliding Window:** The local attention operates within a sliding window that moves across the input sequence.
    *   **Adaptive Span:** The `span_scale` predicted in step 1 determines the *length* of the attention span within each window.  A smaller `span_scale` means a shorter span (focusing on fewer tokens), and a larger `span_scale` means a longer span.
    *   **Span-Dependent Sharpness (Temperature Scaling):**  Critically, the *sharpness* of the local attention is also dynamically adjusted based on the `span_scale`.  A *smaller* `span_scale` (shorter span) results in a *lower* temperature for the softmax calculation, leading to *sharper* attention weights (more focused on a few key tokens within the span).  A *larger* `span_scale` (longer span) results in a *higher* temperature, leading to *softer* attention weights (more distributed across the tokens within the span). This ensures that when the model focuses on a small region, it does so with high precision.
    * **How it all works together** The global attention gives the model a general sense of where important information will be in the local window. If the model decides based on that global information, that local information is all that is required the local span will become smaller, temperature will go down, and the weights will be sharp. If it decides the oposite the span will get larger, the temperature will go up, and the weights will become softer.

3.  **Efficiency:** The sliding window approach breaks down the local attention computation into smaller chunks, making it significantly more efficient than applying full self-attention to the entire sequence, especially for long sequences. This reduces both computational cost and memory footprint.

4.  **Balance:** By combining global and local attention, the model captures both short-range and long-range dependencies. The adaptive span and sharpness mechanisms allow the model to dynamically adjust its focus, attending precisely to relevant local details while still maintaining awareness of the overall context. This provides a balance between fine-grained analysis and broad understanding.
          
          
          class AdaptiveSpanAttention(nn.Module):
              def __init__(self, base, dims, head, max_dist, win_size, max_span, temp_scale=0.01):
                  super().__init__()
          
                  self.max_dist = max_dist
                  self.win_size = win_size
                  self.max_span = max_span
                  self.temp_scale = temp_scale
                  self.multihead_attn = MultiheadAttention(base, dims, head, max_dist)
                  self.span_scale = nn.Parameter(torch.tensor(1.0))
          
              def forward(self, query, key, value, span_scale):
                  span_len = int(self.max_span * span_scale.mean().item())
                  span_len = min(span_len, query.shape[1], key.shape[1], value.shape[1])
                  eff_span = min(span_len, self.max_dist)
          
                  q_span = query[:, :eff_span, :]
                  k_span = key[:, :eff_span, :]
                  v_span = value[:, :eff_span, :]
          
                  attn_out, attn_weights = self.multihead_attn(q_span, k_span, v_span)
          
                  # FINAL CORRECTED TEMPERATURE SCALING:
                  temperature = 0.5 + self.temp_scale * span_scale.mean().item()
          
                  batch_size, _, dims = query.shape
                  scale = (dims // self.multihead_attn.head) ** -0.25
          
                  q = q_span.view(q_span.shape[0], q_span.shape[1], self.multihead_attn.head, -1).permute(0, 2, 1, 3)
                  k = k_span.view(k_span.shape[0], k_span.shape[1], self.multihead_attn.head, -1).permute(0, 2, 1, 3)
                  v = v_span.view(v_span.shape[0], v_span.shape[1], self.multihead_attn.head, -1).permute(0, 2, 1, 3)
          
                  attn_scores = torch.matmul(q, k.transpose(-2, -1))
                  attn_weights = torch.softmax((attn_scores / temperature) * scale, dim=-1)
                  attn_out = torch.matmul(attn_weights, v)
                  attn_out = attn_out.permute(0, 2, 1, 3).flatten(start_dim=2)
          
                  # Corrected final reshaping:
                  attn_out = attn_out.contiguous().view(batch_size, eff_span, dims)
          
                  return attn_out, attn_weights
          
          class SpanPredictor(nn.Module):
              def __init__(self, dims):
                  super().__init__()
                  self.linear = nn.Linear(dims, 1)
          
              def forward(self, global_out):
                  scale = torch.sigmoid(self.linear(global_out))
                  return scale
          
          
          class HybridAttention(nn.Module):
              def __init__(self, base, dims, head, max_dist, win_size=32, max_span=32, slid_win=32):
                  super().__init__()
                  self.max_dist = max_dist
                  self.win_size = win_size
                  self.max_span = max_span
                  self.slid_win = slid_win
          
                  self.span_pred = SpanPredictor(dims)
                  self.dist_local = max_dist
                  self.dist_global = max_dist
                  self.attn_local = AdaptiveSpanAttention(base, dims, head, self.dist_local, win_size, max_span)
                  self.attn_global = MultiheadAttention(base, dims, head, self.dist_global)
                  self.ln_local = LayerNorm(dims)
                  self.ln_global = LayerNorm(dims)
                  self.projection = Linear(2 * dims, dims)
          
              def forward(self, x, new_dist=None, new_base=None, xa=None, mask=None, kv_cache=None):
          
                  local = self.ln_local(x)
                  globe = self.ln_global(x)
          
                  globe_out, _ = self.attn_global(globe, globe, globe)
          
                  span_scale = self.span_pred(globe_out.mean(dim=1))
          
                  win_size = max(1, int(self.slid_win * span_scale.mean().item()))
                  span_len = max(1, int(self.max_span * span_scale.mean().item()))
          
                  effective_max_dist = min(self.max_dist, local.size(1))
                  local_max_dist = min(self.dist_local, span_len, win_size)
                  globe_max_dist = effective_max_dist
          
                  # DYNAMICALLY UPDATE max_dist:
                  self.attn_local.max_dist = local_max_dist
                  self.attn_global.max_dist = globe_max_dist
          
                  local_out = self.slide_win(local, win_size, span_len, span_scale)
          
                  combined = torch.cat([local_out, globe_out], dim=-1)  
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
                      attn_out, _ = self.attn_local(query, key, value, span_scale)
                      out[:, i:end, :] = attn_out 
                  return out

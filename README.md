## Focused Attention ##
### Adaptive Span Attention and Span Predictor

**Role of SpanPredictor:**
- The SpanPredictor uses the global attention output to guide the local attention mechanism.
- The scale value (between 0 and 1) produced by the SpanPredictor directly influences the local attention window size.

**Effect of Small Scale Values:**
- A smaller scale reduces the local attention window size.
- This means local attention will focus on fewer tokens, informed by the global context.

**Why Reduce the Local Attention Window?**
- **Focus on Key Information:** Helps concentrate computational resources on crucial regions identified by the global attention.
- **Reduce Noise:** Limits attention to a smaller context, filtering out irrelevant tokens.
- **Computational Efficiency:** Smaller windows require fewer computations, optimizing resources for long sequences.

**Analogy:**
- Global attention is like understanding the theme of a sentence.
- Local attention focuses on individual words and immediate neighbors, guided by the global context.

### Enhancements: Scaling and Sharpening Attention

1. **Scaling Attention Scores:**
   - Dynamically scale attention scores inversely with span_scale:
     ```python
     attn_weights = attn_weights * (1.0 / span_scale)
     attn_output = torch.bmm(attn_weights, value_span)
     ```

2. **Sharpening Attention:**
   - Apply temperature scaling to soften or sharpen attention distribution:
     ```python
     attn_weights = torch.softmax(attn_scores / temperature, dim=-1)
     ```
   - Dynamic temperature adjustment based on span_scale:
     ```python
     temperature = 1.0 - span_scale  # Lower temperature sharpens attention
     ```
##
Scaling Attention Scores:
Scale up the attention scores within the local attention mechanism inversely with the span_scale to increase focus on the remaining tokens within the smaller window.

Sharpening Attention:
Increase the contrast between high and low attention scores by applying a sharpening function or using temperature scaling in the softmax function.

Temperature Scaling:
Introduce a temperature parameter in the softmax function, where lower values result in sharper attention.

Dynamic Temperature:
Make the temperature adjust dynamically based on the span_scale, decreasing as the scale decreases to sharpen attention.

Interaction with max_rel_dist:
Ensure that max_rel_dist does not exceed the dynamically adjusted span_length to prevent out-of-bounds errors and ensure correct application of relative positional biases.

##
### Key Considerations

- **Over-Sharpening:** Be cautious of too sharp attention distributions.
- **Effective Span Calculation:** Ensure effective_span respects the input tensor's lengths and max_rel_dist to avoid out-of-bounds errors.
- **Evaluation:** Continuously assess changes through visualizing attention weights, tracking WER, and other relevant metrics.

### Max_rel_dist and Positional Bias

- **Interaction with max_rel_dist:** Ensure max_rel_dist does not exceed the dynamically adjusted span_length.
  ```python
  effective_span = min(span_length, self.max_rel_dist)
  ```

### Potential Refinements

- **Explicit Max_rel_dist Setting:** Set max_rel_dist within the `AdaptiveSpanAttention.forward` method for clarity.

### Conclusion

By integrating these scaling and sharpening techniques, you enhance the local attention mechanism dynamically based on global context, making the model more focused and efficient. Extensive experimentation will be crucial to fine-tune these parameters for optimal performance.

  ```python

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
 ```


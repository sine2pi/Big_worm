<img width="925" alt="eval" src="https://github.com/user-attachments/assets/53497866-1df2-427e-a411-83dc12f2592c" />

<img width="925" alt="train" src="https://github.com/user-attachments/assets/955c0cb8-4022-4ee0-9ffa-887e50278d1e" />



<img width="200" alt="legend2" src="https://github.com/user-attachments/assets/1ac69ffd-5e19-49c9-8607-3d4b7a3b8edf" />



Pilot run. A sanity check to see if the models could learn. Attention heads on Echo 8 Whisper 16. Note the runtime changes with the focused attention hybrid. 


1. Long-Range Dependencies and Specificity:

Scenario: Imagine a task involving long documents where you need to identify very specific pieces of information scattered throughout the text. For instance, answering questions about a legal document or summarizing a complex scientific paper.

Reasoning: When the attention span is long, you're allowing the model to consider a wide range of context. In this case, you might actually want the attention to be sharper. You don't want the model to be wishy-washy and distribute its attention equally across a large number of tokens. You want it to pinpoint the few most relevant pieces of information within that broad context. A softer attention (higher temperature) over a long span would likely lead to a diluted, less informative representation.

Example: If the question is "What is the defendant's age in Case 3.14159?", and Case 3.14159 spans several paragraphs, you'd want the model to sharply focus on the specific sentence mentioning the age, even within that large span.

2. Avoiding "Attention Collapse" with Long Spans:

Scenario: With very long spans, standard (or softly scaled) attention can sometimes suffer from a phenomenon where the attention weights become too uniform. The model essentially "gives up" on trying to discriminate between tokens and attends to everything equally.

Reasoning: A sharper softmax (lower temperature) can act as a regularizer, preventing this "attention collapse." It forces the model to make more decisive choices, even when the context is large.

Analogy: Think of it like searching a large library. If you have no idea where to look (soft attention), you might just wander aimlessly. A sharper focus (even if you don't know exactly where to go) forces you to pick specific shelves and sections to examine, increasing your chances of finding what you need.

3. Tasks Requiring Precise Identification within Broad Context:
   
Scenario: Tasks like named entity recognition (NER) or relation extraction, when applied to long documents.

Reasoning: You might need a broad context (long span) to understand the relationships between entities, but you still need to precisely identify the entities themselves (which might be short phrases). Softer attention over a long span might blur the boundaries of the entities, making it harder to extract them accurately.

4. Hierarchical Reasoning:
   
Scenario: Imagine a multi-step reasoning task, where the model needs to first identify relevant sections of a document (long span, sharper attention) and then analyze those sections in more detail (shorter spans, possibly softer attention).

Reasoning: you might want a different temperature scaling approach that is learnable.

5. Sparsity Inducement
   
Scenario: If the model were to be deployed on low power devices.

Reasoning: You want to create as sparse of a weight distribution as possible, and this is done by a lower temperature.

#### There are reasonings behind why one might want the opposite to be true when it comes to focus and that can be changed with a toggle sharpen_longer=False in your model config.
                  
      class Adaptivefocus(nn.Module):
          def __init__(self, base, dims, head, max_dist, sharpen, win_size, max_span, temp_scale=0.01, num_iterations=3):
              super().__init__()
              self.max_dist = max_dist
              self.win_size = win_size
              self.max_span = max_span
              self.temp_scale = temp_scale
              self.multihead_attn = MultiheadAttention(base=base, dims=dims, head=head, max_dist=max_dist)
              self.span_scale = nn.Parameter(torch.tensor(1.0))
              self.sharpen = sharpen
              self.num_iterations = num_iterations
              self.base_threshold = 1e-4
              self.scaling_factor = 0.1
      
          def _focus(self, query, key, value, span_scale):
              max_iterations = self.num_iterations
              iteration = 0
              prev_attn_out = torch.zeros_like(query)
      
              while iteration < max_iterations:
                  span_len = int(self.max_span * span_scale.mean().item())
                  span_len = min(span_len, query.shape[1], key.shape[1], value.shape[1])
                  eff_span = min(span_len, self.max_dist)
      
                  if eff_span == 0:
                      break
      
                  q_span = query[:, :eff_span, :]
                  k_span = key[:, :eff_span, :]
                  v_span = value[:, :eff_span, :]
      
                  batch_size, seq_len, dims = q_span.size()
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
      
                  diff = torch.abs(attn_out - prev_attn_out).mean()
      
                  dynamic_threshold = self.base_threshold + self.scaling_factor * diff
      
                  if diff < dynamic_threshold:
                      break
      
                  prev_attn_out = attn_out
                  query = query + attn_out  # Residual connection
                  iteration += 1  # Increment iteration
      
              return attn_out, attn_weights
      
          def forward(self, query, key, value, span_scale):
              return self._focus(query, key, value, span_scale)
      
      
      class SpanPredictor(nn.Module):
          def __init__(self, dims):
              super().__init__()
              self.linear = nn.Linear(in_features=dims, out_features=1)
      
          def forward(self, global_out):
              scale = torch.sigmoid(self.linear(global_out))
              return scale
      
      
      class FocusedAttention(nn.Module):
          def __init__(self, base, dims, head, max_dist, sharpen, win_size=32, max_span=32, slid_win=32, temp_scale=0.01, num_iterations=3):
              super().__init__()
              self.max_dist = max_dist
              self.win_size = win_size
              self.max_span = max_span
              self.slid_win = slid_win
      
              self.span_pred = SpanPredictor(dims=dims)
              self.dist_local = max_dist
              self.dist_global = max_dist
      
              self.attn_local = Adaptivefocus(base=base, dims=dims, head=head, max_dist=max_dist, sharpen=sharpen, win_size=win_size, max_span=max_span, temp_scale=temp_scale, num_iterations=num_iterations)
              self.attn_global = MultiheadAttention(base=base, dims=dims, head=head, max_dist=self.dist_global)
              self.ln_local = LayerNorm(normalized_shape=dims)
              self.ln_global = LayerNorm(normalized_shape=dims)
              self.projection = Linear(in_features=2 * dims, out_features=dims)
      
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
      
              self.attn_local.max_dist = local_max_dist
              self.attn_global.max_dist = globe_max_dist
      
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
                  attn_out, _ = self.attn_local(query, key, value, span_scale)
                  out[:, i:end, :] = attn_out
      
              return out


#### Different version:
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


              

###Batch: 

    def _window(self, x, win_size, span_len, span_scale):
        batch_size, seq_len, dims = x.size()
        num_windows = (seq_len + win_size - 1) // win_size  # Calculate the number of windows
    
        # Create tensors to store the outputs
        output = torch.zeros_like(x, device=x.device)
    
        # Iterate over the windows in a more efficient manner
        for i in range(num_windows):
            start_idx = i * win_size
            end_idx = min((i + 1) * win_size, seq_len)
            query = x[:, start_idx:end_idx, :]
    
            # Define the range of keys and values
            key_start = max(0, start_idx - span_len + win_size)
            key_end = min(start_idx + span_len, seq_len)
            key = x[:, key_start:key_end, :]
            value = x[:, key_start:key_end, :]
    
            attn_out, _ = self._focus(query, key, value, span_scale)
            output[:, start_idx:end_idx, :] = attn_out
    
        return output

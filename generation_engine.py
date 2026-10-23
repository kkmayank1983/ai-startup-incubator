import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters for Stacked Transformer-lite Architecture
n_embd = 256      # Embedding dimension (size of vector representations)
block_size = 256  # Maximum sequence length (model's memory context window)
n_layer = 6       # Number of stacked Transformer blocks
n_head = 8        # Number of parallel Self-Attention heads
dropout = 0.1     # Dropout rate to prevent overfitting by randomly disabling nodes

# PARAMETER DETAILS: Currently configured for scaled capacity: 5.57M trainable parameters 
# (8.72M total state_dict elements including causal mask buffers).
# Base capacity model used: n_embd=128, n_layer=4, n_head=4 (~1.08M trainable, 2.13M state_dict).

class SelfAttentionHead(nn.Module):
    """
    A single head of self-attention. Computes the relationship (affinity)
    between all tokens in the context window.
    """
    def __init__(self, head_size):
        super().__init__()
        # Projections to calculate Query, Key, and Value vectors
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Register causal mask (tril) to prevent looking at future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # Batch, Time (sequence length), Channels (embedding size)
        k = self.key(x)   # Query what information this token contains
        q = self.query(x) # Key what information this token is looking for
        
        # 1. Calculate raw attention affinity scores: Q * K^T / sqrt(d_k)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        # 2. Mask the future: fill upper triangle of affinity matrix with negative infinity
        # This ensures the model cannot "cheat" by looking ahead during pre-training
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # 3. Softmax converts scores into a neat percentage distribution (weights sum to 1)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # 4. Multiply attention weights by Value vectors to pull the relevant data
        v = self.value(x) 
        out = wei @ v     
        return out

class MultiHeadAttention(nn.Module):
    """
    Runs multiple SelfAttentionHead instances in parallel. 
    Allows the model to focus on different parts of the sentence at the same time.
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
        # Projection layer to combine head outputs back to embedding dimension size
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate outputs from all attention heads along the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linear activation (ReLU) and another linear layer.
    Allows the model to process attention outputs and store factual patterns in its weights.
    """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    A single block containing Multi-Head Attention, Feed-Forward, and LayerNorm.
    Uses Pre-LayerNorm and Residual Connections to prevent vanishing gradients.
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # Normalizes inputs to attention
        self.ln2 = nn.LayerNorm(n_embd) # Normalizes inputs to Feed-Forward

    def forward(self, x):
        # Residual Addition: adding input x back to output (the gradient highway!)
        x = x + self.sa(self.ln1(x))   # Attention Sub-Layer
        x = x + self.ffwd(self.ln2(x)) # Feed-Forward Sub-Layer
        return x

class PredictionEngine(nn.Module):
    """
    The full Transformer-lite model. Combines token embeddings, positional 
    tracking, and stacked blocks to output word probabilities (logits).
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Stack multiple Transformer blocks sequentially
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head=n_head) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd) # Final Layer Normalization
        self.lm_head = nn.Linear(n_embd, vocab_size) # Maps embeddings to vocabulary probabilities

    def forward(self, idx):
        B, T = idx.shape
        # Convert token IDs to vectors
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        # Convert position indices (0 to T-1) to vectors to give the model a sense of word order
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        
        x = tok_emb + pos_emb # Combine identity vectors with positional vectors
        x = self.blocks(x)    # Pass through the stacked Transformer blocks
        x = self.ln_f(x)      
        logits = self.lm_head(x) # Output raw scores for each token in vocab
        return logits

    def generate(self, idx, max_new_tokens):
        """
        Generates new text token-by-token using autoregressive prediction.
        """
        for _ in range(max_new_tokens):
            # Crop context history to block size limit (128 tokens)
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            # Focus only on the prediction scores of the last token
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            # Sample next token from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append predicted token ID to current context history
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
              
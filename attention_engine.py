import torch
import torch.nn as nn
from torch.nn import functional as F
from data_pipeline import vocab_size, stoi

# Small, real numbers. Session 6 grows these up.
n_embd = 256        # numbers used to describe each token
block_size = 64    # how many tokens the AI can see at once
batch_size = 64    # how many windows get trained on together, per step
n_head = 4         # how many heads of attention the AI has

# Where the maths happens. A GPU does thousands of small sums at the same
# time; a CPU does a few at a time. Same code, same model, same answers -
# it just gets there faster.
device = "cuda" if torch.cuda.is_available() else "cpu"

# The token that ends a line. We use .get() rather than stoi["\n"]
# because a word-level tokenizer has no newline token at all -
# .get() returns None instead of crashing.
NEWLINE = stoi.get("\n")

class SelfAttentionHead(nn.Module):
    """One Attention Head. Learns its own Query, Key and Value
    numbers, instead of us picking them by hand like the T/H/E
    example. Does exactly those 6 steps, for every token
    against every token, all at once."""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # STEP 4's "no cheating" mask
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)     # STEP 1
        q = self.query(x)   # STEP 1
        v = self.value(x)   # STEP 1

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # STEP 2 + 3

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # STEP 4

        wei = F.softmax(wei, dim=-1)  # STEP 5

        # Save this so the Focus Map can look at it later
        self.last_weights = wei.detach()

        return wei @ v  # STEP 6


class MultiHeadAttention(nn.Module):
    """Several Attention Heads running side by side.

    Each head gets a SMALLER slice of the numbers to work with, and
    goes hunting for its own kind of relationship. We stick their
    answers back together at the end."""
    def __init__(self, num_heads, head_size):
        super().__init__()

        # Build num_heads copies of the head you already wrote.
        self.heads = nn.ModuleList(
            [SelfAttentionHead(head_size) for _ in range(num_heads)]
        )

        # Sticking together is not the same as mixing together.
        # This layer lets what the heads found blend into each other.
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # Each head hands back head_size numbers per token.
        # 4 heads x 24 numbers = the 96 we started with.
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        return self.proj(out)


class FeedForward(nn.Module):
    """Somewhere for the AI to actually think about what
    attention gathered, instead of just blending it."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class PredictionEngine(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)   # tidy up before attention
        self.ln2 = nn.LayerNorm(n_embd)   # tidy up before thinking
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                # what is this token?
        #pos_emb = self.position_embedding_table(torch.arange(T)) # where does it sit?
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device))     # where does it sit?
        x = tok_emb + pos_emb                                    # combine both cards

        x = x + self.sa_head(self.ln1(x))   # focus, then add the result back in
        x = x + self.ffwd(self.ln2(x))      # think about it, then add that back in too

        logits = self.lm_head(x)  # turn it back into a guess
        return logits

    def generate(self, idx, max_new_tokens, stop_after_lines=None):
        """Writes one token at a time, exactly like Session 3.
        If stop_after_lines is given, it keeps going until it has
        finished that many lines."""
        lines_done = 0
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if stop_after_lines is not None and NEWLINE is not None and idx_next.item() == NEWLINE:
                lines_done += 1
                if lines_done >= stop_after_lines:
                    break
        return idx


#model = PredictionEngine(vocab_size)
model = PredictionEngine(vocab_size).to(device)

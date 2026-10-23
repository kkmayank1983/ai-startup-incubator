import torch
import torch.nn as nn
from torch.nn import functional as F
from data_pipeline import vocab_size, stoi

# Small, real numbers. Session 6 grows these up.
n_embd = 96        # numbers used to describe each character
block_size = 64    # how many characters the AI can see at once
batch_size = 64    # how many windows get trained on together, per step

# Session 5 was character-level, so "\n" was a token of its own.
# Session 6 splits on words, so there is no newline token any more.
NEWLINE = stoi.get("\n")   # None once the tokenizer is word-level

class SelfAttentionHead(nn.Module):
    """One Attention Head. Learns its own Query, Key and Value
    numbers, instead of us picking them by hand like the T/H/E
    example. Does exactly those 6 steps, for every character
    against every character, all at once."""
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

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = SelfAttentionHead(n_embd)
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                # what is this character?
        pos_emb = self.position_embedding_table(torch.arange(T)) # where does it sit?
        x = tok_emb + pos_emb                                    # combine both cards

        x = x + self.sa_head(x)   # focus, then add the result back in
        x = x + self.ffwd(x)      # think about it, then add that back in too

        logits = self.lm_head(x)  # turn it back into a guess
        return logits

    def generate(self, idx, max_new_tokens, stop_after_lines=None):
        """Writes one character at a time, exactly like Session 3.
        If stop_after_lines is given, it keeps going until it has
        finished that many lines - so you get whole lines of the
        play instead of text that stops in the middle of a word."""
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


model = PredictionEngine(vocab_size)
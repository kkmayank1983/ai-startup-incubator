import torch
import torch.nn as nn
import torch.nn.functional as F

from data_pipeline import data, vocab_size, decode

# =========================================================
#  THE ONLY LINE YOU CHANGE.  Run with 1, then 2, then 4.
# =========================================================
CONTEXT_SIZE = 4
# =========================================================


class ContextModel(nn.Module):
    """Looks at CONTEXT_SIZE characters and guesses the next one."""

    def __init__(self):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, 24)
        self.hidden = nn.Linear(CONTEXT_SIZE * 24, 128)
        self.output = nn.Linear(128, vocab_size)

    def forward(self, idx):
        e = self.embed(idx)              # each character becomes 24 numbers
        e = e.view(idx.shape[0], -1)     # lay the whole window out in one row
        h = torch.tanh(self.hidden(e))   # think about them all together
        return self.output(h)            # score every possible next character


torch.manual_seed(1337)
model = ContextModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print(f"Training with CONTEXT_SIZE = {CONTEXT_SIZE} ...")

running_loss = 0
average_loss = 0

for step in range(1, 6001):

    # Grab 64 random windows from the book
    ix = torch.randint(0, len(data) - CONTEXT_SIZE - 1, (64,))
    x = torch.stack([data[i : i + CONTEXT_SIZE] for i in ix])   # the context
    y = torch.stack([data[i + CONTEXT_SIZE] for i in ix])       # the next character

    loss = F.cross_entropy(model(x), y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    if step % 1500 == 0:
        average_loss = running_loss / 1500
        print(f"  step {step:5d} | average loss {average_loss:.3f}")
        running_loss = 0

print(f"\n>>> FINAL LOSS with CONTEXT_SIZE = {CONTEXT_SIZE}  is  {average_loss:.3f}")


# --- Now let it write ---
print(f"\n===== SHAKESPEARE, WRITTEN WITH CONTEXT_SIZE = {CONTEXT_SIZE} =====\n")

window = [0] * CONTEXT_SIZE
result = []

for _ in range(300):
    xb = torch.tensor([window[-CONTEXT_SIZE:]], dtype=torch.long)
    probs = F.softmax(model(xb), dim=-1)
    next_char = torch.multinomial(probs, num_samples=1).item()
    result.append(next_char)
    window.append(next_char)

print(decode(result))
              
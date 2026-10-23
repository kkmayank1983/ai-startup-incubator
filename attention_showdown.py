import torch
import torch.nn as nn
import torch.nn.functional as F

# Seeded before attention_engine is imported, so the model's
# starting weights are controlled too, not just the training batches.
torch.manual_seed(1337)

from data_pipeline import data, vocab_size, decode, encode, stoi
from attention_engine import model, block_size, batch_size

STEPS = 2000

# How big a nudge each training step gives the weights.
# Session 4 used 1e-3. Attention can safely take bigger steps
# than the flat model - Section 7 explains why we raised it.
LEARNING_RATE = 3e-3

def count_params(m):
    return sum(p.numel() for p in m.parameters())

# ----------------------------------------------------
# CONTENDER 1: Session 4's flat model, unchanged
# ----------------------------------------------------
class ContextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, 24)
        self.hidden = nn.Linear(block_size * 24, 128)
        self.output = nn.Linear(128, vocab_size)

    def forward(self, idx):
        e = self.embed(idx)
        e = e.view(idx.shape[0], -1)
        h = torch.tanh(self.hidden(e))
        return self.output(h)

torch.manual_seed(1337)
flat = ContextModel()
opt_flat = torch.optim.AdamW(flat.parameters(), lr=LEARNING_RATE)

print(f"Flat model params      : {count_params(flat):,}")
print(f"Attention model params : {count_params(model):,}")
print(f"Attention uses {count_params(flat)/count_params(model):.2f}x fewer numbers\n")

print("Training the FLAT model (Session 4)...")
running = 0.0
for step in range(1, STEPS + 1):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + block_size] for i in ix])
    loss = F.cross_entropy(flat(x), y)
    opt_flat.zero_grad()
    loss.backward()
    opt_flat.step()
    running += loss.item()
    if step % 500 == 0:
        print(f"  [flat] step {step:5d} | average loss {running/500:.3f}")
        running = 0.0

# ----------------------------------------------------
# CONTENDER 2: your new Attention model
# ----------------------------------------------------
torch.manual_seed(1337)
opt_attn = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("\nTraining the ATTENTION model (today)...")
running = 0.0
for step in range(1, STEPS + 1):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    opt_attn.zero_grad()
    loss.backward()
    opt_attn.step()
    running += loss.item()
    if step % 500 == 0:
        print(f"  [attn] step {step:5d} | average loss {running/500:.3f}")
        running = 0.0

# ----------------------------------------------------
# Generate from the trained Attention model
# ----------------------------------------------------
# Character 0 is a newline, so this starts it at the beginning of a line.
# stop_after_lines makes it write whole lines instead of cutting off
# in the middle of a word.
context = torch.zeros((1, 1), dtype=torch.long)
sample = model.generate(context, max_new_tokens=500, stop_after_lines=12)[0].tolist()
print("\n" + "=" * 55)
print("YOUR ATTENTION MODEL, WRITING SHAKESPEARE")
print("=" * 55)
print(decode(sample))

# ----------------------------------------------------
# Peek at what it is actually focusing on
# ----------------------------------------------------
sentence = "ROMEO:"
ids = torch.tensor([[stoi[c] for c in sentence]])
model(ids)
weights = model.sa_head.last_weights[0]

print("\nFocus Map for 'ROMEO:'\n")
for row, ch in enumerate(sentence):
    print(f"{ch!r:>6} -> ", end="")
    for col in range(row + 1):
        print(f"{weights[row, col].item()*100:5.1f}% ", end="")
    print()

# ----------------------------------------------------
# Talk to it - no second training run needed
# ----------------------------------------------------
print("\n" + "=" * 55)
print("TALK TO YOUR ATTENTION MODEL  (type  quit  to stop)")
print("=" * 55)

while True:
    prompt = input("\nYou: ")
    if prompt.lower() == "quit":
        break

    clean = ''.join(c for c in prompt if c in stoi)
    if clean == "":
        print("   (nothing left for the AI to read - letters only)")
        continue

    window = [0] * block_size + encode(clean)
    ids = torch.tensor([window[-block_size:]], dtype=torch.long)
    result = model.generate(ids, max_new_tokens=400, stop_after_lines=8)[0].tolist()

    print("\nAI: " + clean + decode(result[block_size:]))
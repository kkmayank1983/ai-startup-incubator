import torch
import torch.nn.functional as F
import time


# Same seed as before, so a re-run gives the same model.
torch.manual_seed(1337)

from attention_engine import model, block_size, batch_size, device
from data_pipeline import data, vocab_size, decode, encode
#from attention_engine import model, block_size, batch_size

STEPS = 14000
LEARNING_RATE = 3e-3   # pieces are small, so bigger nudges are safe (Session 5)

print(f"Vocabulary   : {vocab_size} pieces")
print(f"Dataset      : {len(data)} tokens")
#print(f"Model params : {sum(p.numel() for p in model.parameters()):,}")
print(f"Model params : {sum(p.numel() for p in model.parameters()):,}")
seen = STEPS * batch_size * block_size
print(f"Will read    : {seen:,} tokens = {seen/len(data):.1f} passes "
      f"over the book")
print(f"Running on   : {device}")

# The whole book moves onto the GPU once. Every batch is then cut from it
# there, so nothing has to be copied across during training.
data = data.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("\nTraining on the question-and-answer data...")
running = 0.0
t_start = time.perf_counter()
for step in range(1, STEPS + 1):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running += loss.item()
    if step % 500 == 0:
        done = time.perf_counter() - t_start
        left = done / step * (STEPS - step)
        print(f"  step {step:5d} | average loss {running/500:.3f} "
              f"| {left/60:4.1f} min left")


# --- Save it under a NEW name, so last session's model survives ---
torch.save(model.state_dict(), "qa_model_v5.pth")
print("Saved qa_model_v5.pth")

# --- Ask it something straight away ---
# Laid out exactly like every block in input.txt. Now that "\n" is a
# real token again, the prompt can look the same as the book.
prompt = "<|user|>\nWhat is photosynthesis?\n<|assistant|>\n"
ids = encode(prompt)
#out = model.generate(torch.tensor([ids]), max_new_tokens=120)[0].tolist()
out = model.generate(torch.tensor([ids], device=device), max_new_tokens=120)[0].tolist()
answer = decode(out[len(ids):])
for tag in ("<|endoftext|>", "<|user|>", "<|assistant|>"):   # stop at the first signpost
    answer = answer.split(tag)[0]

print("\n" + "=" * 55)
print("YOUR MODEL, ANSWERING A QUESTION")
print("=" * 55)
print(answer.strip())

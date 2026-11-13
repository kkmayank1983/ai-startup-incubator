import sys
import time
import torch
import torch.nn.functional as F

# Same seed as before, so a re-run gives the same model.
torch.manual_seed(1337)

from attention_engine import model, block_size, batch_size, device
from data_pipeline import data, vocab_size, stoi, encode, decode


#     python train_qa.py          -> the full 14,000 steps (about 2 hours on a T4)
#     python train_qa.py 300      -> a short test run
# STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 14000
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 14000
LEARNING_RATE = 5e-4

# ---------------------------------------------------------------
# SESSION 4, CHANGE 2: carry on from where v6 stopped.
# v6 has read each fact about six times. That is not enough to learn
# 38,000 of them, so instead of starting from zero we load v6 and let
# it keep reading. Set START_FROM = None to train from scratch again.
# ---------------------------------------------------------------
START_FROM = "qa_model_v6.pth"
SAVE_AS = "qa_model_v7.pth"
SAVE_EVERY = 2000      # save a copy every 2,000 steps, so a lost Colab session keeps its progress

# ---------------------------------------------------------------
# SESSION 4, CHANGE 1: the red pen only marks the answer.
# Walk through the whole book once and mark every token that belongs
# to a tutor reply. Count how many <|assistant|> tags have gone past,
# and how many <|endoftext|> tags. If more replies have STARTED than
# have ENDED, we are inside a reply.
# ---------------------------------------------------------------
ASSISTANT = stoi["<|assistant|>"]
END = stoi["<|endoftext|>"]

replies_started = torch.cumsum((data == ASSISTANT).int(), 0)
replies_ended = torch.cumsum((data == END).int(), 0)
# Count the ends BEFORE each token, not up to it, so the <|endoftext|>
# tag itself is still graded. The model has to learn when to stop.
replies_ended = torch.cat([torch.zeros(1, dtype=replies_ended.dtype), replies_ended[:-1]])

answer_mask = (replies_started > replies_ended) & (data != ASSISTANT)

print(f"Vocabulary   : {vocab_size} pieces")
print(f"Dataset      : {len(data)} tokens")
print(f"Graded       : {answer_mask.float().mean():.0%} of them (the answers)")
print(f"Model params : {sum(p.numel() for p in model.parameters()):,}")
seen = STEPS * batch_size * block_size
print(f"Will read    : {seen:,} tokens = {seen/len(data):.1f} passes "
      f"over the book")
print(f"Running on   : {device}")

if START_FROM is not None:
    model.load_state_dict(torch.load(START_FROM, map_location=device))
    print(f"Starting from: {START_FROM}")

# The whole book moves onto the GPU once, and so does the mask.
data = data.to(device)
answer_mask = answer_mask.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("\nTraining on the question-and-answer data...")
running = 0.0
t_start = time.perf_counter()
for step in range(1, STEPS + 1):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    graded = torch.stack([answer_mask[i + 1:i + block_size + 1] for i in ix]).float()

    logits = model(x)
    # One mark per token, as before - but now only the answer tokens count.
    every_token = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1),
                                  reduction="none").view(batch_size, block_size)
    loss = (every_token * graded).sum() / graded.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running += loss.item()
    if step % 500 == 0:
        done = time.perf_counter() - t_start
        left = done / step * (STEPS - step)
        print(f"  step {step:5d} | average answer loss {running/500:.3f} "
              f"| {left/60:4.1f} min left")
        running = 0.0
    if step % SAVE_EVERY == 0:
        torch.save(model.state_dict(), SAVE_AS)


# --- Save it under a NEW name, so last session's model survives ---
torch.save(model.state_dict(), SAVE_AS)
print(f"Saved {SAVE_AS}")

# --- Ask it something straight away ---
prompt = "<|user|>\nWhat is photosynthesis?\n<|assistant|>\n"
ids = encode(prompt)
out = model.generate(torch.tensor([ids], device=device), max_new_tokens=120)[0].tolist()
answer = decode(out[len(ids):])
for tag in ("<|endoftext|>", "<|user|>", "<|assistant|>"):   # stop at the first signpost
    answer = answer.split(tag)[0]

print("\n" + "=" * 55)
print("YOUR MODEL, ANSWERING A QUESTION")
print("=" * 55)
print(answer.strip())

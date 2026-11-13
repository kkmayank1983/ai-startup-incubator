"""Session 4 test: did the answers get better?

    python test_session4.py                          -> compares qa_model_v6.pth with qa_model_v7.pth
    python test_session4.py a.pth b.pth c.pth        -> compares any models you name

For every model it prints
  1. the ANSWER LOSS on 300 fixed windows of the book (same windows for
     every model, so the numbers are directly comparable; lower is better)
  2. its greedy answer to ten questions, so you can read the difference.
"""
import os
import sys
import torch
import torch.nn.functional as F

from attention_engine import model, device, block_size
from data_pipeline import data, vocab_size, stoi, encode, decode

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

checkpoints = sys.argv[1:] or ["qa_model_v6.pth", "qa_model_v7.pth"]

QUESTIONS = [
    "Why do satellites stay up instead of crashing?",
    "Why does a straw look bent in a glass of water?",
    "Why does iron go rusty?",
    "What is photosynthesis?",
    "Is air a mixture?",
    "why is the sky blue",
    "what happens when you mix an acid and a base",
    "How does a magnet work?",
    "whats the difference between heat and temperature",
    "What is DNA?",
]

ASSISTANT, END = stoi["<|assistant|>"], stoi["<|endoftext|>"]
STOP = {END, stoi["<|user|>"], ASSISTANT}

# The same mask train_qa.py builds: True on every token of a tutor reply.
started = torch.cumsum((data == ASSISTANT).int(), 0)
ended = torch.cumsum((data == END).int(), 0)
ended = torch.cat([torch.zeros(1, dtype=ended.dtype), ended[:-1]])
answer_mask = (started > ended) & (data != ASSISTANT)

# 300 fixed windows. Seeded, so every model is marked on the same text.
g = torch.Generator().manual_seed(2024)
WINDOWS = torch.randint(0, len(data) - block_size - 1, (300,), generator=g)


def find(path):
    """Look next to this file first, then in the project root."""
    for folder in (HERE, ROOT):
        full = os.path.join(folder, path)
        if os.path.exists(full):
            return full
    sys.exit(f"Cannot find {path} in {HERE} or {ROOT}")


@torch.no_grad()
def answer_loss():
    total, count = 0.0, 0
    for start in range(0, len(WINDOWS), 20):
        ix = WINDOWS[start:start + 20]
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
        w = torch.stack([answer_mask[i + 1:i + block_size + 1] for i in ix]).float().to(device)
        l = F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1), reduction="none").view_as(w)
        total += (l * w).sum().item()
        count += w.sum().item()
    return total / count


@torch.no_grad()
def greedy(question, max_new=120):
    ids = encode(f"<|user|>\n{question}\n<|assistant|>\n")
    idx = torch.tensor([ids], device=device)
    for _ in range(max_new):
        nxt = model(idx[:, -block_size:])[:, -1, :].argmax(-1, keepdim=True)
        idx = torch.cat((idx, nxt), 1)
        if nxt.item() in STOP:
            break
    text = decode(idx[0, len(ids):].tolist())
    for tag in ("<|endoftext|>", "<|user|>", "<|assistant|>"):
        text = text.split(tag)[0]
    return text.strip()


if __name__ == "__main__":
    results = {}
    answers = {}
    for ck in checkpoints:
        model.load_state_dict(torch.load(find(ck), map_location=device))
        model.eval()
        results[ck] = answer_loss()
        answers[ck] = [greedy(q) for q in QUESTIONS]
        print(f"{ck}: answer loss {results[ck]:.3f}", flush=True)

    print("\n" + "=" * 70)
    print("ANSWER LOSS on the same 300 windows (lower is better)")
    print("=" * 70)
    for ck, v in results.items():
        print(f"  {ck:28s} {v:.3f}")

    print("\n" + "=" * 70)
    print("THE SAME TEN QUESTIONS, GREEDY ANSWERS")
    print("=" * 70)
    for n, q in enumerate(QUESTIONS):
        print(f"\nQ: {q}")
        for ck in checkpoints:
            print(f"  [{ck}] {answers[ck][n]}")

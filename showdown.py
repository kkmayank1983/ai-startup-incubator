import torch

from data_pipeline import encode, decode

# engine_v1.py is the copy you made BEFORE editing -- one head, no LayerNorm.
from engine_v1 import model as model_v1
from attention_engine import model as model_v2

model_v1.load_state_dict(torch.load("qa_model.pth"))
model_v2.load_state_dict(torch.load("qa_model_v2.pth"))
model_v1.eval()
model_v2.eval()

QUESTIONS = [
    "Why do kitchen fridge doors snap shut tightly on their own when you get them close?",
    "Why do garbage recycling centers use giant magnets above their conveyor belts?",
    "Does friction depend on surface area?",
]


def ask(model, question, seed):
    torch.manual_seed(seed)          # same seed for both, so it is a fair race
    prompt = f"<|user|> {question} <|assistant|>"
    ids = encode(prompt)
    out = model.generate(torch.tensor([ids]), max_new_tokens=45)[0].tolist()
    answer = decode(out[len(ids):])
    for tag in ("<|endoftext|>", "<|user|>", "<|assistant|>"):
        answer = answer.split(tag)[0]
    return answer.strip()


p1 = sum(p.numel() for p in model_v1.parameters())
p2 = sum(p.numel() for p in model_v2.parameters())

print("=" * 60)
print("SHOWDOWN: Session 6 engine  vs  Session 7 engine")
print("=" * 60)
print(f"v1  1 head,  no LayerNorm : {p1:,} parameters")
print(f"v2  4 heads, LayerNorm    : {p2:,} parameters")
print(f"difference                : {p2 - p1:,} more  ({(p2-p1)/p1*100:.1f}%)")

for n, q in enumerate(QUESTIONS):
    print("\n" + "-" * 60)
    print(f"Q: {q}")
    print("-" * 60)
    print(f"v1: {ask(model_v1, q, 100 + n)}")
    print()
    print(f"v2: {ask(model_v2, q, 100 + n)}")
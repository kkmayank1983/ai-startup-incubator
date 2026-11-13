import sys
import torch
import torch.nn.functional as F

# No training this time. The model is loaded from the file you saved.
from attention_engine import model, device, block_size
from data_pipeline import encode, decode, itos, stoi

#     python talk_to_your_engine.py                     -> today's model, qa_model_v7.pth
#     python talk_to_your_engine.py qa_model_v6.pth     -> last session's model
checkpoint = sys.argv[1] if len(sys.argv) > 1 else "qa_model_v7.pth"

# How the next piece gets picked. 0 = always take the favourite (greedy).
# That is the fairest way to compare two models: ask the same question
# twice and you get the same answer. Try 0.5 for a bit of variety.
TEMPERATURE = 0.7

model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

STOP = {stoi["<|endoftext|>"], stoi["<|user|>"], stoi["<|assistant|>"]}


@torch.no_grad()
def answer(question, max_new_tokens=120):
    # Laid out exactly like every block in input.txt.
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    ids = encode(prompt)
    idx = torch.tensor([ids], device=device)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -block_size:])[:, -1, :]
        if TEMPERATURE <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            nxt = torch.multinomial(F.softmax(logits / TEMPERATURE, dim=-1), num_samples=1)
        idx = torch.cat((idx, nxt), dim=1)
        if nxt.item() in STOP:          # the model said "I'm done"
            break
    text = decode(idx[0, len(ids):].tolist())
    for tag in ("<|endoftext|>", "<|user|>", "<|assistant|>"):
        text = text.split(tag)[0]
    return text.strip()


print("\n" + "=" * 55)
print(f"ASK YOUR AI A SCIENCE QUESTION   [{checkpoint}]")
print("=" * 55)
print("Type a question. Type  quit  to stop.\n")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break

    # Show exactly what it reads. Every character makes it through now.
    pieces = [itos[i] for i in encode(question)]
    print("   (it read: " + "|".join(pieces) + ")")

    print("\nAI: " + answer(question) + "\n")

import torch

from data_pipeline import encode, itos
from attention_engine import model, n_head

# Load the model you just trained.
model.load_state_dict(torch.load("qa_model_v2.pth"))
model.eval()

SENTENCE = "The magnet pulls the iron because it is magnetic"

ids = encode(SENTENCE)
words = [itos[i] for i in ids]

# Running the model fills in .last_weights inside every head.
with torch.no_grad():
    model(torch.tensor([ids]))

print("Sentence:", " ".join(words))
print()
print("Each column is one head. For every word, we show the word it")
print("paid the MOST attention to.")
print()

header = f"{'word':<12}" + "".join(f"head {h+1:<8}" for h in range(n_head))
print(header)
print("-" * len(header))

for pos, word in enumerate(words):
    row = f"{word:<12}"
    for h in range(n_head):
        wei = model.sa_head.heads[h].last_weights[0]   # (T, T)
        best = torch.argmax(wei[pos]).item()           # who did it look at?
        row += f"{words[best]:<13}"
    print(row)

print()
print("If every column were identical, the extra heads would be pointless.")
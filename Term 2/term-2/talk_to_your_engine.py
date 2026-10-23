import torch

# No training this time. The model is loaded from the file you saved.
from attention_engine import model
from data_pipeline import encode, decode, itos

model.load_state_dict(torch.load("qa_model_v3.pth"))
model.eval()

print("\n" + "=" * 55)
print("ASK YOUR AI A PHYSICS QUESTION")
print("=" * 55)
print("Type a question. Type  quit  to stop.\n")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break

    # Show exactly what it reads. Every character makes it through now.
    pieces = [itos[i] for i in encode(question)]
    print("   (it read: " + "|".join(pieces) + ")")

    # Laid out exactly like every block in input.txt.
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    ids = encode(prompt)

    out = model.generate(torch.tensor([ids]), max_new_tokens=120)[0].tolist()

    # Keep only the new text, and stop at the first signpost.
    answer = decode(out[len(ids):])
    for tag in ("<|endoftext|>", "<|user|>", "<|assistant|>"):
        answer = answer.split(tag)[0]

    print("\nAI: " + answer.strip() + "\n")

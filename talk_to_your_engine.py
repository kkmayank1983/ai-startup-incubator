import torch
import sys
import torch

# No training this time. The model is loaded from the file you saved.
#from attention_engine import model
from attention_engine import model, device
from data_pipeline import encode, decode, itos


#model.load_state_dict(torch.load("qa_model_v4.pth"))
#model.load_state_dict(torch.load("qa_model_v5.pth", map_location=device))


# Until now this was typed into the file, so comparing two models meant
# editing code. From Session 4 you will have five of them.
#     python talk_to_your_engine.py qa_model_v4.pth
checkpoint = sys.argv[1] if len(sys.argv) > 1 else "qa_model_v5.pth"

model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()
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

    # Laid out exactly like every block in input.txt.
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    ids = encode(prompt)

    #out = model.generate(torch.tensor([ids]), max_new_tokens=120)[0].tolist()
    out = model.generate(torch.tensor([ids], device=device), max_new_tokens=120)[0].tolist()
    # Keep only the new text, and stop at the first signpost.
    answer = decode(out[len(ids):])
    for tag in ("<|endoftext|>", "<|user|>", "<|assistant|>"):
        answer = answer.split(tag)[0]

    print("\nAI: " + answer.strip() + "\n")

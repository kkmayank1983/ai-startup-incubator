import torch

# No training this time. The model is loaded from the file you saved.
from attention_engine import model, block_size
from data_pipeline import stoi, encode, decode

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

    missing = [w for w in question.split() if w not in stoi]
    if missing:
        print(f"   (these words are not in the vocabulary, so they were dropped: {missing})")

    # Wrap the question the same way every block in input.txt is wrapped.
    prompt = f"<|user|> {question} <|assistant|>"
    ids = encode(prompt)
    if not ids:
        print("   (nothing left for the AI to read)\n")
        continue

    context = torch.tensor([ids[-block_size:]], dtype=torch.long)
    result = model.generate(context, max_new_tokens=60)[0].tolist()

    # Step 1: drop the prompt. generate() returns your question AND the new
    # text, so we keep only the part after what we fed in.
    answer = decode(result[len(ids):])

    # Step 2: stop at the first signpost. The model keeps writing after it
    # finishes an answer -- it starts a whole new question. Cut it off there.
    for tag in ("<|endoftext|>", "<|user|>", "<|assistant|>"):
        answer = answer.split(tag)[0]

    print("\n AI: " + answer.strip() + "\n")
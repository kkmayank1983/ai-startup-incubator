import os
import torch
from bpe_tokenizer import BPETokenizer

current_dir = os.path.dirname(__file__)
input_file = os.path.join(current_dir, "input.txt")
tokenizer_file = os.path.join(current_dir, "tokenizer.json")
tokens_file = os.path.join(current_dir, "tokens.pt")

# 4,096 was enough for four physics topics. Twenty subjects bring
# chlorophyll, voltage and mitochondria, and at 4,096 every one of
# those gets shattered into scraps.
VOCAB_SIZE = 6144


def read_book():
    """Only opened when something actually needs it. The book is 190 MB,
    and once the two saved files exist, nothing does."""
    with open(input_file, "r", encoding="utf-8") as f:
        return f.read()


# Learning the glue rules is slow and the answer never changes, so it
# is done once and written to tokenizer.json. You were given that file.
tokenizer = BPETokenizer()
if os.path.exists(tokenizer_file):
    tokenizer.load(tokenizer_file)
    print(f"Loaded the glue rules from {os.path.basename(tokenizer_file)}")
else:
    print("No saved tokenizer. Learning the rules -- go and have lunch.")
    tokenizer.train(read_book(), VOCAB_SIZE)
    tokenizer.save(tokenizer_file)
    print(f"Saved {os.path.basename(tokenizer_file)}")

stoi = tokenizer.stoi            # piece  -> number
itos = tokenizer.itos            # number -> piece
vocab_size = len(stoi)           # how many different PIECES exist
encode = tokenizer.encode
decode = tokenizer.decode

# Same reasoning. Spelling out 41 million chunks takes a while and the
# answer never changes, so it lives in tokens.pt. Saved as int16 because
# 6,144 pieces fit in one, which halves the file you have to upload.
if os.path.exists(tokens_file):
    data = torch.load(tokens_file).long()
    print(f"Loaded the numbered book from {os.path.basename(tokens_file)}")
else:
    print("Numbering the book -- this one is slow too.")
    data = torch.tensor(encode(read_book()), dtype=torch.long)
    torch.save(data.to(torch.int16), tokens_file)
    print(f"Saved {os.path.basename(tokens_file)}")

print(f"Dataset Size: {len(data)} tokens.")
print(f"Different pieces: {vocab_size}")
print(f"First 10 tokens: {data[:10]}")
print("Decoded:", repr(decode(data[:10].tolist())))

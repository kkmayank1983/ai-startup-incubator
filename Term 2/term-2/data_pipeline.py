import os
import torch
from bpe_tokenizer import BPETokenizer

# --- Open the book ---
current_dir = os.path.dirname(__file__)
input_file = os.path.join(current_dir, "input.txt")

with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

# --- The tokenizer, now BPE: words broken into PIECES ---
# Session 2 used single characters. Session 6 used whole words.
# Today we start from the characters again, then glue the most
# common neighbours together until we have VOCAB_SIZE pieces.
# Any word can be spelled from pieces, so nothing is ever unknown.
VOCAB_SIZE = 1024

tokenizer = BPETokenizer()
tokenizer.train(text, VOCAB_SIZE)

stoi = tokenizer.stoi            # piece  -> number
itos = tokenizer.itos            # number -> piece
vocab_size = len(stoi)           # how many different PIECES exist
encode = tokenizer.encode
decode = tokenizer.decode

# --- Turn the WHOLE book into numbers ---
data = torch.tensor(encode(text), dtype=torch.long)

print(f"Dataset Size: {len(data)} tokens.")
print(f"Different pieces: {vocab_size}")
print(f"First 10 tokens: {data[:10]}")
print("Decoded:", repr(decode(data[:10].tolist())))

import torch
text = "The quick brown fox jumps over the lazy dog."

# Find all unique characters and sort them
chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Vocabulary: {chars}")
print(f"Vocab Size: {vocab_size}")
    
# stoi: string-to-integer (The Encoder)
stoi = { ch:i for i,ch in enumerate(chars) }

# itos: integer-to-string (The Decoder)
itos = { i:ch for i,ch in enumerate(chars) }

print("Character → Token ID Mapping:", stoi)

print(f"Mapping for 's': {stoi['s']}")

sample_id = stoi['s']
print(f"ID {sample_id} belongs to: {itos[sample_id]}")

# Text -> list of IDs
def encode(s):
    return [stoi[c] for c in s]
# list of IDs -> Text
def decode(l):
    return ''.join([itos[i] for i in l])

# Test the engineering
test_msg = "football is fun"
encoded_seq = encode(test_msg)

print(f"Raw Input: {test_msg}")
print(f"Numerical Sequence: {encoded_seq}")
print(f"Decoded Output: {decode(encoded_seq)}")

# Convert our encoded list into a PyTorch Tensor
data_tensor = torch.tensor(encode(text), dtype=torch.long)

# Inspect the architecture
print(data_tensor)
print(f"Tensor Shape: {data_tensor.shape}")
print(f"Tensor Data Type: {data_tensor.dtype}")
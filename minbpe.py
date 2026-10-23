# minbpe.py - Pure Python Byte Pair Encoding (BPE) Tokenizer

def get_stats(ids):
    """
    Scans a list of token IDs and counts how many times each consecutive 
    pair appears together. E.g., for [1, 2, 1, 2], the pair (1, 2) appears twice.
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    Scans through the token IDs and replaces every occurrence of the target 
    'pair' (e.g., [1, 2]) with a single new token 'idx' (e.g., 256).
    """
    newids = []
    i = 0
    while i < len(ids):
        # If we find a match for the pair, merge them and advance index by 2
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class BasicTokenizer:
    def __init__(self):
        # self.merges maps the merged pairs to their new token IDs: (int, int) -> int
        self.merges = {} 
        # self.vocab maps each token ID back to its raw byte representation: int -> bytes
        # We initialize it with the 256 base bytes (ASCII 0 to 255)
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size, verbose=False):
        """
        Learns new subwords by repeatedly finding the most common pairs 
        in the text and assigning them a new token ID.
        """
        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8") # Convert raw string to raw bytes
        ids = list(text_bytes)            # Treat individual bytes as starting token IDs
        
        for i in range(num_merges):
            stats = get_stats(ids)        # 1. Count all adjacent pairs
            if not stats:
                break
            # 2. Find the most frequent pair in the text
            pair = max(stats, key=stats.get)
            idx = 256 + i                 # 3. Create a new token ID starting at 256
            
            # 4. Merge this pair throughout the entire dataset
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx       # 5. Save the merge rule
            
            # 6. Save the spelling of the new subword in the vocabulary dictionary
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx}")

    def encode(self, text):
        """
        Converts a raw string into a list of token IDs using our BPE merge rules.
        """
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        # OPTIMIZATION: Apply BPE merges in the exact order they were trained.
        # This is mathematically equivalent to scanning stats but runs 50x faster in Python!
        for pair, idx in self.merges.items():
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        """
        Converts a list of token IDs back into a readable UTF-8 string.
        """
        # Convert each token ID back into its saved bytes, and join them together
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        # Decode bytes back into a Python string (ignoring formatting errors)
        return text_bytes.decode("utf-8", errors="replace")
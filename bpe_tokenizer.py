import json
import re
from collections import Counter

# The three signposts in input.txt. They must stay in one piece,
# so the tokenizer is told about them before it learns anything.
TAGS = ["<|user|>", "<|assistant|>", "<|endoftext|>"]

# How the text gets cut into chunks BEFORE any learning happens:
#   a signpost            <|user|>
#   a word, with the space in front of it kept    " gravity"
#   one punctuation mark, with its space          "?"  or  " ("
#   any other whitespace                           "\n"
# Pieces are only ever glued together INSIDE a chunk, so a piece can
# never spill across two words.
CHUNK = re.compile(
    r"<\|user\|>|<\|assistant\|>|<\|endoftext\|>| ?[A-Za-z0-9]+| ?[^A-Za-z0-9\s<]|\s|."
)


def glue(chunk, pair):
    """Replace every (a, b) that sits side by side in chunk with a+b."""
    out = []
    i = 0
    while i < len(chunk):
        if i < len(chunk) - 1 and (chunk[i], chunk[i + 1]) == pair:
            out.append(pair[0] + pair[1])
            i += 2
        else:
            out.append(chunk[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    """Learns which letters belong together, then reads ANY text."""

    def __init__(self):
        self.merges = []     # the glue rules, in the order they were learned
        self.stoi = {}       # piece  -> number
        self.itos = {}       # number -> piece
        self.cache = {}      # remembers chunks it has already broken up

    def train(self, text, vocab_size):
        # 1. Cut the book into chunks and count how often each one appears.
        counts = Counter(CHUNK.findall(text))

        # 2. Every chunk starts life as a row of single characters.
        #    (A signpost stays as one whole piece.)
        table = {}
        for chunk, n in counts.items():
            table[(chunk,) if chunk in TAGS else tuple(chunk)] = n

        # 3. The starting alphabet: every single character in the book.
        pieces = sorted({p for row in table for p in row})

        # 4. Glue the most common neighbouring pair. Repeat until full.
        while len(pieces) < vocab_size:
            pairs = Counter()
            for row, n in table.items():
                for a, b in zip(row, row[1:]):
                    pairs[(a, b)] += n
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            pieces.append(best[0] + best[1])
            table = {glue(row, best): n for row, n in table.items()}

        self.stoi = {p: i for i, p in enumerate(pieces)}
        self.itos = {i: p for i, p in enumerate(pieces)}

    def save(self, path):
        """Write the learned rules out, so training happens once."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "pieces": [self.itos[i] for i in range(len(self.itos))],
                "merges": [list(pair) for pair in self.merges],
            }, f)

    def load(self, path):
        """Read the rules back. Gives exactly the tokenizer that saved them."""
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        pieces = saved["pieces"]
        self.stoi = {p: i for i, p in enumerate(pieces)}
        self.itos = {i: p for i, p in enumerate(pieces)}
        self.merges = [tuple(pair) for pair in saved["merges"]]
        self.cache = {}
    def break_up(self, chunk):
        """Turn one chunk into pieces using the learned glue rules."""
        if chunk in self.cache:
            return self.cache[chunk]
        if chunk in TAGS:
            row = (chunk,)
        else:
            # A character the book never contained cannot be read. Skip it.
            row = tuple(c for c in chunk if c in self.stoi)
        for pair in self.merges:
            if len(row) < 2:
                break
            row = glue(row, pair)
        self.cache[chunk] = row
        return row

    def encode(self, text):
        ids = []
        for chunk in CHUNK.findall(text):
            ids.extend(self.stoi[p] for p in self.break_up(chunk))
        return ids

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

from collections import Counter
from data_pipeline import text, encode, itos, vocab_size, data

# Rebuild last session's WORD tokenizer, just to compare against it.
old_words = set(text.split())

QUESTIONS = [
    "What is gravity?",
    "What is entropy?",
    "Why is the sky blue?",
    "What causes a rainbow?",
    "How fast does light travel?",
]

print("=" * 60)
print("WHAT THE MODEL ACTUALLY RECEIVES")
print("=" * 60)
for q in QUESTIONS:
    kept = [w for w in q.split() if w in old_words]
    lost = [w for w in q.split() if w not in old_words]
    pieces = [itos[i] for i in encode(q)]
    print(f"\nYou type   : {q}")
    print(f"Last week  : {' '.join(kept) if kept else '(nothing at all)'}"
          + (f"   <-- dropped {lost}" if lost else ""))
    print(f"This week  : {'|'.join(pieces)}")

# How often does each piece get seen during training?
counts = Counter(data.tolist())
rare_old = sum(1 for w, n in Counter(text.split()).items() if n == 1)
rare_new = sum(1 for p, n in counts.items() if n == 1)

print("\n" + "=" * 60)
print("HOW MUCH PRACTICE DOES EACH TOKEN GET?")
print("=" * 60)
print(f"Last week : {len(old_words):>5} words   - {rare_old:>5} of them appear only ONCE in the book")
print(f"This week : {vocab_size:>5} pieces  - {rare_new:>5} of them appear only ONCE in the book")

longest = sorted(itos.values(), key=len, reverse=True)[:10]
print("\nThe longest pieces it decided to learn on its own:")
print("   " + "  ".join(repr(p) for p in longest))

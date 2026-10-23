import torch

# Today's pipeline: BPE pieces.
from data_pipeline import text, encode, decode, vocab_size
from attention_engine import PredictionEngine, model as model_v3

# Rebuild last session's WORD tokenizer, exactly as Session 6 wrote it,
# so we can load last week's model and race the two side by side.
words = sorted(set(text.split()))
w_stoi = {w: i for i, w in enumerate(words)}
w_itos = {i: w for i, w in enumerate(words)}
w_encode = lambda s: [w_stoi[w] for w in s.split() if w in w_stoi]
w_decode = lambda ids: " ".join(w_itos[i] for i in ids)

model_v2 = PredictionEngine(len(words))          # same engine, word vocabulary
model_v2.load_state_dict(torch.load("qa_model_v2.pth"))
model_v3.load_state_dict(torch.load("qa_model_v3.pth"))
model_v2.eval()
model_v3.eval()

QUESTIONS = [
    "What is gravity?",                                     # in the book, exactly
    "What is inertia?",                                     # in the book, exactly
    "How fast does light travel?",                          # 'travel?' was dropped last week
    "Why do heavy and light objects fall at the same rate?",# 'rate?' was dropped last week
    "Explain gravity to me",                                # 'Explain' was dropped last week
    "What is entropy?",                                     # not in the book at all
]


# How much of an answer is copied word-for-word from the book?
# We slide a 5-word window along the answer and check whether that exact
# run of words appears anywhere in input.txt.
book_words = text.split()
book_runs = {tuple(book_words[i:i + 5]) for i in range(len(book_words) - 4)}

def copied(answer):
    w = answer.split()
    if len(w) < 5:
        return 0.0
    runs = [tuple(w[i:i + 5]) for i in range(len(w) - 4)]
    return 100 * sum(r in book_runs for r in runs) / len(runs)


def ask(model, enc, dec, question, seed, n_tokens):
    torch.manual_seed(seed)                      # same seed for both: a fair race
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    ids = enc(prompt)
    out = model.generate(torch.tensor([ids]), max_new_tokens=n_tokens)[0].tolist()
    answer = dec(out[len(ids):])
    for tag in ("<|endoftext|>", "<|user|>", "<|assistant|>"):
        answer = answer.split(tag)[0]
    return " ".join(answer.split()[:45])         # first 45 words, for both


p2 = sum(p.numel() for p in model_v2.parameters())
p3 = sum(p.numel() for p in model_v3.parameters())

print("=" * 60)
print("SHOWDOWN: word tokens (Session 7)  vs  BPE pieces (today)")
print("=" * 60)
print(f"v2  {len(words):>5} words   : {p2:,} parameters")
print(f"v3  {vocab_size:>5} pieces  : {p3:,} parameters   ({(p2-p3)/p2*100:.0f}% smaller)")

lost_total = 0
copy2 = copy3 = 0.0
for n, q in enumerate(QUESTIONS):
    seen = [w for w in q.split() if w in w_stoi]
    lost = [w for w in q.split() if w not in w_stoi]
    lost_total += len(lost)
    print("\n" + "-" * 60)
    print(f"Q: {q}")
    print("-" * 60)
    print(f"v2 sees : {' '.join(seen) if seen else '(nothing)'}"
          + (f"      dropped {lost}" if lost else ""))
    a2 = ask(model_v2, w_encode, w_decode, q, 100 + n, 45)
    a3 = ask(model_v3, encode, decode, q, 100 + n, 160)
    copy2 += copied(a2)
    copy3 += copied(a3)
    print(f"v2 says : {a2}")
    print(f"          ({copied(a2):.0f}% copied word-for-word from the book)")
    print()
    print(f"v3 sees : {q}      (every character)")
    print(f"v3 says : {a3}")
    print(f"          ({copied(a3):.0f}% copied word-for-word from the book)")

total_words = sum(len(q.split()) for q in QUESTIONS)
print("\n" + "=" * 60)
print(f"Words the OLD tokenizer could not read : {lost_total} of {total_words}")
print(f"Words the NEW tokenizer could not read : 0 of {total_words}")
print(f"Copied word-for-word, on average        : v2 {copy2/len(QUESTIONS):.0f}%   v3 {copy3/len(QUESTIONS):.0f}%")

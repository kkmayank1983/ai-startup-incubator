import os
import difflib
import torch
from generation_engine import PredictionEngine
from data_pipeline import vocab_size, encode, decode

# Resolve file paths robustly relative to this script's directory
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "stem_model.pth")
input_path = os.path.join(current_dir, "input.txt")

# Load model weights locally on CPU
model = PredictionEngine(vocab_size)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Load input.txt questions to check for matches
with open(input_path, 'r', encoding='utf-8') as f:
    raw_text = f.read()

# Extract all valid questions we trained on (between <|user|> and <|assistant|>)
trained_questions = {} # Maps lowercase question -> original casing question
parts = raw_text.split("<|user|>")
for part in parts[1:]:
    q_part = part.split("<|assistant|>")[0].strip()
    trained_questions[q_part.lower()] = q_part

def normalize_query(q):
    """Clean and format student input to match dataset style"""
    q = q.strip()
    if not q:
        return ""
    
    # Capitalize first letter
    q = q[0].upper() + q[1:]
    
    # Clean up spacing before question marks (e.g. "gravity ?" -> "gravity?")
    if q.endswith(" ?"):
        q = q[:-2] + "?"
    elif not q.endswith("?") and not q.endswith(".") and not q.endswith("!"):
        # Add question mark if missing
        q = q + "?"
        
    return q

# List of common conversational stopwords to ignore
STOPWORDS = {
    'what', 'is', 'a', 'an', 'the', 'tell', 'me', 'about', 'explain', 'can', 
    'you', 'please', 'to', 'for', 'of', 'in', 'on', 'with', 'and', 'or', 'but',
    'how', 'does', 'why', 'do', 'we', 'are', 'hey', 'define', 'give', 'detail',
    'concept', 'describe', 'definition', 'get', 'getting', 'sticks', 'stick',
    'rubbed', 'rubbing', 'rub', 'doesnt', 'dont', 'didnt', 'cant', 'isnt', 
    'arent', 'wasnt', 'werent', 'do', 'does', 'did', 'make', 'makes', 'made',
    'due', 'between'
}

SYNONYMS = {
    'gravitational': 'gravity',
    'magnetic': 'magnet',
    'magnets': 'magnet',
    'magnetism': 'magnet',
    'frictional': 'friction',
    'frictions': 'friction',
    'inertial': 'inertia',
    'aluminium': 'aluminum',
    'earths': 'earth'
}

# Core topics to prevent cross-topic bleeding
TOPIC_KEYWORDS = {'gravity', 'inertia', 'friction', 'magnet'}

def extract_keywords(sentence):
    clean_s = sentence.lower()
    clean_s = clean_s.replace("doesn't", "doesnt").replace("don't", "dont").replace("didn't", "didnt")
    for char in ["?", ".", "!", ",", "'s", "'", "-"]:
        clean_s = clean_s.replace(char, " ")
    
    words = clean_s.split()
    cleaned_words = []
    for w in words:
        if w in STOPWORDS:
            continue
        w = SYNONYMS.get(w, w)
        cleaned_words.append(w)
    return cleaned_words

print("=" * 45)
print("     AI NEXUS LOCAL STEM ASSISTANT")
print("=" * 45)
print("Type a prompt (e.g. 'What is gravity?', 'Define energy.')")
print("and watch your local model autocomplete the answer!")
print("Type 'exit' to quit.")

previous_keywords = []

while True:
    question = input("\nYou: ")
    if question.lower() == "exit":
        print("Goodbye!")
        break

    # 1. Normalize the question (e.g. "why do we yawn?" -> "Why do we yawn?")
    normalized_q = normalize_query(question)
    normalized_q_lower = normalized_q.lower()
    
    # Extract current keywords
    current_keywords = extract_keywords(normalized_q_lower)
    
    # Context-carry: if query starts with a preposition or has <= 2 keywords
    is_followup = len(current_keywords) <= 2 and (
        normalized_q_lower.startswith("on ") or 
        normalized_q_lower.startswith("in ") or 
        normalized_q_lower.startswith("at ") or
        normalized_q_lower.startswith("for ")
    )
    
    combined_keywords = current_keywords
    if is_followup and previous_keywords:
        combined_keywords = list(set(current_keywords + previous_keywords))
        
    user_topics = {w for w in combined_keywords if w in TOPIC_KEYWORDS}
    
    # 2. Hybrid Matching: Check character similarity first (essential for math/typos)
    matched_q = None
    best_char_ratio = 0.0
    for q_lower, q_original in trained_questions.items():
        if normalized_q_lower == q_lower:
            matched_q = q_original
            best_char_ratio = 1.0
            break
        ratio = difflib.SequenceMatcher(None, normalized_q_lower, q_lower).ratio()
        if ratio > best_char_ratio:
            best_char_ratio = ratio
            matched_q = q_original
            
    is_matched = False
    if best_char_ratio >= 0.90:
        is_matched = True
        previous_keywords = combined_keywords
    else:
        # 3. Fallback to keyword Jaccard matching with anti-bleed filters
        if combined_keywords:
            best_keyword_match = None
            best_score = 0.0
            best_overlap_char_ratio = 0.0
            
            for q_lower, q_original in trained_questions.items():
                q_keywords = extract_keywords(q_lower)
                
                # Anti-bleed: If user mentioned specific topic(s), penalize candidates containing other topics
                q_topics = {w for w in q_keywords if w in TOPIC_KEYWORDS}
                if user_topics and q_topics:
                    conflicting_topics = q_topics - user_topics
                    if conflicting_topics:
                        continue # Ignore this candidate because it contains a different topic keyword
                
                matches = sum(1 for w in combined_keywords if w in q_keywords)
                union_len = len(set(combined_keywords) | set(q_keywords))
                overlap_ratio = (matches / union_len) if union_len > 0 else 0.0
                
                # Score scales overlap ratio by total matches to prevent short-sentence bias
                score = overlap_ratio * matches
                char_ratio = difflib.SequenceMatcher(None, normalized_q_lower, q_lower).ratio()
                
                if score > best_score:
                    best_score = score
                    best_keyword_match = q_original
                    best_overlap_char_ratio = char_ratio
                elif score == best_score:
                    if char_ratio > best_overlap_char_ratio:
                        best_overlap_char_ratio = char_ratio
                        best_keyword_match = q_original
            
            min_threshold = 0.15
            if best_score >= min_threshold:
                matched_q = best_keyword_match
                is_matched = True
                previous_keywords = combined_keywords

    # If the query is not matched by either path, reject it
    if not is_matched or matched_q is None:
        print("\nAI: I haven't learned about that topic yet! Ask me one of my subjects, like:")
        print(" - 'Hey, what is gravity?'")
        print(" - 'What is the Law of Inertia?'")
        print(" - 'What is friction?'")
        print(" - 'What is a non-contact force?'")
        continue

    # Let the user know if we autocorrected a typo or article mismatch
    final_char_ratio = difflib.SequenceMatcher(None, normalized_q_lower, matched_q.lower()).ratio()
    if final_char_ratio < 1.0:
        print(f"\n[Interpreting as: \"{matched_q}\"]")

    # Format and encode prompt using the ChatML template
    prompt = f"<|user|>\n{matched_q}\n<|assistant|>\n"
    context = torch.tensor([encode(prompt)], dtype=torch.long)

    # Dynamically encode the stop sequence so it adapts to BPE vocab merges
    stop_sequence = encode("<|endoftext|>")
    buffer_tokens = []

    print("\nAI: ", end="")
    
    # Generate tokens one-by-one (up to 300 tokens)
    for _ in range(300):
        # Crop context to block size limit (256)
        idx_cond = context[:, -256:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        
        # Greedy search: Always pick the most probable token
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        next_token_id = idx_next[0].item()
        
        # Append to model context
        context = torch.cat((context, idx_next), dim=1)
        
        # Buffering logic to hide "<|endoftext|>" tag from printing
        expected_next_index = len(buffer_tokens)
        if expected_next_index < len(stop_sequence) and next_token_id == stop_sequence[expected_next_index]:
            buffer_tokens.append(next_token_id)
            if len(buffer_tokens) == len(stop_sequence):
                buffer_tokens = [] # Clear the stop sequence so it won't be printed after the loop
                break
        else:
            # We diverged from the stop sequence. Flush the buffer first.
            if buffer_tokens:
                print(decode(buffer_tokens), end="", flush=True)
                buffer_tokens = []
            
            # If this new token starts the stop sequence, buffer it; otherwise print it
            if next_token_id == stop_sequence[0]:
                buffer_tokens.append(next_token_id)
            else:
                print(decode([next_token_id]), end="", flush=True)
                
    # If the loop finished without breaking and we have a partial buffer, print it
    if buffer_tokens:
        print(decode(buffer_tokens), end="", flush=True)
    print()
              
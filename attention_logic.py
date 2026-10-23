import torch

# Q: Current token says "I'm looking for a vowel"
q = torch.tensor([0.1, 1.0]) 

# K: Past tokens say "I contain a vowel" (k1) or "I contain a space" (k2)
k1 = torch.tensor([0.9, 0.1]) 
k2 = torch.tensor([0.1, 0.9]) 

# Calculate Scores (how much they match)
score1 = q @ k1
score2 = q @ k2

print(f"Attention Score -> Clue 1 : {score1:.2f}")
print(f"Attention Score -> Clue 2 : {score2:.2f}")

if score1 > score2:
    print("\nThe AI focuses on Clue 1.")
else:
    print("\nThe AI focuses on Clue 2.")

# CHALLENGE: Change q to [0.1, 1.0]. What happens to the scores?
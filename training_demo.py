import torch
import torch.nn.functional as F
from generation_engine import PredictionEngine
from data_pipeline import data, vocab_size, encode, decode

# Hyperparameters for training
batch_size = 32     # Number of independent text sequences processed in parallel
block_size = 256    # Maximum context length (number of tokens in a sequence)
max_iters = 6000   # Total training steps
learning_rate = 1e-3

# Check for Nvidia GPU availability. GPUs perform parallel matrix math 100x faster than CPUs!
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Set random seed for reproducibility. This forces weight initialization and batch 
# selection to be identical, ensuring every student generates the exact same model weights.
#torch.manual_seed(1337)

# Initialize the model and load it onto the selected hardware (GPU or CPU)
model = PredictionEngine(vocab_size).to(device)
# AdamW is the standard optimizer used to update weights based on calculated gradients
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def get_batch():
    """
    Pulls a random set of training examples.
    x contains input tokens, y contains target tokens (shifted one token to the right).
    """
    # 1. Grab random starting indices in the dataset
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 2. Extract sequences of length block_size
    x = torch.stack([data[i:i+block_size] for i in ix])
    # 3. Target is the exact same text, but shifted by 1 (what the model must predict!)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # 4. Load the batches onto the GPU
    x, y = x.to(device), y.to(device)
    return x, y

print("="*60)
print("GENERATION BEFORE TRAINING (RANDOM GUESSES)")
print("="*60)
context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with token 0 (null)
generated = model.generate(context, max_new_tokens=50)
print(decode(generated[0].tolist()))

print("\nStarting Training...")
model.train() # Set model to training mode (enables dropout)
for step in range(max_iters):
    # 1. Get a batch of inputs and targets
    xb, yb = get_batch()

    # 2. Run the forward pass to get predictions
    logits = model(xb)
    # 3. Calculate cross entropy loss (difference between predictions and correct target tokens)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

    # 4. Standard PyTorch optimization steps
    optimizer.zero_grad() # Clear gradients from previous step
    loss.backward()        # Backpropagation: Calculate gradients for all parameters
    optimizer.step()       # Optimizer: Adjust model weights slightly

    # Print loss status every 500 iterations
    if step % 500 == 0:
        print(f"Step {step:4d} | Loss: {loss.item():.4f}")

# Verify what the model learned by running inference
model.eval() # Set model to evaluation mode (disables dropout)
print("\n" + "="*60)
print("GENERATION AFTER TRAINING")
print("="*60)
prompt_str = "gravity is "
prompt_ids = torch.tensor([encode(prompt_str)], dtype=torch.long, device=device)
# Generate 100 new tokens based on the prompt
generated = model.generate(prompt_ids, max_new_tokens=100)
print(decode(generated[0].tolist()))

# Save model parameters as a weights file
torch.save(model.state_dict(), "stem_model.pth")
print("\nModel saved successfully as stem_model.pth!")
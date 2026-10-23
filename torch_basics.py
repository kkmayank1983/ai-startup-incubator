import torch

# 1. Create a 1D Tensor (A row of 3 lockers)
row = torch.tensor([10, 20, 30])
print("--- 1D TENSOR ---")
print("row =", row)
print("Shape:", row.shape)  # Should output: torch.Size([3])

# 2. Create a 2D Tensor (A wall of lockers: 2 rows, 3 columns)
wall = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
print("\n--- 2D TENSOR ---")
print("wall =", wall)
print("Shape:", wall.shape)  # Should output: torch.Size([2, 3])

# --- TENSOR MATH ---
tensor_a = torch.tensor([1, 2, 3])
tensor_b = torch.tensor([10, 20, 30])

# This works perfectly because both shapes are [3]
print("\nAddition Success:")
print(tensor_a + tensor_b)

# --- THE CRASH TEST ---
# tensor_c has a shape of [2]. It only has two lockers!
tensor_c = torch.tensor([100, 200])

print("\nAttempting Shape Mismatch...")
# Uncomment the line below to break your code!
print(tensor_a + tensor_c) 
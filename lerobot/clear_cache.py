import torch

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("CUDA cache cleared.")
else:
    print("CUDA is not available.")

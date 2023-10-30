import torch
from transformers import pipeline

print("Start loading model")

codellama = pipeline("text-generation",
"codellama/CodeLlama-7b-Python-hf",
torch_dtype=torch.float16,
device="mps")

print("Finish loading model")
print("===")
print("Start inference")

codellama("Write a code snippet for fibonacci series",
max_new_tokens=50)

print("Finish inference")

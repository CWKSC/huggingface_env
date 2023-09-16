import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "Deci/DeciLM-6b-instruct"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

inputs = tokenizer.encode("How do I make french toast? Think through it step by step", return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_p=0.95)
print(tokenizer.decode(outputs[0]))
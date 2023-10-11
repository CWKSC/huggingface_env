import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "C:/Develop/AI/Model/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
gptq_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=gptq_config)




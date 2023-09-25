from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
gptq_config = GPTQConfig(
    bits = 4, 
    dataset = "c4", 
    tokenizer = tokenizer
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = gptq_config,
    device_map='auto'
)
peft_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM, 
    inference_mode = False, 
    r = 8, 
    lora_alpha = 32,
    lora_dropout = 0.1
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

dataset = load_dataset("imdb", split="train")

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 512,
)

trainer.train()
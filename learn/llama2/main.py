from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, LlamaForCausalLM
from auto_gptq import exllama_set_max_input_length

model_id = "TheBloke/Llama-2-7b-Chat-GPTQ"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# gptq_config = GPTQConfig(
#     bits = 8, 
#     dataset = "c4", 
#     tokenizer = tokenizer
# )
model = LlamaForCausalLM.from_pretrained(
    model_id,
    revision = "gptq-4bit-64g-actorder_True",
    # quantization_config = gptq_config,
    device_map='auto'
)
model.enable_input_require_grads()
model = exllama_set_max_input_length(model, 4096)
peft_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM, 
    inference_mode = False, 
    r = 8, 
    lora_alpha = 32,
    lora_dropout = 0.1
)
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()

dataset = load_dataset("imdb", split="train")

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 512,
    peft_config = peft_config
)

trainer.train()
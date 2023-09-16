from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/CodeLlama-7B-GGUF", 
    model_file="codellama-7b.q4_K_M.gguf", 
    model_type="llama", 
    gpu_layers=50
)

print(llm("AI is going to"))

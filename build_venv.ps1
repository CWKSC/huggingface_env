py -3.9 -m venv venv
.\venv\Scripts\activate

python -m pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install huggingface_hub
pip install transformers
pip install datasets
pip install accelerate soundfile librosa evaluate jiwer tensorboard gradio

# Fine tune
pip install peft

# Quantization
# https://huggingface.co/docs/transformers/main_classes/quantization
pip install auto-gptq
pip install optimum
# For GGUF
pip install ctransformers[cuda]


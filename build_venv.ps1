py -3.9 -m venv venv
.\venv\Scripts\activate

python -m pip install --upgrade pip

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

pip install huggingface_hub
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/datasets.git
pip install evaluate jiwer

# Fine tune
pip install git+https://github.com/huggingface/peft

# Quantization
# https://huggingface.co/docs/transformers/main_classes/quantization
pip install auto-gptq
pip install git+https://github.com/huggingface/optimum.git
pip install --upgrade accelerate
# For GGUF
pip install ctransformers[cuda]


from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2")

print(pipe("learn\\audio-to-text\\data\\sample1.flac"))

from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-large-v2", 
    chunk_length_s = 30
)

print(pipe("learn\\audio-to-text\\data\\min5.mp3", return_timestamps = True))

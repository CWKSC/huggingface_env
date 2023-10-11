from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-tiny", 
    chunk_length_s = 30
)

print(pipe("learn\\automatic-speech-recognition\\data\\min5.mp3", return_timestamps = True))

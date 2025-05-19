import whisper
import imageio_ffmpeg
from pydub import AudioSegment
import re
import os


ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
os.environ["FFMPEG_BINARY"] = ffmpeg_path

model = whisper.load_model("medium")  # "large"


def patched_load_audio(filename, sr=16000):
    import subprocess
    import numpy as np

    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-threads", "0",
        "-i", filename,
        "-f", "f32le",
        "-ac", "1",
        "-acodec", "pcm_f32le",
        "-ar", str(sr),
        "-"
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(out, np.float32).flatten()
    return audio


whisper.audio.load_audio = patched_load_audio


def transcribe_with_words(audio_path):
    return model.transcribe(audio_path, word_timestamps=True)


def group_words_by_punctuation(words, punctuations={'.', ',', '?', '!', ';', ':'}):
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if any(p in word['word'] for p in punctuations):
            chunks.append(current_chunk)
            current_chunk = []
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def extract_audio_chunks(audio_path, word_chunks):
    audio = AudioSegment.from_file(audio_path)
    audio_chunks = []
    text_chunks = []

    for chunk in word_chunks:
        if not chunk:
            continue
        start_time = chunk[0]['start'] * 1000  # ms
        end_time = chunk[-1]['end'] * 1000
        chunk_audio = audio[start_time:end_time]
        chunk_text = ''.join([w['word'] for w in chunk]).strip()
        audio_chunks.append(chunk_audio)
        text_chunks.append(chunk_text)
    return audio_chunks, text_chunks


def save_chunks(audio_chunks, text_chunks, folder="chunks"):
    os.makedirs(folder, exist_ok=True)
    for i, (a, t) in enumerate(zip(audio_chunks, text_chunks)):
        a.export(f"{folder}/chunk_{i}.wav", format="wav")
        with open(f"{folder}/chunk_{i}.txt", "w", encoding="utf-8") as f:
            f.write(t)


def process_audio_file(audio_path):
    result = transcribe_with_words(audio_path)
    all_words = []
    for segment in result['segments']:
        all_words.extend(segment['words'])

    word_chunks = group_words_by_punctuation(all_words)
    audio_chunks, text_chunks = extract_audio_chunks(audio_path, word_chunks)
    print(text_chunks)
    # save_chunks(audio_chunks, text_chunks)
    print(f"Saved {len(audio_chunks)} aligned audio-text chunks.")
    return audio_chunks, text_chunks


def process_and_classify(audio_path):
    result = transcribe_with_words(audio_path)
    all_words = [w for segment in result['segments'] for w in segment['words']]
    word_chunks = group_words_by_punctuation(all_words)
    audio_chunks, text_chunks = extract_audio_chunks(audio_path, word_chunks)

    from audio_classifier import classify_audio_chunks
    from text_classifier import classify_text_chunks

    audio_preds = classify_audio_chunks(audio_chunks)
    print("Audio predictions:", audio_preds)
    text_preds = classify_text_chunks(text_chunks)
    print("Text predictions:", text_preds)

    return text_chunks, audio_preds, text_preds

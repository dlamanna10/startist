from fastapi import FastAPI, Query
import yt_dlp
import librosa
import uvicorn
import ffmpeg
import os

startist = FastAPI()

@startist.get("/convert")
def to_mp3(url: str):
    output_file = "output.mp3"
    ydl_opts = {
        'format' : 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_file,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    bpm, key = analyze_audio(output_file)
    return {'mp3_url': output_file, 'bpm': bpm, 'key': key}

def analyze_audio(mp3_path):
    y, sr = librosa.load(mp3_path, sr = 32000, duration = 30)
    tempo, _ = librosa.beat.beat_track(y = y, sr = sr)
    key = librosa.feature.chroma_cqt(y = y, sr = sr).mean(axis = 1).argmax()
    return round(tempo), f'Key Index {key}'

if __name__ == '__main__':
    uvicorn.run(startist, host='0.0.0.0', port = 8000)
    
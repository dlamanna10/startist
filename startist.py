from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import yt_dlp
import librosa
import numpy as np
import scipy.signal
import os
import re
import time
import uvicorn

# Initialize FastAPI app
startist = FastAPI()

# Define musical key mapping
KEY_MAPPING = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# FFMPEG Path for Windows
FFMPEG_PATH = r"C:\ffmpeg\bin"

# ✅ Ignore favicon.ico requests
@startist.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})

# ✅ Home route
@startist.get("/")
def read_root():
    return {"message": "Welcome to the YouTube to MP3 Converter!"}

# ✅ Validate YouTube URL before downloading
def is_valid_youtube_url(url: str) -> bool:
    youtube_regex = re.compile(
        r"(https?://)?(www\.)?"
        r"(youtube\.com|youtu\.be)/"
        r"(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})"
    )
    return bool(youtube_regex.match(url))

# ✅ Convert YouTube video to MP3
@startist.get("/convert")
def to_mp3(url: str, filename: str = "output"):
    if not is_valid_youtube_url(url):
        return JSONResponse(status_code=400, content={"error": "Invalid YouTube URL format."})

    try:
        print(f"Downloading from URL: {url}")  # Debugging info
        output_file = filename  # ✅ No .mp3 extension in filename

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'ffmpeg_location': FFMPEG_PATH,
            'outtmpl': output_file,  
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # 🔥 Allow time for yt-dlp to fully save the file
        time.sleep(2)

        # ✅ Ensure MP3 file exists before analysis
        mp3_file = f"{output_file}.mp3"
        if not os.path.exists(mp3_file):
            raise HTTPException(status_code=500, detail="MP3 file was not created. Check if the YouTube link is valid.")

        bpm, key = analyze_audio(mp3_file)

        # ✅ Handle cases where BPM/Key could not be detected
        if bpm is None or key == "Unknown":
            return JSONResponse(status_code=500, content={"error": "BPM or Key could not be detected."})

        return {"mp3_url": mp3_file, "bpm": bpm, "key": key}

    except Exception as e:
        print(f"Error: {str(e)}")  # Logs error to terminal
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ Improved BPM & Key Detection
def analyze_audio(mp3_path):
    try:
        y, sr = librosa.load(mp3_path, sr=32000, duration=30)

        # ✅ Harmonic-Percussive Separation (HPS) to remove percussive noise
        harmonic, percussive = librosa.effects.hpss(y)

        # ✅ Improved BPM Detection using librosa autocorrelation
        onset_env = librosa.onset.onset_strength(y=percussive, sr=sr)
        tempo_candidates = librosa.autocorrelate(onset_env)
        peaks, _ = scipy.signal.find_peaks(tempo_candidates, height=0)
        
        if peaks.any():
            tempo = peaks[0]  # Use the most dominant peak
        else:
            tempo = librosa.beat.tempo(y=y, sr=sr)[0]  # Fallback

        bpm = int(round(tempo))

        # ✅ Improved Key Detection using Harmonic Component
        chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr).mean(axis=1)
        key_index = int(np.argmax(chroma))  # Get root note index

        # ✅ Compare with Major/Minor Scale Profiles
        major_score = np.corrcoef(chroma, np.roll(KEY_MAPPING, key_index))[0, 1]
        minor_score = np.corrcoef(chroma, np.roll(KEY_MAPPING, key_index + 3))[0, 1]

        scale_type = "Major" if major_score > minor_score else "Minor"
        musical_key = f"{KEY_MAPPING[key_index]} {scale_type}"

        return bpm, musical_key

    except Exception as e:
        print(f"Error in analyze_audio: {str(e)}")
        return None, "Unknown"

# ✅ Allow users to download MP3 file
@startist.get("/download")
def download(filename: str = "output.mp3"):
    file_path = os.path.join(os.getcwd(), filename)  # Ensure full path

    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})

    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)

# ✅ Start FastAPI Server
if __name__ == '__main__':
    uvicorn.run(startist, host='0.0.0.0', port=8000)

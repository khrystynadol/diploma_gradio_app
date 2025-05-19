import os
import imageio_ffmpeg
from yt_dlp import YoutubeDL

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()


def download_audio_from_youtube(youtube_url, output_path="samples/sample_1_full.wav"):
    temp_file = "temp_audio.wav"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ydl_opts = {
        'ffmpeg_location': ffmpeg_path,
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    if os.path.exists(temp_file):
        os.rename(temp_file, output_path)
        print(f"Audio downloaded and saved as {output_path}")
    else:
        print("Download failed: 'temp_audio.wav' not found.")


if __name__ == "__main__":
    # youtube_link_1 = "https://youtu.be/OBG50aoUwlI?si=6T6py3NCk2K_AaOh"
    # download_audio_from_youtube(youtube_link_1, output_path="samples/sample_1_full.wav")
    #
    # youtube_link_2 = "https://www.youtube.com/watch?v=SLEvWp8JS1c"
    # download_audio_from_youtube(youtube_link_2, output_path="samples/sample_2_full.wav")
    #
    # youtube_link_3 = "https://www.youtube.com/watch?v=gkr57P0fwbI"
    # download_audio_from_youtube(youtube_link_3, output_path="samples/sample_3_full.wav")

    youtube_link_4 = "https://youtu.be/F45PvP5g4A8?si=3dt0E4wOM_n0jApS"
    download_audio_from_youtube(youtube_link_4, output_path="samples/sample_4_full.wav")

import os
import subprocess
import shutil

def convert_wavs(audio_dir):
    """
    Convert all WAV files in the specified directory to 16-bit PCM, 22050 Hz, mono.
    Creates a backup of original files in a 'backup' subdirectory.
    
    Args:
        audio_dir (str): Path to the directory containing WAV files.
    """
    # Create a backup directory
    backup_dir = os.path.join(audio_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)

    # Process each WAV file
    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):
            input_path = os.path.join(audio_dir, filename)
            backup_path = os.path.join(backup_dir, filename)
            output_path = os.path.join(audio_dir, "cleaned",filename)

            # Backup original file
            shutil.copy(input_path, backup_path)

            # Convert using ffmpeg
            command = [
                "ffmpeg",
                "-i", input_path,
                "-ac", "1",           # Mono
                "-ar", "22050",       # 22050 Hz sample rate
                "-acodec", "pcm_s16le",  # 16-bit PCM
                "-y",                 # Overwrite output without asking
                output_path
            ]
            try:
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"Converted: {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {filename}: {e.stderr.decode()}")

if __name__ == "__main__":
    # Replace with your audio directory path
    audio_directory = "/home/tsee/Tsee/Personal/mongol-tts/dataset/wavs"  # e.g., "/home/user/dataset/100jil"
    convert_wavs(audio_directory)
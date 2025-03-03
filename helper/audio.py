import os
import speech_recognition as sr
from pydub import AudioSegment
import wave



def prepare_voice_file(path: str) -> str:
    """
    Converts the input audio file to WAV format if necessary and returns the path to the WAV file.
    """
    if os.path.splitext(path)[1] == '.wav':
        return path
    elif os.path.splitext(path)[1] in ('.mp3', '.m4a', '.ogg', '.flac'):
        audio_file = AudioSegment.from_file(
            path, format=os.path.splitext(path)[1][1:])
        wav_file = os.path.splitext(path)[0] + '.wav'
        audio_file.export(wav_file, format='wav')
        return wav_file
    else:
        raise ValueError(
            f'Unsupported audio format: {format(os.path.splitext(path)[1])}')
    
def fetch_audio_file(audio_path):
    # Check if the file exists
    if not os.path.exists(audio_path):
        print(f"Audio file not found at {audio_path}")
        return None
    
    # Open the .wav file
    try:
        with wave.open(audio_path, 'rb') as audio_file:
            # You can read or process the audio file here
            print(f"Successfully fetched audio file: {audio_path}")
            # Example: Fetch audio file details
            params = audio_file.getparams()
            print(f"Audio Parameters: {params}")
            return audio_file
    except Exception as e:
        print(f"Error opening audio file: {str(e)}")
        return None


def transcribe_audio(audio_data, language) -> str:
    """
    Transcribes audio data to text using Google's speech recognition API.
    """
    r = sr.Recognizer()
    text = r.recognize_google(audio_data, language=language)
    return text


def write_transcription_to_file(text, output_file) -> None:
    """
    Writes the transcribed text to the output file.
    """
    with open(output_file, 'w') as f:
        f.write(text)

def normalize_path(path):
    return os.path.normpath(path.replace('\\', '/'))


def speech_to_text(input_path: str, language: str) -> None:
    """
    Transcribes an audio file at the given path to text and writes the transcribed text to the output file.
    """
    wav_file = prepare_voice_file(input_path)
    with sr.AudioFile(wav_file) as source:
        audio_data = sr.Recognizer().record(source)
        text = transcribe_audio(audio_data, language)
        # print(f"####################### Processed audio file: {text}")
        # write_transcription_to_file(text, output_path)
        return text


# if __name__ == '__main__':
#     print('Please enter the path to an audio file (WAV, MP3, M4A, OGG, or FLAC):')
#     input_path = input().strip()
#     if not os.path.isfile(input_path):
#         print('Error: File not found.')
#         exit(1)
#     else:
#         print('Please enter the path to the output file:')
#         output_path = input().strip()
#         print('Please enter the language code (e.g. en-US):')
#         language = input().strip()
#         try:
#             speech_to_text(input_path, output_path, language)
#         except Exception as e:
#             print('Error:', e)
#             exit(1)
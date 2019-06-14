from pydub import AudioSegment
from glob import iglob
import os

DATA_FILES_MP3 = 'audio'
DATA_FILES_WAV = 'audio_wav'
DATA_FILES_MP3_CONVERTED = 'songs_converted'


def convert_mp3_to_wav():
    if not os.path.exists(DATA_FILES_MP3):
        print('The mp3 folder wasnt found')
        return None
    if not os.path.exists(DATA_FILES_WAV):
        os.makedirs(DATA_FILES_WAV)
    index = 0
    for file in iglob(DATA_FILES_MP3 + '/*.mp3'):
        mp3_to_wav = AudioSegment.from_mp3(file)
        mp3_to_wav = mp3_to_wav.set_frame_rate(44000)
        mp3_to_wav.export(DATA_FILES_WAV + '/' +
                          str(index) + '.wav', format='wav')
        index += 1
        print("Processed", file)


def convert_wav_to_mp3():
    if not os.path.exists(DATA_FILES_WAV):
        print('The wav folder wasnt found')
        return None
    if not os.path.exists(DATA_FILES_MP3_CONVERTED):
        os.makedirs(DATA_FILES_MP3_CONVERTED)
    index = 0
    for file in iglob(DATA_FILES_WAV + '/*.wav'):
        wav_to_mp3 = AudioSegment.from_wav(file)
        wav_to_mp3 = wav_to_mp3.set_frame_rate(48000)
        wav_to_mp3.export(DATA_FILES_MP3_CONVERTED + '/' +
                          str(index) + '.mp3', format='mp3')
        index += 1
        print("Processed", file)

convert_mp3_to_wav()
# convert_wav_to_mp3()

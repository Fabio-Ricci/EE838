from pydub import AudioSegment
from glob import iglob
import os

DATA_FILES_MP3 = 'songs'
DATA_FILES_WAV = 'songs_wav'


def convert_mp3_to_wav():
    if not os.path.exists(DATA_FILES_MP3):
        print('The mp3 folder wasnt found')
        return None
    if not os.path.exists(DATA_FILES_WAV):
        os.makedirs(DATA_FILES_WAV)
    index = 0
    for file in iglob(DATA_FILES_MP3 + '/*.mp3'):
        mp3_to_wav = AudioSegment.from_mp3(file)
        mp3_to_wav = mp3_to_wav.set_frame_rate(48000)
        mp3_to_wav.export(DATA_FILES_WAV + '/' +
                          str(index) + '.wav', format='wav')


        index += 1
        print("Processed", file)


convert_mp3_to_wav()

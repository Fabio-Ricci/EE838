from pydub import AudioSegment
from glob import iglob

DATA_FILES_MP3 = 'songs'
DATA_FILES_WAV = 'songs_wav'

def convert_mp3_to_wav():
 index = 0
 for file in iglob('../' + DATA_FILES_MP3 + '/*.mp3'):
  mp3_to_wav = AudioSegment.from_mp3(file)
  mp3_to_wav.export('../' + DATA_FILES_WAV + '/' + 
                       str(index) + '.wav', format='wav')
  index += 1

convert_mp3_to_wav()
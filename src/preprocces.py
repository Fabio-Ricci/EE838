from scipy.fftpack import rfft, irfft
from tensorflow.contrib.framework.python.ops.audio_ops import decode_wav
import math
import tensorflow as tf
import numpy as np
from glob import iglob
from sklearn.preprocessing import MinMaxScaler
import pickle

DATA_FILES_WAV = 'songs_wav'
PREPROCESSED_DATA = 'preprocessed'
SECTION_SIZE = 12348 // 2


def normalize(v):
    scalerX = MinMaxScaler().fit(v[:, np.newaxis])

    audio = scalerX.transform(v[:, np.newaxis]).flatten()
    audios = [audio[i * SECTION_SIZE:(i + 1) * SECTION_SIZE]
                for i in range((len(audio) + SECTION_SIZE - 1) // SECTION_SIZE)]
    return np.array(audios), scalerX


def unnormalize(v, scaler):
    return scaler.inverse_transform(v[:, np.newaxis]).flatten()


def prepare_preprocess_data():
    i = 0
    file_arr = list(iglob(DATA_FILES_WAV + '/*.wav'))
    sess = tf.Session()

    wav_arr_ch1 = []
    wav_arr_ch2 = []

    for f in file_arr:
        print(f)
        song_wav_arr_ch1 = []
        song_wav_arr_ch2 = []

        audio_binary = tf.read_file(f)
        wav_decoder = decode_wav(
            audio_binary, desired_channels=2)
        sample_rate, audio = sess.run(
            [wav_decoder.sample_rate,
             wav_decoder.audio])
        audio = np.array(audio)
        print("audio ready")
        audios = [audio[i * SECTION_SIZE:(i + 1) * SECTION_SIZE]
                for i in range((len(audio) + SECTION_SIZE - 1) // SECTION_SIZE)]
        rfft0 = []
        rfft1 = []
        for a in audios:
            rfft0.extend(rfft(a[:, 0]))
            rfft1.extend(rfft(a[:, 1]))
        rfft0 = np.array(rfft0)
        rfft1 = np.array(rfft1)
        print(rfft0)
        print("rfft done")
        song_wav_arr_ch1, scaler = normalize(rfft0)
        song_wav_arr_ch2, scaler = normalize(rfft1)
        print("scaling done")
        with open(f + '-0.pickle', 'wb') as handle:
            pickle.dump(song_wav_arr_ch1, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f + '-1.pickle', 'wb') as handle:
            pickle.dump(song_wav_arr_ch2, handle, protocol=pickle.HIGHEST_PROTOCOL)

def preprocess_data():
    i = 0
    file_arr = list(iglob(PREPROCESSED_DATA + '/*.pickle'))
    file_arr.sort()

    wav_arr_ch1 = []
    wav_arr_ch2 = []
    left_channel = True
    
    i = 0
    for f in file_arr:
        if i == 50:
            break
        i += 1
        pickleFile = open(f, 'rb')
        data = pickle.load(pickleFile)
        for d in data:
            if len(d) != SECTION_SIZE:
                print("wrong sample size")
                print(len(d))
                continue
            if left_channel:
                left_channel = False
                wav_arr_ch1.append(d)
            else: 
                left_channel = True
                wav_arr_ch2.append(d)
        

    print("Number of returned chuncks", len(wav_arr_ch1), len(wav_arr_ch2))

    if len(wav_arr_ch1) <= 0:
        print('No data')
        print('Quitting')
        exit()

    return wav_arr_ch1, wav_arr_ch2

from scipy.fftpack import rfft, irfft
from tensorflow.contrib.framework.python.ops.audio_ops import decode_wav
import math
import tensorflow as tf
import numpy as np
from glob import iglob
from sklearn.preprocessing import MinMaxScaler

DATA_FILES_WAV = 'songs_wav'
SECTION_SIZE = 12348 // 2


def normalize(v):
    scalerX = MinMaxScaler().fit(v[:, np.newaxis])

    audio = scalerX.transform(v[:, np.newaxis]).flatten()
    audios = [audio[i * SECTION_SIZE:(i + 1) * SECTION_SIZE]
                for i in range((len(audio) + SECTION_SIZE - 1) // SECTION_SIZE)]
    return np.array(audios), scalerX


def unnormalize(v, scaler):
    return scaler.inverse_transform(v[:, np.newaxis]).flatten()


def preprocess_data():
    i = 0
    file_arr = list(iglob(DATA_FILES_WAV + '/*.wav'))
    sess = tf.Session()

    wav_arr_ch1 = []
    wav_arr_ch2 = []

    for f in file_arr:
        print(f)
        song_wav_arr_ch1 = []
        song_wav_arr_ch2 = []

        if i == 50:
            break
        i += 1
        audio_binary = tf.read_file(f)
        wav_decoder = decode_wav(
            audio_binary, desired_channels=2)
        sample_rate, audio = sess.run(
            [wav_decoder.sample_rate,
             wav_decoder.audio])
        audio = np.array(audio)

        rfft0 = rfft(audio[:, 0])
        rfft1 = rfft(audio[:, 1])
        song_wav_arr_ch1, scaler = normalize(rfft0)
        song_wav_arr_ch2, scaler = normalize(rfft1)
        for s1, s2 in zip(song_wav_arr_ch1, song_wav_arr_ch2):
            if len(s1) != SECTION_SIZE:
                print(len(s1))
                print("wrong sample")
                continue
            wav_arr_ch1.append(s1)
            wav_arr_ch2.append(s2)
        print("Returning File: " + f)
        print("sample rate", sample_rate)

    print("Number of returned chuncks", len(wav_arr_ch1))

    if len(wav_arr_ch1) <= 0:
        print('No data')
        print('Quitting')
        exit()

    return wav_arr_ch1, wav_arr_ch2, sample_rate

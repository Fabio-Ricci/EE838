from scipy.fftpack import rfft, irfft
from tensorflow.contrib.framework.python.ops.audio_ops import decode_wav
import math
import tensorflow as tf
import numpy as np
from glob import iglob
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
import random


DATA_FILES_WAV = 'audio_wav'
PREPROCESSED_DATA = 'preprocessed'
SECTION_SIZE = 12348 // 2


def preprocess_data(batch_size):
    i = 0
    file_arr = list(iglob(DATA_FILES_WAV + '/*.wav'))    
    np.random.shuffle(file_arr)
    sess = tf.Session()

    wav_arr_ch1 = []
    wav_arr_ch2 = []

    for f in file_arr:
        if i == batch_size:
            break
        i += 1
        audio_binary = tf.read_file(f)
        wav_decoder = decode_wav(
            audio_binary, desired_channels=2)
        sample_rate, audio = sess.run(
            [wav_decoder.sample_rate,
             wav_decoder.audio])
        audio = np.array(audio)
        audio = audio[:5292000]
        print(len(audio[:, 0]))
        # We want to ensure that every song we look at has the same
        # number of samples!
        section_size = 12348 // 2
        
        a0 = rfft(audio[:, 0], overwrite_x=True))
        a1 = rfft(audio[:, 1], overwrite_x=True)))

        s_a0 = [a0[i * section_size:(i + 1) * section_size] for i in range((len(a0) + section_size - 1) // section_size )] 
        s_a1 = [a0[i * section_size:(i + 1) * section_size] for i in range((len(a0) + section_size - 1) // section_size )] 

        for a in zip(s_a0, s_a1):
            if len(a[0]) != section_size:
                print(len(a[0]))
                print("wrong sample")
                continue
            wav_arr_ch1.append(a[0])
            wav_arr_ch2.append(a[1])
        print("Returning File: " + f)
        print("sample rate", sample_rate)
    print("Number of returned chuncks", len(wav_arr_ch1))

    if len(wav_arr_ch1) <= 0:
        print('No data')
        print('Quitting')
        exit()

    return wav_arr_ch1, wav_arr_ch2, sample_rate


if __name__ == "__main__":
    prepare_preprocess_data()
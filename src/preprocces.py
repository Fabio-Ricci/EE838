from scipy.fftpack import rfft, irfft
from tensorflow.contrib.framework.python.ops.audio_ops import decode_wav
import math
import tensorflow as tf
import numpy as np
from glob import iglob


DATA_FILES_WAV = 'songs_wav'

def preprocess_data(sess):
    print('lalala3')
    i = 0
    file_arr = list(iglob(DATA_FILES_WAV + '/*.wav'))    
    np.random.shuffle(file_arr)

    wav_arr_ch1 = []
    wav_arr_ch2 = []

    for f in file_arr:
        if i == 22:
            break
        i += 1
        audio_binary = tf.read_file(f)
        wav_decoder = decode_wav(
            audio_binary, desired_channels=2)
        sample_rate, audio = sess.run(
            [wav_decoder.sample_rate,
             wav_decoder.audio])
        audio = np.array(audio)
        # We want to ensure that every song we look at has the same
        # number of samples!
        section_size = 12348 // 2
        audios = [audio[i * section_size:(i + 1) * section_size] for i in range((len(audio) + section_size - 1) // section_size )] 
        for a in audios:
            if len(a[:, 0]) != section_size:
                print(len(a[:, 0]))
                print("wrong sample")
                continue
            wav_arr_ch1.append(rfft(a[:, 0]))
            wav_arr_ch2.append(rfft(a[:, 1]))
        print("Returning File: " + f)
        print("sample rate", sample_rate)
    print("Number of returned chuncks", len(wav_arr_ch1))

    if len(wav_arr_ch1) <= 0:
        print('No data')
        print('Quitting')
        exit()

    return wav_arr_ch1, wav_arr_ch2, sample_rate


from scipy.fftpack import rfft, irfft
from tensorflow.contrib.framework.python.ops.audio_ops import decode_wav
import tensorflow as tf
import numpy as np
from glob import iglob


DATA_FILES_WAV = 'songs_wav'

def preprocess_data():
    file_arr = iglob(DATA_FILES_WAV + '/*.wav')
    sess = tf.Session()

    wav_arr_ch1 = []
    wav_arr_ch2 = []

    for f in file_arr:
        audio_binary = tf.read_file(f)
        wav_decoder = decode_wav(
            audio_binary, desired_channels=2)
        sample_rate, audio = sess.run(
            [wav_decoder.sample_rate,
             wav_decoder.audio])
        audio = np.array(audio)
        # We want to ensure that every song we look at has the same
        # number of samples!
        if len(audio[:, 0]) != 5292000:
            continue
        wav_arr_ch1.append(rfft(audio[:, 0]))
        wav_arr_ch2.append(rfft(audio[:, 1]))
        print("Returning File: " + f)

    return wav_arr_ch1, wav_arr_ch2, sample_rate

preprocess_data()
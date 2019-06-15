import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import math
from tensorflow.keras.models import model_from_json
from scipy.fftpack import rfft, irfft
import tensorflow as tf
from glob import iglob
from tensorflow.contrib.framework.python.ops.audio_ops import decode_wav, encode_wav
from tensorflow.contrib import ffmpeg
import numpy as np
from autoencoder import compile_model, load_model
import matplotlib.pyplot as plt
from preprocces import normalize

autoencoder = load_model('models/model-50eps')

file_arr = iglob('test/*.wav')
sess = tf.Session()

section_size = 12348 // 2

file_number = 0
for f in file_arr:
    ch1_song = np.array([]).astype(float)
    ch2_song = np.array([]).astype(float)

    audio_binary = tf.read_file(f)
    wav_decoder = decode_wav(
        audio_binary, desired_channels=2)
    sample_rate, audio = sess.run(
        [wav_decoder.sample_rate,
         wav_decoder.audio])
    audio = np.array(audio)
    audio = np.array(audio)
    audio = audio[:5280000]
    if len(audio[:, 0]) != 5280000:
        continue
    print(len(audio[:, 0]))
    print(audio.shape)

    a0 = rfft(audio[:, 0])
    a1 = rfft(audio[:, 1])

    a0, max1 = normalize(a0)
    a1, max2 = normalize(a1)

    s_a0 = [a0[i * section_size:(i + 1) * section_size] for i in range((len(a0) + section_size - 1) // section_size )] 
    s_a1 = [a0[i * section_size:(i + 1) * section_size] for i in range((len(a0) + section_size - 1) // section_size )] 

    i = 0

    song_wav_arr_ch1 = np.array([])
    song_wav_arr_ch2 = np.array([])

    print("normalized")
    i = 0
    for norm1, norm2 in zip(s_a0, s_a1):
        i += 1
        if len(norm1) != section_size:
            print(len(norm1))
            print("wrong sample")
            continue

        merged = np.hstack((norm1, norm2))
        plt.plot(merged)
        plt.show()
        merged = np.reshape(merged, (1,12348))
        predicted = autoencoder.predict(merged)
        # predicted = merged
        
        splitted = np.hsplit(predicted[0], 2)
        plt.plot(predicted[0])
        plt.show()
        channel1 = splitted[0]
        channel2 = splitted[1]
        print(ch1_song.shape)
        print(ch2_song.shape)
        ch1_song = np.concatenate((ch1_song, channel1))
        ch2_song = np.concatenate((ch2_song, channel2))
    ch1_song = ((ch1_song - 1) * 2) * max1
    ch2_song = ((ch2_song - 1) * 2) * max2
    ch1_song = irfft(ch1_song)
    ch2_song = irfft(ch2_song)
    audio_arr = np.hstack(np.array((ch1_song, ch2_song)).T)
    cols = 2
    rows = math.floor(len(audio_arr)/2)
    audio_arr = audio_arr.reshape(rows, cols)
   


    wav_encoder = ffmpeg.encode_audio(
		audio_arr, file_format='wav', samples_per_second=sample_rate)

    wav_file = sess.run(wav_encoder)
    open('test_reconstructed/' + str(file_number) + ".wav", 'wb').write(wav_file)
    file_number += 1

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
from preprocces import normalize, unnormalize

autoencoder = load_model('models/model-100eps')

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
    # We want to ensure that every song we look at has the same
    # number of samples!
    audios = [audio[i * section_size:(i + 1) * section_size]
                                          for i in range((len(audio) + section_size - 1) // section_size)]
    i = 0
    for a in audios:
        # if i == 500:
        #     break
        i += 1
        if len(a[:, 0]) != section_size:
            print(len(a[:, 0]))
            print("wrong sample")
            continue
        rfft0, scaler0 = normalize(rfft(a[:, 0]))
        rfft1, scaler1 = normalize(rfft(a[:, 1]))
        
        merged = np.hstack((rfft0, rfft1))
        merged = np.reshape(merged, (1,12348))


        predicted = autoencoder.predict(merged)
        # predicted = merged
        splitted = np.hsplit(predicted[0], 2)
        channel1 = irfft(unnormalize(splitted[0], scaler0))
        channel2 = irfft(unnormalize(splitted[1], scaler1))
        print(ch1_song.shape)
        print(ch2_song.shape)
        ch1_song = np.concatenate((ch1_song, channel1))
        ch2_song = np.concatenate((ch2_song, channel2))

    audio_arr = np.hstack(np.array((ch1_song, ch2_song)).T)
    cols = 2
    rows = math.floor(len(audio_arr)/2)
    audio_arr = audio_arr.reshape(rows, cols)
   


    wav_encoder = ffmpeg.encode_audio(
		audio_arr, file_format='wav', samples_per_second=sample_rate)

    wav_file = sess.run(wav_encoder)
    open('test_reconstructed/' + str(file_number) + ".wav", 'wb').write(wav_file)
    file_number += 1

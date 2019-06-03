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

def compile_model(model):
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model

def load_model(name):
    # load json and create model
    json_file = open(name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name + ".h5")
    loaded_model = compile_model(loaded_model)
    return loaded_model


autoencoder = load_model('models/model-v1-1150eps')

file_arr = iglob('test/*.wav')
sess = tf.Session()

section_size = 12348 // 2


for f in file_arr:
    ch1_song = np.array([]).reshape((1,0)).astype(float)
    ch2_song = np.array([]).reshape((1,0)).astype(float)

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
        # if i == 20:
        #     break
        i += 1
        if len(a[:, 0]) != section_size:
            print(len(a[:, 0]))
            print("wrong sample")
            continue
        merged = np.hstack((rfft(a[:, 0]), rfft(a[:, 1])))
        merged = np.reshape(merged, (1,12348))


        predicted = autoencoder.predict(merged)
        splitted = np.hsplit(predicted[0], 2)
        channel1 = splitted[0]
        channel2 = splitted[1]
        print(ch1_song.shape)
        print(ch2_song.shape)
        ch1_song = np.concatenate((ch1_song, channel1.reshape((1, section_size))), axis=1)
        ch2_song = np.concatenate((ch2_song, channel2.reshape((1, section_size))), axis=1)
    
    audio_arr_ch1 = irfft(np.hstack(np.hstack(ch1_song)))
    audio_arr_ch2 = irfft(np.hstack(np.hstack(ch2_song)))

    audio_arr = np.hstack(np.array((audio_arr_ch1, audio_arr_ch2)).T)
    cols = 2
    rows = math.floor(len(audio_arr)/2)
    audio_arr = audio_arr.reshape(rows, cols)
    print(audio_arr)


    wav_encoder = ffmpeg.encode_audio(
		audio_arr, file_format='wav', samples_per_second=sample_rate)

    wav_file = sess.run(wav_encoder)
    open('test_reconstructed/out.wav', 'wb').write(wav_file)

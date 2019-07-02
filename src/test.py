import os
import sys
import time
import math
from glob import iglob

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.contrib import ffmpeg
from tensorflow.contrib.framework.python.ops.audio_ops import decode_wav, encode_wav

from scipy.fftpack import rfft, irfft

import numpy as np
import matplotlib.pyplot as plt

from autoencoder import compile_model, load_model, parse_args, Params, default_args_dict
from preprocces import normalize, segment, add_overlap

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'




args_dict = parse_args()
if args_dict == None:
    # define the default parameters to be used
    args_dict = default_args_dict()

params = Params(args_dict)
params.initial_epoch = 1650
print("[warning] overwriting: initial_epoch=1650")




full_path = params.load_model_path_start
full_path += params.load_model_path_version
full_path += f"model-{params.initial_epoch}eps"
autoencoder = load_model(full_path, lr=params.learning_rate)

file_arr = iglob(params.wav_test_data_path_start + "*.wav")

sess = tf.Session()

file_number = 0
section_size = params.section_size
for f in file_arr:
    ch1_song = np.array([]).astype(float)
    ch2_song = np.array([]).astype(float)

    audio_binary = tf.read_file(f)
    wav_decoder = decode_wav(audio_binary, desired_channels=2)

    sample_rate, audio = sess.run(
        [wav_decoder.sample_rate, wav_decoder.audio])
    audio = np.array(audio)
    
    print(len(audio[:, 0]))
    print(audio.shape)

    a0 = audio[:, 0]
    a1 = audio[:, 1]
    a0 = normalize(a0)
    a1 = normalize(a1)

    if params.overlap:
        s_a0 = segment(a0, params.overlap_size, section_size)
        s_a1 = segment(a1, params.overlap_size, section_size)
    else:
        s_a0 = [a0[i * section_size:(i + 1) * section_size] for i in range((len(a0) + section_size - 1) // section_size)]
        s_a1 = [a1[i * section_size:(i + 1) * section_size] for i in range((len(a1) + section_size - 1) // section_size)] 

    wav_arr_ch1 = []
    wav_arr_ch2 = []

    for a in zip(s_a0, s_a1):
        if len(a[0]) != section_size:
            print(len(a[0]))
            print("Wrong sample")
            continue
        wav_arr_ch1.append(a[0])
        wav_arr_ch2.append(a[1])

    wav_arr_ch1 = np.array(wav_arr_ch1)
    wav_arr_ch2 = np.array(wav_arr_ch2)

    data = np.concatenate((wav_arr_ch1, wav_arr_ch2), axis=1)

    song_wav_arr_ch1 = np.array([])
    song_wav_arr_ch2 = np.array([])

    print("normalized")
    i = 0
    for d in data:
        i += 1
        if len(d) != section_size * 2:
            print(len(d))
            print("wrong sample")
            continue

        # plt.plot(d)
        # plt.show()
        
        merged = np.reshape(d, (1, section_size * 2))
        predicted = autoencoder.predict(merged)
        # predicted = merged
        
        # plt.plot(predicted[0])
        # plt.show()
        
        splitted = np.hsplit(predicted[0], 2)
        
        channel1 = splitted[0]
        channel2 = splitted[1]
        print(ch1_song.shape)
        print(ch2_song.shape)
        ch1_song = np.concatenate((ch1_song, channel1))
        ch2_song = np.concatenate((ch2_song, channel2))
    
    if params.overlap:
        ch1_song = [ch1_song[i : i+section_size] for i in range(0, len(ch1_song), section_size)] # [...] -> [[..], [..], ...]
        ch2_song = [ch2_song[i : i+section_size] for i in range(0, len(ch2_song), section_size)]
        ch1_song = add_overlap(ch1_song, overlap_size) # [[..], [..], ...] -> [...]
        ch2_song = add_overlap(ch2_song, overlap_size)

    # maps sigmoid [0,1] output to [-1,1] for .wav
    ch1_song = ((ch1_song * 2) - 1)
    ch2_song = ((ch2_song * 2) - 1)

    audio_arr = np.hstack(np.array((ch1_song, ch2_song)).T)
    cols = 2
    rows = math.floor(len(audio_arr)/2)
    audio_arr = audio_arr.reshape(rows, cols)

    wav_encoder = ffmpeg.encode_audio(
        audio_arr, file_format='wav', samples_per_second=sample_rate)

    wav_file = sess.run(wav_encoder)

    with open(f"{params.save_test_output_path_start}{file_number}.wav", 'wb') as f:
        f.write(wav_file)

    file_number += 1

print('all done')

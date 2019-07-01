from preprocces import normalize, segment, add_overlap
import matplotlib.pyplot as plt
from autoencoder import compile_model, load_model
import numpy as np
from tensorflow.contrib import ffmpeg
from tensorflow.contrib.framework.python.ops.audio_ops import decode_wav, encode_wav
from glob import iglob
import tensorflow as tf
from scipy.fftpack import rfft, irfft
from tensorflow.keras.models import model_from_json
import math
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


autoencoder = load_model('/content/gdrive/Shared drives/EE838/models/v27/model-1650eps')

file_arr = iglob('/content/gdrive/Shared drives/EE838/test/*.wav')
sess = tf.Session()

section_size = 12348 // 2
OVERLAP_SEGMENTS = True

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
    
    print(len(audio[:, 0]))
    print(audio.shape)

    a0 = audio[:, 0]
    a1 = audio[:, 1]

    a0 = normalize(a0)
    a1 = normalize(a1)

    overlap_size = 1029 # ~1.6% of section_size
    if OVERLAP_SEGMENTS:
        s_a0 = segment(a0, overlap_size, section_size)
        s_a1 = segment(a1, overlap_size, section_size)
        # FIXME AE output transformation into audio
    else:
        s_a0 = [a0[i * section_size:(i + 1) * section_size] for i in range((len(a0) + section_size - 1) // section_size)]
        s_a1 = [a1[i * section_size:(i + 1) * section_size] for i in range((len(a1) + section_size - 1) // section_size)] 

    wav_arr_ch1 = []
    wav_arr_ch2 = []

    for a in zip(s_a0, s_a1):
        if len(a[0]) != section_size:
            print(len(a[0]))
            print("wrong sample")
            continue
        wav_arr_ch1.append(a[0])
        wav_arr_ch2.append(a[1])

    wav_arr_ch1 = np.array(wav_arr_ch1)
    wav_arr_ch2 = np.array(wav_arr_ch2)

    data = np.concatenate((wav_arr_ch1, wav_arr_ch2), axis=1)
    i = 0

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
        
        merged = np.reshape(d, (1, 12348))
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
    
    if OVERLAP_SEGMENTS:
        ch1_song = [ch1_song[i : i+section_size] for i in range(0, len(ch1_song), section_size)] # [...] -> [[..], [..], ...]
        ch2_song = [ch2_song[i : i+section_size] for i in range(0, len(ch2_song), section_size)]
        ch1_song = add_overlap(ch1_song, overlap_size) # [[..], [..], ...] -> [...]
        ch2_song = add_overlap(ch2_song, overlap_size)

    # maps sigmoid [0,1] output to [-1,1] for .wav
    ch1_song = ((ch1_song * 2)-1)
    ch2_song = ((ch2_song * 2)-1)

    audio_arr = np.hstack(np.array((ch1_song, ch2_song)).T)
    cols = 2
    rows = math.floor(len(audio_arr)/2)
    audio_arr = audio_arr.reshape(rows, cols)

    wav_encoder = ffmpeg.encode_audio(
        audio_arr, file_format='wav', samples_per_second=sample_rate)

    wav_file = sess.run(wav_encoder)
    f = open('/content/gdrive/Shared drives/EE838/test_reconstructed/' + str(file_number) + ".wav", 'wb')
    f.write(wav_file)
    f.close()
    file_number += 1
print('all done')

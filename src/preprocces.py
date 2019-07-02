import random
import pickle
from glob import iglob
from functools import reduce

import tensorflow as tf
from tf.contrib.framework.python.ops.audio_ops import decode_wav

from scipy.fftpack import rfft, irfft
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import matplotlib.pyplot as plt


DATA_FILES_WAV = '/content/gdrive/Shared drives/EE838/songs_wav/' # define as wav_data_path_start in args.txt


def normalize(v):
    return (v + 1) / 2 # [-1, 1] -> [0, 1]


def segment(sequence, overlap_size, seg_size):
    delta = seg_size - overlap_size
    return [sequence[d : d + seg_size] for d in range(0, len(sequence) - seg_size + 1, delta)]


def add_overlap(segmented, overlap_size, add_type=-1):
    if add_type < 0:
        func = lambda acc, x: np.concatenate((acc, x[overlap_size:])) # ignores head overlap
    elif add_type == 0:
        # averages overlaps
        func = lambda acc, x: np.concatenate((
            acc[:-overlap_size], 
            [sum(x) / 2 for x in zip(acc[-overlap_size:], x[:overlap_size+1])], 
            x[overlap_size:]))
    else:
        func = lambda acc, x: np.concatenate((acc[:-overlap_size], x)) # ignores tail overlap
    return reduce(func, segmented)


def preprocess_data(batch_size, wav_data_path_start=DATA_FILES_WAV, section_size=12348//2, 
                    overlap=False, overlap_size=1029):
    file_arr = list(iglob(wav_data_path_start + '*.wav'))
    np.random.shuffle(file_arr)
    
    sess = tf.Session()

    wav_arr_ch1 = []
    wav_arr_ch2 = []

    i = 0
    for f in file_arr:
        if i == batch_size:
            break
        i += 1

        audio_binary = tf.read_file(f)
        wav_decoder = decode_wav(audio_binary, desired_channels=2)

        sample_rate, audio = sess.run(
            [wav_decoder.sample_rate, wav_decoder.audio])
        audio = np.array(audio)

        # audio = audio[:5280000]
        # if len(audio[:, 0]) != 5280000:
        #     continue
        print(len(audio[:, 0]))
        print(audio.shape)
        # We want to ensure that every song we look at has the same
        # number of samples!
        
        a0 = audio[:, 0]
        a1 = audio[:, 1]
        a0 = normalize(a0)
        a1 = normalize(a1)

        if overlap:
            s_a0 = segment(a0, overlap_size, section_size)
            s_a1 = segment(a1, overlap_size, section_size)
        else:
            s_a0 = [a0[i * section_size:(i + 1) * section_size] for i in range((len(a0) + section_size - 1) // section_size)] 
            s_a1 = [a1[i * section_size:(i + 1) * section_size] for i in range((len(a1) + section_size - 1) // section_size)] 

        for a in zip(s_a0, s_a1):
            if len(a[0]) != section_size:
                print(len(a[0]))
                print("Wrong sample")
                continue
            wav_arr_ch1.append(a[0])
            wav_arr_ch2.append(a[1])
        print("Returning File: " + f)
        print("Sample rate", sample_rate)

    print("Number of returned chuncks", len(wav_arr_ch1))

    if len(wav_arr_ch1) <= 0:
        print('No data')
        print('Quitting')
        exit()

    sess.close()
    return wav_arr_ch1, wav_arr_ch2, sample_rate

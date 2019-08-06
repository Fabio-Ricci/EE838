import random
import pickle
from glob import iglob
from functools import reduce

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.audio_ops import decode_wav

from scipy.fftpack import rfft, irfft
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import matplotlib.pyplot as plt

# FIXME TODO improve functions performance/readability after testing

'''
    segment_t(sequence, seg_size) is equal to preprocess.segment
    with overlap_size = seg_size/2, delta = seg_size/2
'''
def segment_t(sequence, seg_size):
    assert(seg_size % 2 == 0) # TODO treat odd cases
    return [sequence[d : d + seg_size] for d in range(0, len(sequence) - seg_size + 1, seg_size / 2)]

'''
    adds a segmented wave modulated by a triangular function
'''
def add_overlap_t(segmented, seg_size):
    assert(seg_size % 2 == 0) # TODO treat odd cases
    half_seg_size = seg_size // 2

    # assert(len(segmented[0]) == seg_size)
    result_wave = np.array(trianglify(segmented[0], up=False))

    for i in range(1, len(segmented) - 1):
        # assert(len(segmented[i]) == seg_size)
        t = trianglify(segmented[i])
        result_wave[-half_seg_size:] += t[:half_seg_size]
        result_wave = np.append(result_wave, t[half_seg_size:])

    # assert(len(segmented[-1]) == seg_size)
    t = trianglify(segmented[-1], down=False)
    result_wave[-half_seg_size:] += t[:half_seg_size]
    result_wave = np.append(result_wave, t[half_seg_size:])
    return result_wave

'''
    multiplies a wave segment by a triangular function with height = 1 and width = seg_size
'''
def trianglify(segment, up=True, down=True):
    assert(len(segment) % 2 == 0) # TODO treat odd cases
    half_seg_size = len(segment) // 2
    # dx = 1
    dy = 1 / half_seg_size # 2 / len(segment)

    ascent = np.linspace(dy, 1-dy, half_seg_size) # np.linspace(0, 1, half_seg_size)
    descent = np.ones(ascent.shape) - ascent

    # 1st half, x in [0, w/2): f(x) = x / (w/2)
    if up: segment[:half_seg_size] *= ascent

    # 2nd half, x in [w/2, w]: f(x) = 2 - x / (w/2)
    if down: segment[half_seg_size:] *= descent

    return segment


DATA_FILES_WAV = '/content/gdrive/Shared drives/EE838/songs_wav/' # define as wav_data_path_start in args.txt


def normalize(v):
    return (v + 1) / 2 # [-1, 1] -> [0, 1]


def preprocess_data(batch_size, wav_data_path_start=DATA_FILES_WAV, section_size=12348//2, overlap=False):
    file_arr = list(iglob(wav_data_path_start + "*.wav"))
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

        print(len(audio[:, 0]))
        print(audio.shape)
        
        a0 = audio[:, 0]
        a1 = audio[:, 1]
        a0 = normalize(a0)
        a1 = normalize(a1)

        if overlap:
            s_a0 = segment_t(a0, section_size)
            s_a1 = segment_t(a1, section_size)
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

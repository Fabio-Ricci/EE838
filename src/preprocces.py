from scipy.fftpack import rfft, irfft
from tensorflow.contrib.framework.python.ops import audio_ops
'''
curr_batch - The current batch of the training data we are looking at.
songs_per_batch - How songs we want to load in per batch
sess - Our TensorFlow session object
'''


def get_next_batch(curr_batch, songs_per_batch, sess):
    wav_arr_ch1 = []
    wav_arr_ch2 = []
    if (curr_batch) >= (len(file_arr)):
        curr_batch = 0
    start_position = curr_batch * songs_per_batch
    end_position = start_position + songs_per_batch
    for idx in range(start_position, end_position):
        audio_binary = tf.read_file(file_arr[idx])
        wav_decoder = audio_ops.decode_wav(
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
    print("Returning File: " + file_arr[idx])

    return wav_arr_ch1, wav_arr_ch2, sample_rate

from preprocces import preprocess_data
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

wav_arr_ch1, wav_arr_ch2, sample_rate = preprocess_data()
wav_arr_ch1 = np.array(wav_arr_ch1)
wav_arr_ch2 = np.array(wav_arr_ch2)

data = np.concatenate((wav_arr_ch1, wav_arr_ch2), axis=1)


print(len(data[0]))

# inputs = 12348
# hidden_1_size = 8400
# hidden_2_size = 3440
# hidden_3_size = 2800
# batch_size = 50
# lr = 0.0001
# l2 = 0.0001

# this is the size of our encoded representations
encoding_dim = 2800  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

input_img = Input(shape=(12348,))
encoded = Dense(8400, activation='relu')(input_img)
encoded = Dense(3440, activation='relu')(encoded)
encoded = Dense(2800, activation='relu')(encoded)

decoded = Dense(3440, activation='relu')(encoded)
decoded = Dense(8400, activation='relu')(decoded)
decoded = Dense(12348, activation='relu')(decoded)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


autoencoder.fit(data, data,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(data, data))


encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

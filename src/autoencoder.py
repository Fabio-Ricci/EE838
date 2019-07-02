import os
import gc
import sys
import ast
import datetime

import tensorflow as tf
from tf.keras.models import Model, model_from_json
from tf.keras.layers import Input, Dense
from tf.keras.callbacks import ModelCheckpoint
from tf.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

from preprocces import preprocess_data

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def compile_model(model, lr=0.0001):
    model.compile(optimizer=Adam(lr=lr), loss='mse')
    return model


def load_model(name):
    with open(f"{name}.json", 'r') as json_file:
        loaded_model = model_from_json(json_file.read())

    # load weights into new model
    loaded_model.load_weights(f"{name}.h5")
    loaded_model = compile_model(loaded_model)

    return loaded_model


def save_model(model, name):
    # serialize model to JSON
    model_json = model.to_json()

    if os.path.isfile(f"{name}.json"):
        name = f"{name}-{datetime.datetime.now()}"

    path = '/'.join(name.split('/')[:-1])
    if not os.path.exists(path):
        os.makedirs(path)
    
    with open(f"{name}.json", 'w') as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(f"{name}.h5")


def create_graphs(history, name=''):
    '''
    ref.: http://flothesof.github.io/convnet-face-keypoint-detection.html#Towards-more-complicated-models
    '''
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.tight_layout()

    curr_time = datetime.datetime.now()
    if name == '':
        name = f"graphs/{curr_time}"
    
    path = '/'.join(name.split('/')[:-1])
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f"{name}-training-info.png")


def parse_args():
    args_dict = {}
    try:
        input_dict_file = sys.argv[1]
        with open(input_dict_file, 'r') as f:
            # expects a file with a dictionary
            args_dict = ast.literal_eval(f.read())
    except:
        pass
    return args_dict


if __name__ == "__main__":
    args_dict = parse_args()
    
    load = True
    if load:
        full_path = '/content/gdrive/Shared drives/EE838/models/v27/model-2265eps'
        autoencoder = load_model(full_path)
        print(f"Model loaded succesfully from \'{full_path}\'")
    else:
        input_img = Input(shape=(12348,))
        encoded = Dense(8400, activation='relu')(input_img)
        encoded = Dense(5000, activation='relu')(encoded)

        encoded = Dense(4000, activation='relu')(encoded)

        decoded = Dense(5000, activation='relu')(encoded)
        decoded = Dense(8400, activation='relu')(decoded)
        decoded = Dense(12348, activation='sigmoid')(decoded)

        autoencoder = Model(input_img, decoded)
        autoencoder = compile_model(autoencoder)

    # checkpoint
    # filepath="weights-improvement-{epoch:02d}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, verbose=1, mode='max', period=50)
    callbacks_list = []  # [checkpoint]
    scores = []

    num_epochs = 100
    initial_epoch = 2265
    for i in range(30000):  # 100 epochs = 0.56h = 34 min
        gc.collect()

        wav_arr_ch1, wav_arr_ch2, sample_rate = preprocess_data(30)
        wav_arr_ch1 = np.array(wav_arr_ch1)
        wav_arr_ch2 = np.array(wav_arr_ch2)

        data = np.concatenate((wav_arr_ch1, wav_arr_ch2), axis=1)
        del(wav_arr_ch1, wav_arr_ch2)

        # fit the model
        epochs = (i+1) * num_epochs + initial_epoch
        history = autoencoder.fit(data, data,
                                  epochs=epochs,
                                  shuffle=True,
                                  callbacks=callbacks_list,
                                  batch_size=64,
                                  validation_split=0.15,
                                  initial_epoch=epochs - num_epochs)

        score = autoencoder.evaluate(data, data, verbose=0)
        scores.append(score)
        print(f"Test loss: {score}")

        # NOTE v27 uses overlapping segments
        name = f"/v27/model-{epochs}eps"
        save_model(autoencoder, f"/content/gdrive/Shared drives/EE838/models{name}")
        create_graphs(history, f"/content/gdrive/Shared drives/EE838/graphs{name}")

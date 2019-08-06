import os
import gc
import sys
import ast
import datetime

import tensorflow as tf
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

# from preprocces import preprocess_data
from preproccess_experimental import preprocess_data

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def compile_model(model, lr=0.0001):
    model.compile(optimizer=Adam(lr=lr), loss='mse')
    return model


def load_model(name, lr):
    with open(f"{name}.json", 'r') as json_file:
        loaded_model = model_from_json(json_file.read())

    # load weights into new model
    loaded_model.load_weights(f"{name}.h5")
    loaded_model = compile_model(loaded_model, lr)

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
    args_dict = None
    try:
        print(f"argv={sys.argv}")
        input_dict_file = sys.argv[1]
        with open(input_dict_file, 'r') as f:
            # expects a file with a dictionary
            args_dict = ast.literal_eval(f.read())
    except:
        pass
    return args_dict


class Params():
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])

    def __repr__(self):
        attrs = str([attr for attr in self.__dict__])
        return "<Params: %s>" % attrs


def default_args_dict():
    return {
        'section_size': 6174,
        'overlap_sections': True,
        'overlap_section_size': 1029,
        'wav_train_data_path_start': '/content/gdrive/Shared drives/EE838/songs_wav/',
        'wav_test_data_path_start': '/content/gdrive/Shared drives/EE838/test/',

        'save_test_output_path_start': '/content/gdrive/Shared drives/EE838/test_reconstructed/',
        'save_model_path_start': '/content/gdrive/Shared drives/EE838/models/',
        'save_graph_path_start': '/content/gdrive/Shared drives/EE838/graphs/',
    
        'load_model': True,
        'load_model_path_start': '/content/gdrive/Shared drives/EE838/models/',
        'load_model_path_version': 'v27/',

        'preprocess_batch_size': 30,
        
        'initial_epoch': 2265,
        'num_epochs': 100,
        'learning_rate': 0.0001,
        'gradient_batch_size': 64,
        'validation_split': 0.15,
        
        'layer_dim_io': 12348,
        'layer_dim_hidden_large': 8400,
        'layer_dim_hidden_small': 5000,
        'layer_dim_code': 4000
    }


if __name__ == "__main__":
    args_dict = parse_args()
    if args_dict == None:
        # define the default parameters to be used
        args_dict = default_args_dict()
    
    params = Params(args_dict)
    
    if params.load_model:
        full_path = params.load_model_path_start
        full_path += params.load_model_path_version
        full_path += f"model-{params.initial_epoch}eps"

        autoencoder = load_model(full_path, lr=params.learning_rate)
        print(f"Model loaded succesfully from \'{full_path}\'")

    else:
        input_img = Input(shape=(params.layer_dim_io,))
        encoded = Dense(params.layer_dim_hidden_large, activation='relu')(input_img)
        encoded = Dense(params.layer_dim_hidden_small, activation='relu')(encoded)

        encoded = Dense(params.layer_dim_code, activation='relu')(encoded)

        decoded = Dense(params.layer_dim_hidden_small, activation='relu')(encoded)
        decoded = Dense(params.layer_dim_hidden_large, activation='relu')(decoded)
        decoded = Dense(params.layer_dim_io, activation='sigmoid')(decoded)

        autoencoder = Model(input_img, decoded)
        autoencoder = compile_model(autoencoder, lr=params.learning_rate)

    # checkpoint
    # filepath="weights-improvement-{epoch:02d}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, verbose=1, mode='max', period=50)
    callbacks_list = []  # [checkpoint]
    scores = []

    for i in range(30000):
        gc.collect()

        wav_arr_ch1, wav_arr_ch2, sample_rate = preprocess_data(
            params.preprocess_batch_size, 
            wav_data_path_start=params.wav_train_data_path_start,
            overlap=params.overlap_sections) # TODO test if overlapping leads to better results

        wav_arr_ch1 = np.array(wav_arr_ch1)
        wav_arr_ch2 = np.array(wav_arr_ch2)

        data = np.concatenate((wav_arr_ch1, wav_arr_ch2), axis=1)
        del(wav_arr_ch1, wav_arr_ch2)

        # fit the model
        epochs = (i+1) * params.num_epochs + params.initial_epoch
        history = autoencoder.fit(data, data,
                                  epochs=epochs,
                                  shuffle=True,
                                  callbacks=callbacks_list,
                                  batch_size=params.gradient_batch_size,
                                  validation_split=params.validation_split,
                                  initial_epoch=epochs - params.num_epochs)

        score = autoencoder.evaluate(data, data, verbose=0)
        scores.append(score)
        print(f"Test loss: {score}")

        # NOTE v27 uses overlapping segments
        save_name = params.load_model_path_version + f"model-{epochs}eps"
        save_model(autoencoder, params.save_model_path_start + save_name)
        create_graphs(history, params.save_graph_path_start + save_name)

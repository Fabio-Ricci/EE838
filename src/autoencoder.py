import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf
import numpy as np
from preprocces import preprocess_data
import os
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def compile_model(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='mse')
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


def save_model(model, name):
    # serialize model to JSON
    model_json = model.to_json()
    if os.path.isfile(name+'.json'):
        name = name+'-'+str(datetime.datetime.now())
    if not os.path.exists('/'.join(name.split('/')[:-1])):
        os.makedirs('/'.join(name.split('/')[:-1]))
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name + ".h5")


# http://flothesof.github.io/convnet-face-keypoint-detection.html
def create_graphs(history, name=''):
    # loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.tight_layout()
    if name == '':
        name = 'graphs/'+str(datetime.datetime.now())
    if not os.path.exists('/'.join(name.split('/')[:-1])):
        os.makedirs('/'.join(name.split('/')[:-1]))
    plt.savefig(name+'-training-info.png')


if __name__ == "__main__":

    wav_arr_ch1, wav_arr_ch2 = preprocess_data()
    wav_arr_ch1 = np.array(wav_arr_ch1)
    wav_arr_ch2 = np.array(wav_arr_ch2)

    data = np.concatenate((wav_arr_ch1, wav_arr_ch2), axis=1)
    plt.plot(data[10])
    plt.show()
    del(wav_arr_ch1, wav_arr_ch2)
    print(len(data[0]))
    

    # inputs = 12348
    # hidden_1_size = 8400
    # hidden_2_size = 3440
    # hidden_3_size = 2800
    # batch_size = 50
    # lr = 0.0001
    # l2 = 0.0001

    # this is the size of our encoded representations
    # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    encoding_dim = 2800
    load = False

    if load:
        autoencoder = load_model(
            '/content/gdrive/My Drive/models/v16/model-100eps')
        print("model loaded succesfully")
    else:
        input_img = Input(shape=(12348,))
        encoded = Dense(12348, activation='sigmoid')(input_img)
        # encoded = Dense(8000, activation='sigmoid')(encoded)
        # encoded = Dense(6000, activation='sigmoid')(encoded)

        # decoded = Dense(5000, activation='sigmoid')(encoded)
        decoded = Dense(12348, activation='sigmoid')(encoded)

        autoencoder = Model(input_img, decoded)
        autoencoder = compile_model(autoencoder)

    # checkpoint
    # filepath="weights-improvement-{epoch:02d}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, verbose=1, mode='max', period=50)
    callbacks_list = []  # [checkpoint]

    for i in range(100):  # 100 epochs = 0.56h = 34 min
        

        initial_epoch = 0
        epochs = 50
        # Fit the model
        history = autoencoder.fit(data, data,
                                  validation_split=0.20,
                                  batch_size=1024,
                                  epochs=epochs,
                                  shuffle=True,
                                  callbacks=callbacks_list)

        score = autoencoder.evaluate(data, data, verbose=0)
        print('Test loss:', score)

        name = '/v16/model-'+str(((i+1)*epochs)+initial_epoch)+'eps'
        save_model(autoencoder, '/content/gdrive/My Drive/models'+name)
        create_graphs(history, '/content/gdrive/My Drive/graphs'+name)
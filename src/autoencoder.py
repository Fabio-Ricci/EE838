import matplotlib.pyplot as plt
import datetime
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model, model_from_json
from tensorflow.python.keras.layers import Input, Dense
import tensorflow as tf
import numpy as np
from preprocces import preprocess_data
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def compile_model(model):
    model.compile(optimizer=tf.train.AdamOptimizer(0.0001), loss='mse')
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
            '/content/gdrive/My Drive/models/v6/model-700eps')
        print("model loaded succesfully")
    else:
        input_img = Input(shape=(12348,))
        encoded = Dense(8400, activation='relu')(input_img)
        encoded = Dense(3440, activation='relu')(encoded)
        encoded = Dense(2800, activation='relu')(encoded)

        decoded = Dense(3440, activation='relu')(encoded)
        decoded = Dense(8400, activation='relu')(decoded)
        decoded = Dense(12348, activation='relu')(decoded)

        autoencoder = Model(input_img, decoded)
        autoencoder = compile_model(autoencoder)

    wav_arr_ch1, wav_arr_ch2, sample_rate = preprocess_data(
        tf.keras.backend.get_session())
    wav_arr_ch1 = np.array(wav_arr_ch1)
    wav_arr_ch2 = np.array(wav_arr_ch2)

    # checkpoint
    # filepath="weights-improvement-{epoch:02d}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, verbose=1, mode='max', period=50)
    callbacks_list = []  # [checkpoint]
    TPU_ADDRESS = ''
    try:
        device_name = os.environ['COLAB_TPU_ADDR']
        TPU_ADDRESS = 'grpc://' + device_name
        print('Found TPU at: {}'.format(TPU_ADDRESS))
    except KeyError:
        print('TPU not found')

    autoencoder = tf.contrib.tpu.keras_to_tpu_model(
        autoencoder,
        strategy=tf.contrib.tpu.TPUDistributionStrategy(
            tf.contrib.cluster_resolver.TPUClusterResolver(
                tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
        )
    )

    print('lalala')

    for i in range(100):  # 100 epochs = 0.56h = 34 min
        print('lalala2')

        data = np.concatenate((wav_arr_ch1, wav_arr_ch2), axis=1)
        del(wav_arr_ch1, wav_arr_ch2, sample_rate)
        print(len(data[0]))

        def train_input_fn(batch_size=1024):
            # Convert the inputs to a Dataset.
            dataset = tf.data.Dataset.from_tensor_slices((data, data))
        # Shuffle, repeat, and batch the examples.
            dataset = dataset.cache()
            dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
            dataset = dataset.repeat()
            dataset = dataset.batch(batch_size, drop_remainder=True)
        # Return the dataset.
            return dataset

        initial_epoch = 700
        epochs = 50
        # Fit the model
        history = autoencoder.fit(data, data,
                                  batch_size=128*4,
                                  validation_split=0.20,
                                  epochs=epochs,
                                  callbacks=callbacks_list,
                                  shuffle=Ture,
                                  steps_per_epoch=60)

        score = autoencoder.evaluate(data, data, verbose=0, batch_size=128 * 8)
        print('Test loss:', score)

        name = '/v6/model-'+str(((i+1)*epochs)+initial_epoch)+'eps'
        save_model(autoencoder, '/content/gdrive/My Drive/models'+name)
        create_graphs(history, '/content/gdrive/My Drive/graphs'+name)

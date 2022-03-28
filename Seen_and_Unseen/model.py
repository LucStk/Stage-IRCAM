from turtle import backward
import tensorflow as tf
import tensorflow.keras.activations  as act
import tensorflow.keras as ks
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
import numpy as np

MAX = 3
MIN = -14
class Encodeur(tf.keras.Model):
  def __init__(self):
    super(Encodeur, self).__init__()
    
    act_rnn = act.relu
    #Couche convolutions
    self.conv = tf.keras.models.Sequential([
        layers.Masking(mask_value=0.),
        layers.Conv1D(32, 4, activation=act_rnn),
        layers.MaxPool1D(pool_size=2),
        layers.BatchNormalization(),
        layers.Conv1D(64, 4, activation=act_rnn),
        layers.MaxPool1D(pool_size=2),
        layers.BatchNormalization()
    ])
    self.lstm_1  = layers.LSTM(128, activation = act_rnn, return_sequences = True)
    #Réseau bi-LSTM
    self.lstm_fw = layers.LSTM(200, activation = act_rnn)
    self.bi_lstm = layers.Bidirectional(self.lstm_fw)
  
    self.latent  = layers.Dense(64)

  def call(self, x):
    x = self.conv(x)
    epoch_lstm = x.shape[1]
    #x = self.lstm_1(x)
    x = self.bi_lstm(x)
    #x = self.latent(x)
    return x, epoch_lstm

class Decodeur(tf.keras.Model):
  def __init__(self):
    super(Decodeur, self).__init__()
    act_rnn = act.relu
    
    self.lstm_1  = layers.LSTM(64, activation = act_rnn)
    # Réseau bi lstm
    self.lstm_fw = layers.LSTM(80, activation = act_rnn)
    self.bi_lstm = layers.Bidirectional(self.lstm_fw, return_sequences = True)
    
    self.convT = tf.keras.models.Sequential([
        layers.UpSampling1D(size=2),
        layers.Conv1DTranspose(32, 4, activation=act_rnn),
        layers.BatchNormalization(),
        layers.UpSampling1D(size=2),
        layers.Conv1DTranspose(80, 4, activation=act_rnn)
    ])    

  def call(self, x, step):
    """
    Génére une sortie de taille size à partir d'un espace latent (batch, latent_size)
    """
    x = tf.expand_dims(x, axis = 1)
    x_shape = x.shape
    ret = []
    for _ in range(step): #Expand dimension
        x = self.lstm_1(x)
        ret.append(tf.expand_dims(x, axis = 1))
        x = tf.ones(x_shape)
    x = tf.concat(ret, axis = 1)
    x = self.bi_lstm(x)
    x = self.convT(x)
    x = ks.activations.sigmoid(x)*(MAX-MIN)+MIN
    return x

if __name__ == "__main__":
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


    from utilitaires.dataloader_ESD_tf import ESD_data_generator
    encodeur = Encodeur()
    decodeur = Decodeur()
    FILEPATH = r"/home/luc/Documents/STAGE_IRCAM/data/ESD_Mel/"
    base = ESD_data_generator(FILEPATH, batch_size=10, shuffle=True, langage="english")
    for x,y in base:
        tmp = x
        tmp = tf.transpose(tmp, perm = [0,2,1])#batch, lenght, n
        out, step = encodeur(tmp)
        rec = decodeur(out, step)
        print(rec.shape, tmp.shape)
    print("ok")    
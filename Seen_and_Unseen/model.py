from turtle import backward
import tensorflow as tf
import tensorflow.keras.activations  as act
import tensorflow.keras.layers as layers
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
import numpy as np
LATENT_SPACE = 10



class Auto_encodeur(tf.keras.Model):
  def __init__(self):
    super(Encodeur, self).__init__()
    self.lstm = layers.LSTM(30, activation = act.tanh)

  def call(self, x):
    x = self.lstm(x)
    return x

class Encodeur(tf.keras.Model):
  def __init__(self):
    super(Encodeur, self).__init__()
    act_rnn = act.tanh

    self.lstm_1  = layers.LSTM(80, activation = act_rnn, return_sequences = True)
    self.lstm_fw = layers.LSTM(80)
    self.lstm_bw = layers.LSTM(80, activation = act_rnn, go_backwards = True)
    self.bi_lstm = layers.Bidirectional(self.lstm_fw, backward_layer=self.lstm_bw)
    self.latent  = layers.Dense(128)


  def call(self, x):
    x = self.lstm_1(x)
    x = self.bi_lstm(x)
    x = self.latent(x)
    return x

class Decodeur(tf.keras.Model):
  def __init__(self):
    super(Decodeur, self).__init__()
    act_rnn = act.relu
    self.lstm_1  = layers.LSTM(80, activation = act_rnn)
    self.lstm_fw = layers.LSTM(80, activation = act_rnn)
    self.lstm_bw = layers.LSTM(80, activation = act_rnn, go_backwards=True)
    self.bi_lstm = layers.Bidirectional(self.lstm_fw, backward_layer=self.lstm_bw)
    self.latent  = layers.Dense(128)


  def call(self, x, size):
    """
    Génére une sortie de taille size à partir d'un espace latent (batch, latent_size)
    """
    x = self.latent(x)
    x = tf.expand_dims(x, axis = 1)
    x_shape = x.shape
    ret = []
    for _ in range(size):
      x = self.bi_lstm(x)
      x = self.lstm_1(tf.expand_dims(x, axis = 1))
      ret.append(tf.expand_dims(x, axis=0))
      x = tf.ones(x_shape)

    return tf.transpose(tf.concat(ret, axis = 0), perm = [1,0,2])

import tensorflow as tf
import tensorflow.keras.activations  as act
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv3D, LSTM
from tensorflow.keras import Model
import numpy as np
LATENT_SPACE = 10


class Encodeur(tf.keras.Model):
  def __init__(self):
    super(Encodeur, self).__init__()
    self.lstm = LSTM(30, activation = act.tanh)

  def call(self, x):
    x = self.lstm(x)
    return x

class Auto_encodeur(tf.keras.Model):
  def __init__(self):
    super(Encodeur, self).__init__()
    self.lstm = LSTM(30, activation = act.tanh)

  def call(self, x):
    x = self.lstm(x)
    return x

class Decodeur(tf.keras.Model):
  def __init__(self):
    super(Decodeur, self).__init__()
    self.lstm = LSTM(80, activation = None)

  def call(self, x, size):
    """
    Génére une sortie de taille size à partir d'un espace latent (batch, latent_size)
    """
    #x = self.h3(x)
    x = tf.expand_dims(x, axis = 1)
    ret = []
    for i in range(size):
      ret += [tf.expand_dims(self.lstm(x), axis=0)]
      x = tf.ones(x.shape)
    return tf.transpose(tf.concat(ret, axis = 0), perm = [1,0,2])

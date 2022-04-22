import tensorflow as tf
import tensorflow.keras.activations  as act
import tensorflow.keras as ks
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from utilitaires.utils import *
from utilitaires.All_model import Auto_Encodeur_rnn
import numpy as np
MAX = 3
MIN = -14
MEAN_DATASET  = -6.0056405
STD_DATASET   =  2.4420118
MAX_normalised = (MAX-MEAN_DATASET)/STD_DATASET
MIN_normalised = (MIN-MEAN_DATASET)/STD_DATASET


"""
IMPLEMENTATION Papier Seen and Unseen
"""
class Encodeur_SAU(tf.keras.Model):
  def __init__(self):
    super(Encodeur_SAU, self).__init__()
    act_conv = act.relu
    self.conv = tf.keras.models.Sequential([
        layers.Masking(mask_value=0.),
        
        layers.Conv1D(8, 4, activation=act_conv),
        layers.MaxPool1D(pool_size=2),
        layers.BatchNormalization(),
        
        layers.Conv1D(16, 4, activation=act_conv),
        layers.MaxPool1D(pool_size=2),
        layers.BatchNormalization(),
        
        layers.Conv1D(32, 4, activation=act_conv),
        layers.MaxPool1D(pool_size=3),
        layers.BatchNormalization(),

        layers.Conv1D(64, 4, activation=act_conv),
    ])

  def call(self, x):
    x = tf.expand_dims(x, axis = -1)
    x = self.conv(x)
    return x
    

class Decodeur_SAU(tf.keras.Model):
  def __init__(self):
    super(Decodeur_SAU, self).__init__()
    act_conv = act.elu
    self.convT = tf.keras.models.Sequential([
        layers.UpSampling1D(size=2),
        layers.Conv1DTranspose(64, 3, activation=act_conv),
        layers.BatchNormalization(),

        layers.UpSampling1D(size=2),
        layers.Conv1DTranspose(32, 4, activation=act_conv),
        layers.BatchNormalization(),
        
        layers.UpSampling1D(size=2),
        layers.Conv1DTranspose(16, 4, activation=act_conv),
        layers.BatchNormalization(),

        layers.UpSampling1D(size=3),
        layers.Conv1DTranspose(8, 3, activation=act_conv),
        layers.BatchNormalization(),

        layers.Conv1DTranspose(1, 4, activation=act_conv),
    ])
  def call(self, x):
    """
    Génére une sortie de taille size à partir d'un espace latent (batch, latent_size)
    """
    x = self.convT(x)
    x = tf.squeeze(x)
    x = tf.math.sigmoid(x)*(MAX_normalised-MIN_normalised)+MIN_normalised
    return x

class Auto_Encodeur_SAU(Auto_Encodeur_rnn):
  def __init__(self):
    super(Auto_Encodeur_SAU, self).__init__()
    self.encodeur = Encodeur_SAU()
    self.decodeur = Decodeur_SAU()


  def call(self,x, phi):
    """
    x   : (b*lenght, 128) non normalisé
    phi : (b*lenght, 128)
    """
    phi    = tf.expand_dims(phi, axis=1) #(b*lenght, 1, 128)
    latent = self.encodeur(x)#(b*lenght, 1, 128)
    latent = tf.concat((phi,latent), axis = 2)
    out    = self.decodeur(latent)
    return out

class Discriminator_SAU(tf.keras.Model):
  def __init__(self):
    super(Discriminator_SAU, self).__init__()
    
    act_conv  = act.elu
    act_dense = act.elu
    self.conv = tf.keras.models.Sequential([
        layers.Masking(mask_value=0.),
        
        layers.Conv1D(8, 4, activation=act_conv),
        layers.MaxPool1D(pool_size=4),
        layers.BatchNormalization(),
        layers.Dropout(.2),
        
        layers.Conv1D(16, 4, activation=act_conv),
        layers.MaxPool1D(pool_size=4),
        layers.BatchNormalization(),
        layers.Dropout(.2),
        
        layers.Conv1D(32, 4, activation=act_conv),
    ])
    self.H = tf.keras.models.Sequential([
      layers.Flatten(),
      layers.Dense(30, activation=act_dense),
      layers.Dense(1, activation=act.sigmoid),
    ])

  def save_weights(self, file, step):
    super().save_weights(file+'/Discriminator/'+str(step))

  def load_weights(self, file,  step = None):
    if step is None:
      f = tf.train.latest_checkpoint(file+'/Discriminator')
    else:
      f = file+'/Discriminator/'+str(step)
    super().load_weights(f)

  def call(self, x):
    """
    x : (b*lenght, 80)
    """
    x = tf.expand_dims(x, axis=-1)
    x = self.conv(x)
    x = self.H(x)
    return x



class SER(tf.keras.Model):
  def __init__(self):
    super(SER, self).__init__()
    act_rnn  = act.tanh
    act_conv = act.elu
    act_dens = act.elu

    self.conv = tf.keras.models.Sequential([
        layers.Masking(mask_value=0.),

        layers.Conv2D(8, 5, activation=act_conv),
        layers.MaxPool2D(pool_size=2),
        layers.BatchNormalization(),
        layers.Dropout(.2),

        layers.Conv2D(16, 5, activation=act_conv),
        layers.MaxPool2D(pool_size=2),
        layers.BatchNormalization(),
        layers.Dropout(.2),
        
        layers.Conv2D(32, 5, activation=act_conv),
        layers.MaxPool2D(pool_size=2),
        layers.BatchNormalization(),
        layers.Dropout(.2),
        
        layers.Conv2D(64, 5,activation=act_conv),
    ])
    self.bi_lstm = layers.Bidirectional(
                            layers.GRU(64, 
                                        activation = act_rnn, 
                                        return_sequences=True,
                                        dropout=0.2))  
 
    self.W = layers.Dense(1)
    self.H = tf.keras.models.Sequential([
      layers.Dense(5),])

  def transformer_block(self, x):
    pass

  def call_latent(self, x):
    x = tf.expand_dims(x, axis=-1)
    x = self.conv(x)
    x = tf.reshape(x, (x.shape[0], -1, 128))
    x = self.bi_lstm(x)
    x = self.attention(x)
    return x

  def attention(self,x):
    alpha = tf.keras.activations.softmax(self.W(x), axis = 1)
    alpha = tf.repeat(alpha, repeats=x.shape[-1], axis = -1)
    x     = tf.math.reduce_sum(tf.math.multiply(x,alpha), axis = 1)  
    return x
  
  def call(self, x):
    x = self.call_latent(x)
    x = self.H(x)    
    return x

  def save_weights(self, file, step):
    super().save_weights(file+'/SER/'+str(step))

  def load_weights(self, file,  step = None):
    if step is None:
      f = tf.train.latest_checkpoint(file+'/SER/')
    else:
      f = file+'/SER/'+str(step)
    super().load_weights(f)

class little_SER(SER):
  def __init__(self):
    super(little_SER, self).__init__()
    act_rnn  = act.elu
    act_conv = act.elu
    act_dens = act.elu

    self.conv = tf.keras.models.Sequential([
        layers.Masking(mask_value=0.),

        layers.Conv2D(4, 5, activation=act_conv),
        layers.MaxPool2D(pool_size=2),
        layers.BatchNormalization(),
        layers.Dropout(.2),

        layers.Conv2D(8, 5, activation=act_conv),
        layers.MaxPool2D(pool_size=2),
        layers.BatchNormalization(),
        layers.Dropout(.2),
        
        layers.Conv2D(16, 5, activation=act_conv),
        layers.MaxPool2D(pool_size=2),
        layers.BatchNormalization(),
        layers.Dropout(.2),
        
        layers.Conv2D(32, 5,activation=act_conv),
    ])
    self.lstm_1  = layers.GRU(128, 
                              activation = act_rnn, 
                              return_sequences = True,
                              dropout=0.2)

    self.bi_lstm = layers.Bidirectional(
                            layers.GRU(32, 
                                        activation = act_rnn, 
                                        return_sequences=True,
                                        dropout=0.2))  
 
    self.W = layers.Dense(1)
    self.H = tf.keras.models.Sequential([
      layers.Dense(5),])


class SAU_GAN(tf.keras.Model):
  def __init__(self):
    super(SAU_GAN, self).__init__()
    self.ae = Auto_Encodeur_SAU()
    self.encodeur = Encodeur_SAU()
    self.decodeur = Decodeur_SAU()

  def compile(self, ae_optim, disc_optim, loss):
    super(SAU_GAN, self).compile()
    self.ae_optim   = ae_optim
    self.disc_optim = disc_optim
    self.loss       = loss

  def train_step(self, input):
    x   = input[:,:80]
    phi = input[:,80:]
    x = normalisation(x)

    with tf.GradientTape() as tape_gen:
      phi    = tf.expand_dims(phi, axis=1) #(b*lenght, 1, 128)
      latent = self.encodeur(x)#(b*lenght, 1, 128)
      l_gen  = self.loss(x, tf.ones_like(x))
      """
      latent = tf.concat((phi,latent), axis = 2)
      out    = self.decodeur(latent)
      l_gen  = self.loss(x, out)
      """
    tr_variables = self.encodeur.trainable_variables
    #tr_variables = [*self.decodeur.trainable_variables,*self.encodeur.trainable_variables]

    grad_gen = tape_gen.gradient(l_gen, tr_variables)
    self.ae_optim.apply_gradients(zip(grad_gen, tr_variables))
    return {"l_gen":l_gen}
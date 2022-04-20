import tensorflow as tf
import tensorflow.keras.activations  as act
import tensorflow.keras as ks
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
import numpy as np

# Valeurs observés dans les data
MAX = 3
MIN = -14
MEAN_DATASET  = -6.0056405
STD_DATASET   =  2.4420118
MAX_normalised = (MAX-MEAN_DATASET)/STD_DATASET
MIN_normalised = (MIN-MEAN_DATASET)/STD_DATASET


"""
Decodeur et Encodeur_rnn Convolutif/RNN
"""

def conv_shape(Hin, kernel, p_max = 1, stride =1, pad = 0, dil = 1,):
  a = Hin + 2*pad -dil*(kernel-1) -1
  a = np.floor((a/stride)+1)
  a = np.floor(a/p_max)
  return a

def deconv_shape(rows, strides, k, p = 0):
  return (rows-1)*strides + k - 2*p


class Encodeur_rnn(tf.keras.Model):
  def __init__(self):
    super(Encodeur_rnn, self).__init__()
    act_rnn  = act.elu
    act_conv = act.elu

    self.conv = tf.keras.models.Sequential([
        layers.Masking(mask_value=0.),
        layers.Conv1D(8, 5, activation=act_conv),
        layers.MaxPool1D(pool_size=2),
        layers.BatchNormalization(),
        layers.Conv1D(16, 5, activation=act_conv),
        layers.MaxPool1D(pool_size=2),
        layers.BatchNormalization(),
        layers.Conv1D(32, 5, activation=act_conv),
        layers.MaxPool1D(pool_size=2),
    ])
    self.lstm_1  = layers.GRU(64, activation = act_rnn, return_sequences = True)
    self.bi_lstm = layers.Bidirectional(layers.LSTM(128, activation = act_rnn))  
    self.latent  = layers.Dense(64)

  def call(self, x):
    #->pi[Encodeur_rnn-call]
    x = self.conv(x)
    epoch_lstm = x.shape[1]
    x = self.lstm_1(x)
    x = self.bi_lstm(x)
    #x = self.latent(x)
    #<-
    return x, epoch_lstm

class Decodeur_rnn(tf.keras.Model):
  def __init__(self):
    super(Decodeur_rnn, self).__init__()
    #->pi[activation]
    act_rnn  = act.elu
    act_conv = act.elu
    #<-   

    #->pi[lstm-Decodeur_rnn]
    self.lstm_1  = layers.GRU(64, activation = act_rnn)
    self.bi_lstm = layers.Bidirectional(layers.LSTM(32, activation = act_rnn, return_sequences=True))
    #<-
    #->pi[deconv]
    self.convT = tf.keras.models.Sequential([
        layers.UpSampling1D(size=3),
        layers.Conv1DTranspose(16, 5, activation=act_conv),
        layers.BatchNormalization(),
        layers.UpSampling1D(size=3),
        layers.Conv1DTranspose(8, 5, activation=act_conv),
        layers.BatchNormalization(),
        layers.UpSampling1D(size=3),
        layers.Conv1DTranspose(80, 5, activation=act_conv)
    ])
    #<-
  def call(self, x, step):
    """
    Génére une sortie de taille size à partir d'un espace latent (batch, latent_size)
    """
    #<-pi[Decodeur_rnn-call]
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
    #<-
    return x

class Auto_Encodeur_rnn(tf.keras.Model):
  def __init__(self):
    super(Auto_Encodeur_rnn, self).__init__()
    self.encodeur = Encodeur_rnn()
    self.decodeur = Decodeur_rnn()

  def call(self,x):
    """
    Prend un vecteur de dimension (b, lenght, 80)
    """
    latent, step = self.encodeur(x)
    out  = self.decodeur(latent,step)
    return out

  def save_weights(self, file, step):
    self.encodeur.save_weights(file+'/Encodeur_checkpoint/'+str(step))
    self.decodeur.save_weights(file+'/Decodeur_checkpoint/'+str(step))


  def load_weights(self, file,  step = None):
    if step is None:
      f_enco = tf.train.latest_checkpoint(file+'/Encodeur_checkpoint')
      f_deco = tf.train.latest_checkpoint(file+'/Decodeur_checkpoint')
    else:
      f_enco = file+'/Encodeur_checkpoint/'+str(step)
      f_deco = file+'/Decodeur_checkpoint/'+str(step)

    self.encodeur.load_weights(f_enco)
    self.decodeur.load_weights(f_deco)


"""
Auto-enco-rnn-conv2D
"""
class Encodeur_rnn2D(tf.keras.Model):
  def __init__(self):
    super(Encodeur_rnn2D, self).__init__()
    act_rnn  = act.elu
    act_conv = act.elu

    self.conv = tf.keras.models.Sequential([
        layers.Masking(mask_value=0.),

        layers.Conv2D(16, 5, activation=act_conv),
        layers.MaxPool2D(pool_size=2),
        layers.BatchNormalization(),

        layers.Conv2D(32, 5, activation=act_conv),
        layers.MaxPool2D(pool_size=2),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, 5, activation=act_conv),
        layers.MaxPool2D(pool_size=2),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, 5,activation=act_conv),
    ])
    self.lstm_1  = layers.GRU(64, activation = act_rnn, return_sequences = True)
    self.bi_lstm = layers.Bidirectional(layers.LSTM(128, activation = act_rnn))  
    self.latent  = layers.Dense(64)

  def call(self, x):
    x = tf.expand_dims(x, dims=-1)
    x = self.conv(x)
    x = tf.squeeze(x)
    epoch_lstm = x.shape[1]
    x = self.lstm_1(x)
    x = self.bi_lstm(x)
    return x, epoch_lstm

class Decodeur_rnn2D(tf.keras.Model):
  def __init__(self):
    super(Decodeur_rnn2D, self).__init__()
    act_rnn  = act.elu
    act_conv = act.elu
    self.lstm_1  = layers.GRU(64, activation = act_rnn)
    self.bi_lstm = layers.Bidirectional(layers.LSTM(128, activation = act_rnn, return_sequences=True))
    self.convT = tf.keras.models.Sequential([
      layers.Conv2DTranspose(64, 5, activation=act_conv),
      layers.BatchNormalization(),
      
      layers.UpSampling2D(size=2),
      layers.Conv2DTranspose(32, 5, activation=act_conv),
      layers.BatchNormalization(),

      layers.UpSampling2D(size=2),
      layers.Conv2DTranspose(16, 5, activation=act_conv),
      layers.BatchNormalization(),

      layers.UpSampling2D(size=2),
      layers.Conv2DTranspose(1, (5), activation=act_conv)
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
    x = tf.expand_dims(x, dims = 2)
    x = self.convT(x)
    x = tf.squeeze(x)
    x = ks.activations.sigmoid(x)*(MAX-MIN)+MIN
    return x
"""
Encodeur conv2D
"""
class Encodeur_conv2D(tf.keras.Model):
  def __init__(self):
    super(Encodeur_conv2D, self).__init__()
    act_conv = act.relu
    self.conv = tf.keras.models.Sequential([
        layers.Masking(mask_value=0.),
        
        layers.Conv2D(16, (5,4), activation=act_conv),
        layers.MaxPool2D(pool_size=(4,2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(32, (5,4), activation=act_conv),
        layers.MaxPool2D(pool_size=(4,2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (5,6), activation=act_conv),
        layers.MaxPool2D(pool_size=(3,2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, (5,5), activation=act_conv),
        layers.MaxPool2D(pool_size=(4,2)),
    ])
    self.flatten = layers.Flatten()
    self.latent  = layers.Dense(128+256+1)

  def call(self, x):
    x = self.conv(x)
    x = self.flatten(x)
    x = self.latent(x)
    return x

class Decodeur_conv2D(tf.keras.Model):
  def __init__(self):
    super(Decodeur_conv2D, self).__init__()
    act_conv = act.elu
    self.convT = tf.keras.models.Sequential([
    layers.UpSampling2D(size=(4,2)),
    layers.Conv2DTranspose(64, (5,5), activation=act_conv),
    layers.BatchNormalization(),
    
    layers.UpSampling2D(size=(4,2)),
    layers.Conv2DTranspose(32, (7,6), activation=act_conv),
    layers.BatchNormalization(),
    
    layers.UpSampling2D(size=(4,2)),
    layers.Conv2DTranspose(16, (8,5), activation=act_conv),
    layers.BatchNormalization(),

    layers.UpSampling2D(size=(4,2)),
    layers.Conv2DTranspose(1, (12,5), activation=act_conv),
    ])

  def call(self, x):
    """
    Génére une sortie de taille size à partir d'un espace latent (batch, latent_size)
    """
    x = self.convT(x)
    x = tf.squeeze(x)
    x = ks.activations.sigmoid(x)*(MAX-MIN)+MIN
    return x

class Discriminateur_conv2D(tf.keras.Model):
  def __init__(self):
    super(Discriminateur_conv2D, self).__init__()
    act_conv = act.relu
    self.conv = tf.keras.models.Sequential([
        layers.Masking(mask_value=0.),
        
        layers.Conv2D(16, (5,4), activation=act_conv),
        layers.MaxPool2D(pool_size=(4,2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(32, (5,4), activation=act_conv),
        layers.MaxPool2D(pool_size=(4,2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (5,6), activation=act_conv),
        layers.MaxPool2D(pool_size=(3,2)),
        layers.BatchNormalization(),

        layers.Conv2D(128, (11, 6), activation=act_conv)
    ])
    self.flatten = layers.Flatten()
    self.dense  = tf.keras.models.Sequential([
        layers.Dense(64),
        layers.Dense(32),
        layers.Dense(1),])

  def call(self, x):
    x = tf.expand_dims(x, axis=-1)
    x = self.conv(x)
    x = self.flatten(x)
    x = self.dense(x)
    x = ks.activations.sigmoid(x)
    return x


class Auto_Encodeur_conv2D(Auto_Encodeur_rnn):
  def __init__(self):
    super(Auto_Encodeur_conv2D, self).__init__()
    self.encodeur = Encodeur_conv2D()
    self.decodeur = Decodeur_conv2D()

  def call(self,x):
    """
    Prend un vecteur de dimension (b, lenght, 80)
    """
    x = tf.expand_dims(x, axis=-1)
    latent = self.encodeur(x)
    latent = tf.reshape(latent,(latent.shape[0], 1,1, -1))
    out    = self.decodeur(latent)
    return out



############################################################
# ##########################################################
# ##########################################################
############################################################
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
        layers.MaxPool1D(pool_size=2),
        layers.BatchNormalization(),

        layers.Conv1D(64, 4, activation=act_conv),
        layers.MaxPool1D(pool_size=3),
    ])
    self.latent = layers.Dense(128)
  def call(self, x):
    x = tf.expand_dims(x, axis = -1)
    x = self.conv(x)
    x = self.latent(x)
    return x

class Decodeur_SAU(tf.keras.Model):
  def __init__(self):
    super(Decodeur_SAU, self).__init__()
    act_conv = act.elu
    self.convT = tf.keras.models.Sequential([
        layers.UpSampling1D(size=4),
        layers.Conv1DTranspose(32, 4, activation=act_conv),
        layers.BatchNormalization(),
        layers.Dropout(.2),

        layers.UpSampling1D(size=2),
        layers.Conv1DTranspose(16, 4, activation=act_conv),
        layers.BatchNormalization(),
        layers.Dropout(.2),
        
        layers.UpSampling1D(size=2),
        layers.Conv1DTranspose(8, 4, activation=act_conv),
        layers.BatchNormalization(),
        layers.Dropout(.2),

        layers.UpSampling1D(size=2),
        layers.Conv1DTranspose(1, 7, activation=act_conv),
    ])
  def call(self, x):
    """
    Génére une sortie de taille size à partir d'un espace latent (batch, latent_size)
    """
    x = self.convT(x)
    x = tf.squeeze(x)
    x = ks.activations.sigmoid(x)*(MAX_normalised-MIN_normalised)+MIN_normalised
    return x

class Auto_Encodeur_SAU(Auto_Encodeur_rnn):
  def __init__(self):
    super(Auto_Encodeur_SAU, self).__init__()
    self.encodeur = Encodeur_SAU()
    self.decodeur = Decodeur_SAU()


  def call(self,x, phi):
    """
    x   : (b*lenght, 80)
    phi : (b*lenght, 128)
    """
    phi = np.expand_dims(phi, axis=1) #(b*lenght, 1, 128)
    latent = self.encodeur(x)#(b*lenght, 1, 80)
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
    act_rnn  = act.elu
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
    self.lstm_1  = layers.GRU(128, 
                              activation = act_rnn, 
                              return_sequences = True,
                              dropout=0.2)
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
    #x = self.lstm_1(x)
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










if __name__ == "__main__":
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    import os
    MEAN_DATASET = -6.0056405
    STD_DATASET  = 2.4420118
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


    from utilitaires.dataloader_ESD_tf import ESD_data_generator
    Encodeur_rnn = Encodeur_rnn()
    decodeur = Decodeur_rnn()
    FILEPATH = r"/home/luc/Documents/STAGE_IRCAM/data/ESD_Mel/"
    base = ESD_data_generator(FILEPATH, batch_size=10, shuffle=True, langage="english")
    
    def mse(y_hat, y):
      sub = tf.math.subtract(tf.cast(y, tf.float64),tf.cast(y_hat, tf.float64))
      r   = tf.math.reduce_sum(tf.math.pow(sub,2))
      r  /= tf.math.reduce_sum(tf.cast(y != 0, tf.float64))
      return r

    Autoenco = Auto_Encodeur_rnn()
    Autoenco.load_weights(os.getcwd()+"/Seen_and_Unseen/backup/logs/20220329-114214")
    #Autoenco.build(tf.TensorShape([10, None, 80]))
    
    for x, y in base:
      x = (x - MEAN_DATASET)/STD_DATASET 
      #batch, lenght, n
      out = Autoenco(x)
      x    = x[:,:out.shape[1]] #Crop pour les pertes de reconstruction du decodeur
      mask = tf.cast(x, tf.bool)
      out  = tf.multiply(out, tf.cast(mask, tf.float32))
      loss = mse(x, out)
      print(loss)
import tensorflow as tf
import tensorflow.keras.activations  as act
import tensorflow.keras as ks
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
import numpy as np

# Valeurs observés dans les data
MAX = 3
MIN = -14
"""
Decodeur et Encodeur Convolutif/RNN
"""
class Encodeur(tf.keras.Model):
  def __init__(self):
    super(Encodeur, self).__init__()
    
    #->pi[activation]
    act_rnn  = act.elu
    act_conv = act.elu
    #<-

    #->pi[conv]
    self.conv = tf.keras.models.Sequential([
        layers.Masking(mask_value=0.),
        layers.Conv1D(8, 5, activation=act_conv),
        layers.MaxPool1D(pool_size=3),
        layers.BatchNormalization(),
        layers.Conv1D(16, 5, activation=act_conv),
        layers.MaxPool1D(pool_size=3),
        layers.BatchNormalization(),
        layers.Conv1D(32, 5, activation=act_conv),
        layers.MaxPool1D(pool_size=3),
    ])
    #<-
    #->pi[lstm-enco]
    self.lstm_1  = layers.GRU(64, activation = act_rnn, return_sequences = True)
    self.bi_lstm = layers.Bidirectional(layers.LSTM(128, activation = act_rnn))
    #<-  
    self.latent  = layers.Dense(64)

  def call(self, x):
    #->pi[encodeur-call]
    x = self.conv(x)
    epoch_lstm = x.shape[1]
    x = self.lstm_1(x)
    x = self.bi_lstm(x)
    #x = self.latent(x)
    #<-
    return x, epoch_lstm

class Decodeur(tf.keras.Model):
  def __init__(self):
    super(Decodeur, self).__init__()
    #->pi[activation]
    act_rnn  = act.elu
    act_conv = act.elu
    #<-   

    #->pi[lstm-decodeur]
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
    #<-pi[decodeur-call]
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

class Auto_encodeur_rnn(tf.keras.Model):
  def __init__(self):
    super(Auto_encodeur_rnn, self).__init__()
    self.encodeur = Encodeur()
    self.decodeur = Decodeur()

  def call(self,x):
    """
    Prend un vecteur de dimension (b, lenght, 80)
    """
    latent, step = self.encodeur(x)
    out  = self.decodeur(latent,step)
    return out

  def save_weights(self, file, step):
    self.encodeur.save_weights(file+'/encodeur_checkpoint/'+str(step))
    self.decodeur.save_weights(file+'/decodeur_checkpoint/'+str(step))


  def load_weights(self, file,  step = None):
    if step is None:
      f_enco = tf.train.latest_checkpoint(file+'/encodeur_checkpoint')
      f_deco = tf.train.latest_checkpoint(file+'/decodeur_checkpoint')
    else:
      f_enco = file+'/encodeur_checkpoint/'+str(step)
      f_deco = file+'/dncodeur_checkpoint/'+str(step)

    self.encodeur.load_weights(f_enco)
    self.decodeur.load_weights(f_deco)

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
    encodeur = Encodeur()
    decodeur = Decodeur()
    FILEPATH = r"/home/luc/Documents/STAGE_IRCAM/data/ESD_Mel/"
    base = ESD_data_generator(FILEPATH, batch_size=10, shuffle=True, langage="english")
    
    f = './test/save'
    Autoenco = Auto_encodeur_rnn()
    Autoenco.load_weights(os.getcwd()+"/Seen_and_Unseen/backup/logs/20220329-114214")
    #Autoenco.build(tf.TensorShape([10, None, 80]))
    for x, y in base:
      tmp = (x - MEAN_DATASET)/STD_DATASET 
      tmp = tf.transpose(tmp, perm = [0,2,1])#batch, lenght, n
      out = Autoenco(tmp)
      break
    Autoenco.save_weights(f,0)
    Autoenco.load_weights(f)

    for x,y in base:
        tmp = x
        tmp = tf.transpose(tmp, perm = [0,2,1])#batch, lenght, n
        out, step = encodeur(tmp)
        rec = decodeur(out, step)
        print(rec.shape, tmp.shape)
    print("ok")    
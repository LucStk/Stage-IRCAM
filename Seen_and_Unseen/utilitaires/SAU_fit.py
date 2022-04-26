from csv import list_dialects
import tensorflow as tf
import tensorflow.keras.activations  as act
import tensorflow.keras as ks
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from utilitaires.utils import *
from utilitaires.All_model import *
import numpy as np
MAX = 3
MIN = -14
MEAN_DATASET  = -6.0056405
STD_DATASET   =  2.4420118
MAX_normalised = (MAX-MEAN_DATASET)/STD_DATASET
MIN_normalised = (MIN-MEAN_DATASET)/STD_DATASET

class SAU_GAN(tf.keras.Model):
  def __init__(self):
    super(SAU_GAN, self).__init__()
    self.auto_encodeur  = Auto_Encodeur_SAU()
    self.discriminateur = Discriminateur_SAU()

  def compile(self, ae_optim, disc_optim, steps_per_execution=1):
    super(SAU_GAN, self).compile(steps_per_execution=steps_per_execution)
    self.ae_optim   = ae_optim
    self.disc_optim = disc_optim
    self.BCE        = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

  def call(self, x):
    pass

  def train_step(self, input):
    x = input[:,:80]
    z = input[:,80:]
    x = normalisation(x)
    l_gen =0; l_disc = 0; mcd = 0

    with tf.GradientTape() as tape_gen,tf.GradientTape() as tape_disc:
        out_ae  = self.auto_encodeur(x, z)
        d_false = self.discriminateur(out_ae)
        l_gen   = tf.math.reduce_mean(self.BCE(tf.ones_like(d_false), d_false))

        d_true = self.discriminateur(x)
        l_disc = tf.math.reduce_mean(self.BCE(tf.ones_like(d_true), d_true) +\
                                     self.BCE(tf.zeros_like(d_false), d_false))
    

    tr_var_ae = self.auto_encodeur.trainable_variables
    grad_gen  = tape_gen.gradient(l_gen  , tr_var_ae)
    self.ae_optim.apply_gradients(zip(grad_gen, tr_var_ae))

    tr_var_disc = self.discriminateur.trainable_variables
    grad_disc   = tape_disc.gradient(l_disc, tr_var_disc)
    self.disc_optim.apply_gradients(zip(grad_disc, tr_var_disc))
    
    mcd = MCD_1D(out_ae, x)
    
    return {"loss_generateur": l_gen,"loss_discriminateur": l_disc, "MCD":mcd}

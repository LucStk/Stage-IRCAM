#%% importation 
import pickle
import os
import pandas as pd
import tensorflow as tf
import numpy as np
"""
data_name example : 'F03_a2_s036_v04_melspc.p'

a1 : amical
a2 : distant
a3 : dominant
a4 : séducteur
"""
#%% Load la data_base de mel spéctrogram en format pandas
FILEPATH = r"/home/luc/Documents/STAGE_IRCAM/data/melspcs/"
entries =  os.listdir(FILEPATH)
def dec_name_file(f):
    """
    Décompose le nom du fichier pour en extraire les informations pertinentes
    """
    return {"sex": f[0], "num":int(f[1:3]), "emotion":int(f[5]), "séquence": int(f[8:11]), "version": int(f[13:15])}
data_base = pd.DataFrame([{**pd.read_pickle(FILEPATH+e), **dec_name_file(e)} for e in entries])
print(data_base.head())

#%% Création du dataloader tensorflow
mell = data_base["mell"].to_numpy()
#remplissage
max_lengths = np.max([x.shape[1] for x in mell])
def remplissage(x, max, pad = 0):
    """
    Prend une matrice en entrée en rajoute (max-x.shape[1]) padd à la fin
    """
    return np.concatenate((x, np.ones((x.shape[0], max-x.shape[1]))*pad), axis = 1)
mell_remplit = np.array([remplissage(x, max_lengths) for x in mell])

# 

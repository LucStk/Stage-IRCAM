import pickle
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Sequence
from pathlib import Path
import re
import math

"""
Contient la class AttHACK_Mell_Spect, 

data_name example : 'F03_a2_s036_v04_melspc.p'

a1 : amical
a2 : distant
a3 : dominant
a4 : séducteur
"""

def remplissage(x, max, pad = 0):
    """
    Prend une matrice en entrée en rajoute (max-x.shape[1]) padd à la fin
    """
    return np.concatenate((x, np.ones((x.shape[0], max-x.shape[1]))*pad), axis = 1)

def collate_fn(x):
    """
    Prend un liste de matrices en entrée construit une matrice tridimensionnelle carré
    en ajoutant un pad de 0
    """
    max_val = max([i.shape[1] for i in x])
    r = np.array([remplissage(i, max_val, pad = 0) for i in x])
    return r

class ESD_data_generator(Sequence):
    def __init__(self, file_path, batch_size=1, shuffle=True, 
                 langage=None, type_ = 'train', transform=None,):
        """
        file_path (string) : chemin du fichier ESD_mell 
        transform (callable, optional) : Optional transformation à appliquer
        language : chinese, english ou None, si None prend les deux langues
        type_ : train, test, validation, default train
        """
        if type_ not in ['train', 'test', 'validation']:
            raise 'InputError {} is not a valide type'.format(type_)

        def filtre_lg(l, lg): # Selection english or chinese audio or both
            dict_lg = {"english": "([0]{2}(?:[1][1-9]|[2][0]))", 
                       "chinese" :"([0]{2}(?:[0]\d|[1][0]))", 
                       None : ".*"}
            ret = []
            for i in l:
                i = i.__str__() #Only work if the directory is named ESD
                if re.match('(?:.*\/ESD_Mel\/'+dict_lg[lg]+')',i):
                    ret += [i]
            return ret

        p = Path(file_path)
        self.dataset = filtre_lg(p.glob('**/{}/*.p'.format(type_)), langage)        
        self.shuffle    = shuffle
        self.file_path  = file_path
        self.transform  = transform
        self.batch_size = batch_size

        if shuffle:
            np.random.shuffle(self.dataset)

    def __len__(self):
        return math.ceil(len(self.dataset)/self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.dataset)


    def __getitem__(self, idx):
        """
        labels : 0 : Angry, 1 : Happy, 2: Neutral, 3: Sad, 4: Surprise
        """
        list_emotions = ['Angry', "Happy", "Neutral", "Sad", "Surprise"]
        data_name = self.dataset[idx*self.batch_size:(idx+1)*self.batch_size]
        y = [list_emotions.index(re.findall("((?:\w|\.)+)", l)[-3]) for l in data_name]
        x = collate_fn([pd.read_pickle(f) for f in data_name])

        return x, y

if __name__ == "__main__":
    FILEPATH = r"/home/luc/Documents/STAGE_IRCAM/data/ESD_Mel/"
    base = ESD_data_generator(FILEPATH, batch_size=1000, shuffle=True, langage="english")
    base_test = ESD_data_generator(FILEPATH, batch_size=10, shuffle=True,type_="test", langage="english")
    print(len(base))
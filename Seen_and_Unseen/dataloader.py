import pickle
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import torch
"""
Contient la class AttHACK_Mell_Spect, 

data_name example : 'F03_a2_s036_v04_melspc.p'

a1 : amical
a2 : distant
a3 : dominant
a4 : séducteur
"""

class AttHACK_Mell_spectrogram(torch.utils.data.Dataset):
    def __init__(self, file_path, transform = None):
        """
        file_path (string) : chemin du fichier avec les mel spectrogram
        transform (callable, optional) : Optional transformation à appliquer
        """
        self.entries   =  os.listdir(file_path)
        self.file_path = file_path
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        def dec_name_file(f):
            """
            Décompose le nom du fichier mel pour en extraire les informations pertinentes
            """
            return {"sex": f[0], "num":int(f[1:3]), "emotion":int(f[5]), "séquence": int(f[8:11]), "version": int(f[13:15])}

        e = self.entries[idx]
        sample = {"data": pd.read_pickle(self.file_path+e)["mell"], **dec_name_file(e)}
        if self.transform:
            sample = self.transform(sample)
        return sample

def remplissage(x, max, pad = 0):
    """
    Prend une matrice en entrée en rajoute (max-x.shape[1]) padd à la fin
    """
    return np.concatenate((x, np.ones((x.shape[0], max-x.shape[1]))*pad), axis = 1)

def collate_fn(x):
    """
    Prend un liste de matrices en entrée en rajoute (max-x.shape[1]) padd à la fin de toutes
    """
    tmp = [list(i.values()) for i in x]
    tmp = np.expand_dims(np.array(tmp, dtype=object), axis = 2)
    ret = np.concatenate(tmp, axis=1)
    max_val = max([i.shape[1] for i in ret[0]])
    r = np.array([remplissage(i, max_val, pad = 0) for i in ret[0]])
    ret = [r] + list(ret[1:])
    return dict(zip(x[0].keys(),ret))

if __name__ == "__main__":
    FILEPATH = r"/home/luc/Documents/STAGE_IRCAM/data/melspcs/"
    base = AttHACK_Mell_spectrogram(FILEPATH)
    print(len(base))
    print(next(iter(base)))
    data_loader = torch.utils.data.DataLoader(base, batch_size = 10, shuffle = True, collate_fn = collate_fn)
    i = next(iter(data_loader))
    print("ok")
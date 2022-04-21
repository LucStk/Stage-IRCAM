import enum
import tensorflow as tf
import pandas as pd
from utilitaires.dataloader_ESD_tf import *
MEAN_DATASET = -6.0056405
STD_DATASET  = 2.4420118
MAX = 3
MIN = -14


def normalisation(x):
    mask = tf.cast(x!=0, tf.float64)
    x = (x - MEAN_DATASET)/STD_DATASET #Normalisation
    x = tf.multiply(tf.cast(x, tf.float64), mask)
    #x = tf.cast(x, tf.float32)
    return x

def de_normalisation(x):
    x = tf.cast(x, tf.float64)
    mask = tf.cast(x!=0, tf.float64)
    x = (x*STD_DATASET + MEAN_DATASET) #dé-Normalisation
    x = tf.multiply(x, mask)
    return x

def dataloader(FILEPATH, batch_size=30, batch_size_test = 50, shuffle=True, 
               langage = 'english', use_data_queue= False):
    data_queue = None
    train_dataloader = ESD_data_generator(FILEPATH, batch_size, shuffle, langage)
    len_train = len(train_dataloader)
    if use_data_queue:
        print("begin data_queue")
        data_queue = tf.keras.utils.OrderedEnqueuer(train_dataloader, use_multiprocessing=False, shuffle=True)
        data_queue.start()
        train_dataloader = data_queue.get()    
    test_dataloader = ESD_data_generator(FILEPATH, batch_size_test, langage=langage, type_='test',shuffle=True)
    return train_dataloader, test_dataloader, data_queue, len_train

def dataloader_SAU(FILEPATH, batch_size=30, batch_size_test = 50, shuffle=True, 
               langage = 'english', use_data_queue= False):

    data_queue = None
    train_dataloader = ESD_data_generator(FILEPATH, batch_size, shuffle, langage)
    len_train = len(train_dataloader)
    if use_data_queue:
        print("begin data_queue")
        data_queue = tf.keras.utils.OrderedEnqueuer(train_dataloader, use_multiprocessing=False, shuffle=True)
        data_queue.start()
        train_dataloader = data_queue.get()
            
    test_dataloader = ESD_data_generator(FILEPATH, batch_size_test, langage=langage, type_='test',shuffle=True)
    return train_dataloader, test_dataloader, data_queue, len_train


def mean_SER_emotion(FILEPATH, SER, batch_size):
    """
    Effectue, pour un chaque émotion, une moyenne d'espace latent SER avec un
    sample de taille batch_size
    return : (5, 128)
    """
    dataloader = ESD_batch_data_generator(FILEPATH, shuffle = False, langage='english')
    gr_emotion = dataloader.dataset_emotion # (5,~) nom_fichier listé par émotion
    ret = []
    for g in gr_emotion:
        tf.random.shuffle(g)
        batch = g[:batch_size]#list de batch_size nom de fichier d'émotion i
        x = [pd.read_pickle(f) for f in batch]
        x = auto_padding(x)
        x = tf.transpose(x, perm = [0,2,1])
        x = normalisation(x)
        tmp = SER.call_latent(x)
        latent = tf.math.reduce_mean(tmp, axis = 0)
        ret += [latent]
    return ret

def emotion_echantillon(FILEPATH):
    """
    Fournis un échantillion pour chaque émotions 
    return : (L, 80)
    """
    dataloader = ESD_batch_data_generator(FILEPATH, shuffle = False, langage='english')
    gr_emotion = dataloader.dataset_emotion # (5,~) nom_fichier listé par émotion
    ret = []
    for g in gr_emotion:
        tf.random.shuffle(g)
        x = pd.read_pickle(g[0])
        x = tf.transpose(x, perm = [1,0])
        x = normalisation(x)
        ret += [x]

    return ret


class Mel_inverter():
    def __init__(self):
        try :
            from svp_cmds.calc_melspec import calc_melspec
            from MBExWN_NVoc import mel_inverter, list_models, mbexwn_version
            from fileio import iovar
        except:
            print("Not at IRCAM, Mel_inverter not working")

        self.MelInv = mel_inverter.MELInverter("SPEECH")
        self.dd = {'nfft': 2048,
        'hoplen': 200,
        'winlen': 800,
        'nmels': 80,
        'sr': 16000,
        'fmin': 0.0,
        'fmax': 8000.0,
        'lin_spec_offset': None,
        'lin_spec_scale': 1,
        'log_spec_offset': 0,
        'log_spec_scale': 1,
        'time_axis': 1,
        'mell':None}

    def convert(self, x):
        """
        Convertion mell spectro to audio
        """
        self.dd['mell'] = x.numpy().T
        log_mel_spectrogram = self.MelInv.scale_mel(self.dd, verbose=False)
        rec_audio = self.MelInv.synth_from_mel(log_mel_spectrogram)
        return tf.experimental.numpy.atleast_3d(rec_audio)

def MDC(x_hat, x):
    x    = x[:,:x_hat.shape[1]]#Crop pour les pertes de reconstruction du decodeur
    mask = tf.cast(x!=0, tf.float64)
    n   = tf.math.reduce_sum(mask[:,:,0], axis=1)#nombre de values a comparer
    sub = tf.math.subtract(tf.cast(x, tf.float64),tf.cast(x_hat, tf.float64))
    r   = tf.multiply(sub, mask)
    a = tf.math.pow(r,2)
    a = tf.math.sqrt(tf.math.reduce_sum(a, axis=2))
    a = tf.math.reduce_sum(a, axis = 1)/n
    tmp = (10/tf.math.log(10))*tf.math.sqrt(2)*STD_DATASET
    a = tf.math.multiply(tf.cast(a, dtype=tf.float64), tf.cast(tmp,dtype=tf.float64))
    return tf.math.reduce_mean(a)

def MDC_1D(x_hat, x):
    sub = tf.math.subtract(tf.cast(x, tf.float64),tf.cast(x_hat, tf.float64))
    a = tf.math.pow(sub,2)
    a = tf.math.sqrt(tf.math.reduce_sum(a, axis=1))
    a = tf.math.reduce_mean(a)
    tmp = (10/tf.math.log(10.))*tf.math.sqrt(2.)*STD_DATASET
    a = tf.math.multiply(tf.cast(a, dtype=tf.float64), tf.cast(tmp,dtype=tf.float64))
    return a
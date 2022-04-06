import os
from pickletools import optimize
from sklearn.utils import shuffle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from utilitaires.dataloader_ESD_tf import ESD_data_generator
from model import Encodeur, Decodeur, Auto_encodeur_rnn
import datetime
import sys
import numpy as np
import getopt

#valeurs obersvé empiriquement, utilisé pour la normalisation
MEAN_DATASET = -6.0056405
STD_DATASET  = 2.4420118


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def normalisation(x):
    mask = tf.cast(x!=0, tf.float64)
    x = (x - MEAN_DATASET)/STD_DATASET #Normalisation
    x = tf.multiply(x, mask)
    #x = tf.cast(x, tf.float32)
    return x

def de_normalisation(x):
    x = tf.cast(x, tf.float64)
    mask = tf.cast(x!=0, tf.float64)
    x = (x*STD_DATASET + MEAN_DATASET) #dé-Normalisation
    x = tf.multiply(x, mask)
    return x

def dataloader(FILEPATH, batch_size=30, shuffle=True, langage = 'english', use_data_queue= False):
    data_queue = None
    train_dataloader = ESD_data_generator(FILEPATH, batch_size, shuffle, langage)
    len_train = len(train_dataloader)
    if use_data_queue:
        print("begin data_queue")
        data_queue = tf.keras.utils.OrderedEnqueuer(train_dataloader, use_multiprocessing=False, shuffle=True)
        data_queue.start()
        train_dataloader = data_queue.get()    
    test_dataloader = ESD_data_generator(FILEPATH, batch_size=400, langage=langage, type_='test',shuffle=True)
    return train_dataloader, test_dataloader, data_queue, len_train

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
        print(x.shape)
        self.dd['mell'] = x.numpy().T
        log_mel_spectrogram = self.MelInv.scale_mel(self.dd, verbose=False)
        rec_audio = self.MelInv.synth_from_mel(log_mel_spectrogram)
        return np.atleast_3d(rec_audio)


def train(train_dataloader, test_dataloader, len_train, 
          test = False, 
          ircam = False, 
          load_path = None):
    print("Training Beging")
    EPOCH = 10
    LR = 1e-5
    TEST_EPOCH = 1/2

    log_dir        = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)
    if ircam:
        audio_log_dir        = "audio_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        audio_summary_writer = tf.summary.create_file_writer(audio_log_dir)
        mel_inv = Mel_inverter()

    def mse(x_hat, x):
        x    = x[:,:x_hat.shape[1]]#Crop pour les pertes de reconstruction du decodeur
        mask = tf.cast(x!=0, tf.float64)
        sub = tf.math.subtract(tf.cast(x, tf.float64),tf.cast(x_hat, tf.float64))
        r   = tf.multiply(sub, mask)
        r   = tf.math.reduce_sum(tf.math.pow(r,2)) /tf.math.reduce_sum(mask)
        return r

    def MDC(x_hat, x):
        x    = x[:,:x_hat.shape[1]]#Crop pour les pertes de reconstruction du decodeur
        mask = tf.cast(x!=0, tf.float64)
        n   = tf.math.reduce_sum(mask[:,:,0], axis=1)#nombre de values a comparer
        sub = tf.math.subtract(tf.cast(x, tf.float64),tf.cast(x_hat, tf.float64))
        r   = tf.multiply(sub, mask)
        a = tf.math.pow(r,2)
        a = tf.math.sqrt(tf.math.reduce_sum(a, axis=2))
        a = tf.math.reduce_sum(a, axis = 1)/n
        a *= (10/np.log(10))*np.sqrt(2)*STD_DATASET 
        return tf.math.reduce_mean(a)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate = LR)
    Model = Auto_encodeur_rnn()
    if load_path is not None :
        try:
            Model.load_weights(os.getcwd()+load_path)
            print('model load sucessfuly')
        except:
            print("Load not succesful from"+os.getcwd()+load_path)

    print("Every thing is ready")
    for cpt, data  in enumerate(train_dataloader):
        print(cpt)
        x, y = data
        x = normalisation(x)
        with tf.GradientTape() as tape: #Normalisation
            out  = Model(x)
            loss = mse(out, x)
        mdc  = MDC(out, x)

        #tmp = loss(out_, x_)
        with summary_writer.as_default(): 
            tf.summary.scalar('train/loss',loss , step=cpt)
            tf.summary.scalar('train/mdc',mdc , step=cpt)

        gradients = tape.gradient(loss, Model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, Model.trainable_variables))            
        """
        TEST
        """
        if test and ((cpt+1)%int(TEST_EPOCH*len_train) == 0):
            print('test_time')
            for c, data  in enumerate(test_dataloader):
                x, y = data
                x    = normalisation(x) 
                out  = Model(x)
                loss = mse(out, x)
                mcd  = MDC(out, x)
                
                with summary_writer.as_default(): 
                    tf.summary.scalar('test/loss',loss, step=cpt)
                    tf.summary.scalar('test/mcd',mcd , step=cpt)
                if c == 2:
                    break
            test_dataloader.shuffle()

        if (cpt+1) % len_train == 0:
            print("End batch")

            if ircam:
                #c = np.random.choice(range(len(test_dataloader)))
                #i = np.random.choice(range(len(x_)))
                
                c = 0;i = 0
                x, y = test_dataloader[c]
                x    = normalisation(tf.expand_dims(x[i],0))
                out  = Model(x)

                x_   = de_normalisation(x)
                out_ = de_normalisation(out)
                mask_lenght = int(tf.math.reduce_sum(tf.cast(x[0,:,0]!=0, tf.float64)))
                rec_x   = mel_inv.convert(x_[:,:mask_lenght][0])
                rec_out = mel_inv.convert(out_[:,:mask_lenght][0])

                with audio_summary_writer.as_default(): 
                    tf.summary.audio('Original',rec_x, 24000, step=cpt)
                    tf.summary.audio('Reconstruct',rec_out, 24000,step=cpt)

            print("save")
            Model.save_weights(log_dir, format(cpt//len_train))

if __name__ == "__main__":
    longoptions = ['lock=', 'place=', 'load=']
    ov, ra = getopt.getopt(sys.argv[1:], "", longoptions)
    ov = dict(ov)
    place = ov.get("--place")

    if place == 'ircam':
        print("ircam connexion")

        import manage_gpus as gpl
        lck = ov.get("--lock")
        if lck is None:
            soft = None
            print("WARNING : No lock taken")
        elif lck.lower() == "soft": soft = True
        elif lck.lower() == "hard": soft = False 
        else:
            print("WARNING : No lock taken")
            soft = None
        
        if soft is not None:
            try:
                gpu_id_locked = gpl.get_gpu_lock(gpu_device_id=-1, soft=soft)
                comp_device = "/GPU:0"
                print("Gpu taken")
            except gpl.NoGpuManager:
                print("no gpu manager available - will use all available GPUs", file=sys.stderr)
            except gpl.NoGpuAvailable:
                # there is no GPU available for locking, continue with CPU
                comp_device = "/cpu:0" 
                os.environ["CUDA_VISIBLE_DEVICES"]=""

        try:
            FILEPATH = r"/data2/anasynth_nonbp/sterkers/ESD_Mel/"
            os.listdir(FILEPATH)
        except:
            try:
                FILEPATH = r"/data/anasynth_nonbp/sterkers/ESD_Mel/"
                os.listdir(FILEPATH)
            except:
                raise Exception('Data not found')
        
        BATCH_SIZE = 256
        SHUFFLE    = True
        LANGAGE    = "english"
        USE_DATA_QUEUE = True
        load_path = ov.get('--load')

        train_dataloader, test_dataloader, data_queue, len_train = dataloader(FILEPATH, BATCH_SIZE, SHUFFLE, 
                                                                    LANGAGE, USE_DATA_QUEUE,)
        print("test_dataloader")
        for i in train_dataloader:
            print("ok")
            break
        print("end_test_dataloader")
        with tf.device(comp_device) :
            train(train_dataloader, test_dataloader, len_train, test = True, 
                        ircam=True, load_path=load_path)
        if data_queue:
            data_queue.stop()

    else:
        FILEPATH = r"/home/luc/Documents/STAGE_IRCAM/data/ESD_Mel/"
        BATCH_SIZE = 30
        SHUFFLE    = True
        LANGAGE    = "english"
        USE_DATA_QUEUE = False
        load_path = ov.get('--load')
        train_dataloader, test_dataloader, data_queue, len_train = dataloader(FILEPATH, BATCH_SIZE, SHUFFLE, 
                                                                    LANGAGE, USE_DATA_QUEUE)
        train(train_dataloader, test_dataloader, len_train, test = True, load_path=load_path)
        if data_queue:
            data_queue.stop()
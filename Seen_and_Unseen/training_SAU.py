import os
from pickletools import optimize
from pexpect import ExceptionPexpect
from sklearn.utils import shuffle
from sympy import discriminant
from yaml import load
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from utilitaires.model import SER, Auto_Encodeur_SAU, Discriminator_SAU 
from utilitaires.utils import *
import datetime
import sys
import numpy as np
import getopt

#valeurs obersvé empiriquement, utilisé pour la normalisation
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def train(FILE_PATH, train_dataloader, test_dataloader, len_train, 
          test  = False, 
          ircam = False, 
          load_path = None,
          load_SER_path = None):

    print("Training Beging")
    EPOCH = 100
    LR = 1e-5
    TEST_EPOCH = 1/2
    BATCH_SIZE = 256

    optimizer = tf.keras.optimizers.RMSprop(learning_rate = LR)
    auto_encodeur = Auto_Encodeur_SAU()
    ser = SER()
    discriminator = Discriminator_SAU()
    """
    def BCE(y, yhat, eps = 1e-4):
        return tf.reduce_mean(-(y*tf.math.log(yhat + eps) + (y-1)*tf.math.log(1-yhat + eps)))
    """

    BCE = tf.keras.losses.BinaryCrossentropy()

    if load_path is not None :
        try:
            auto_encodeur.load_weights(os.getcwd()+load_path)
            print('auto_encodeur load sucessfuly')
        except:
            print("auto_encodeur not load succesfully from"+os.getcwd()+load_path)

    if load_SER_path is not None:
        try:
            ser.load_weights(os.getcwd()+load_path)
            print('ser load sucessfuly')
        except:
            print("ser not load succesfully from"+os.getcwd()+load_path)
            raise Exception('No SER load')
    #else:
    #    raise Exception("No SER path, please give one")

    #Création des summary
    log_dir        = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)
    if ircam:
        audio_log_dir        = "audio_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        audio_summary_writer = tf.summary.create_file_writer(audio_log_dir)
        mel_inv = Mel_inverter()
        l_mean_latent_ser = mean_SER_emotion(FILE_PATH, ser, 100)
        echantillon = emotion_echantillon(FILEPATH)
    

    print("Every thing is ready")
    cpt = 0
    for x,y in train_dataloader:
        
        x = normalisation(x)
        x_ = tf.reshape(x,(-1, 80)) #format (b*lenght, 80)
        lignes = np.repeat(np.arange(x.shape[0]),x.shape[1], axis = 0)
        mask   = np.where(x_ == 0)[0]
        x_     = np.delete(x_,mask, axis=0) # Delete le padding
        lignes = np.delete(lignes, mask, axis=0)
        ser_latent = ser.call_latent(x)
        ser_latent = np.array(ser_latent)[lignes] #association latent -> lignes
        l_order = np.arange(len(x_))

        np.random.shuffle(l_order)

        for b in range(len(x_)//BATCH_SIZE):
            cpt += 1
            print(cpt)
            indices = l_order[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            x__ = x_[indices]
            ser_latent_ = ser_latent[indices]

            with tf.GradientTape() as tape_gen: #, tf.GradientTape() as tape_disc:
                # Apprentissage générateur
                out = auto_encodeur(x__, ser_latent_)
                d_gen = discriminator(out)
                l_gen = BCE(tf.ones_like(d_gen),d_gen)

            grad_gen  = tape_gen.gradient(l_gen, auto_encodeur.trainable_variables)
            optimizer.apply_gradients(zip(grad_gen, auto_encodeur.trainable_variables))

            with tf.GradientTape() as tape_disc:
                # Apprentissage générateur
                d_true  = discriminator(x__)
                d_false = discriminator(out)
                l_disc  = BCE(tf.ones_like(d_true),d_true)+BCE(tf.zeros_like(d_false),d_false)
        
            grad_disc = tape_disc.gradient(l_disc, discriminator.trainable_variables)
            optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

            mdc = MDC_1D(out, x__)

            with summary_writer.as_default(): 
                tf.summary.scalar('train/loss_generateur',l_gen, step=cpt)
                tf.summary.scalar('train/loss_discriminateur',l_disc, step=cpt)
                tf.summary.scalar('train/mdc',mdc , step=cpt)
            break

        """
        TEST
        """
        if True : #test and ((cpt+1)%int(TEST_EPOCH*len_train) == 0):
            print('test_time')
            for c, (x,y)  in enumerate(test_dataloader):
                x = normalisation(x)
                x_ = tf.reshape(x,(-1, 80)) #format (b*lenght, 80)
                lignes = np.repeat(np.arange(x.shape[0]),x.shape[1], axis = 0)
                mask   = np.where(x_ == 0)[0]
                x_     = np.delete(x_,mask, axis=0) # Delete le padding
                lignes = np.delete(lignes, mask, axis=0)
                ser_latent = ser.call_latent(x)
                ser_latent = np.array(ser_latent)[lignes] #association latent -> lignes
                l_order = np.arange(len(x_))

                np.random.shuffle(l_order)
                l_order = l_order[:500]
                x_         = x_[l_order]
                ser_latent = ser_latent[l_order]
                
                out = auto_encodeur(x_, ser_latent)
                
                # Apprentissage générateur
                d_gen = discriminator(out)
                l_gen = BCE(np.ones(d_gen.shape),d_gen)

                d_true  = discriminator(x_)
                d_false = discriminator(tf.stop_gradient(out))
                
                l_true  = BCE(np.ones(d_true.shape),d_true)
                l_false = BCE(np.zeros(d_false.shape),d_false)

                mdc = MDC_1D(out, x_)
                with summary_writer.as_default(): 
                    tf.summary.scalar('test/loss_generateur',l_gen, step=cpt)
                    tf.summary.scalar('test/loss_discriminateur_true',l_true, step=cpt)
                    tf.summary.scalar('test/loss_discriminateur_false',l_false, step=cpt)
                    tf.summary.scalar('test/mdc',mdc , step=cpt)
                
                break
            test_dataloader.shuffle()

        if True : #(cpt+1) % len_train == 0:
            print("End batch")

            if ircam:
                """
                Pour un échantillon neutre, effectue une EVC avec toutes les
                émotions.
                """

                l_emotion = ['Angry','Happy', 'Neutral', 'Sad', 'Surprise']
                with audio_summary_writer.as_default(): 
                    x = echantillon[2] # On ne prend que le neutre
                    rec_x   = mel_inv.convert(de_normalisation(x)[0])
                    tf.summary.audio('Original',rec_x, 24000, step=cpt)
                    for i, emo in enumerate(l_emotion):
                        phi  = np.expand_dims(l_mean_latent_ser[i], axis=0)
                        out  = auto_encodeur(x, phi)
                        rec_out = mel_inv.convert(de_normalisation(out)[0])
                        tf.summary.audio('Reconstruct '+emo,rec_out, 24000,step=cpt)

            print("save")
            Auto_Encodeur_SAU.save_weights(log_dir, format(cpt//len_train))
            Discriminator_SAU.save_weights(log_dir, format(cpt//len_train))
            raise "End Test"


if __name__ == "__main__":
    longoptions = ['lock=', 'place=', 'load=', 'load_SER=']
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
        load_path     = ov.get('--load')
        load_SER_path = ov.get('--load_SER')
        train_dataloader, test_dataloader, data_queue, len_train = dataloader(FILEPATH, BATCH_SIZE, SHUFFLE, 
                                                                    LANGAGE, USE_DATA_QUEUE,)
        print("test_dataloader")
        for i in train_dataloader:
            print("ok")
            break
        print("end_test_dataloader")

        with tf.device(comp_device) :
            train(FILEPATH,train_dataloader, test_dataloader, len_train, test = True, 
                        ircam=True, load_path=load_path, load_SER_path = load_SER_path)
        if data_queue:
            data_queue.stop()

    else:
        FILEPATH = r"/home/luc/Documents/STAGE_IRCAM/data/ESD_Mel/"
        BATCH_SIZE = 30
        SHUFFLE    = True
        LANGAGE    = "english"
        USE_DATA_QUEUE = False
        load_path = ov.get('--load')
        load_SER_path = ov.get('--load_SER') 
        train_dataloader, test_dataloader, data_queue, len_train = dataloader(FILEPATH, BATCH_SIZE, SHUFFLE, 
                                                                    LANGAGE, USE_DATA_QUEUE)
        train(FILEPATH, train_dataloader, test_dataloader, len_train, test = True,
             load_path=load_path, load_SER_path = load_SER_path)
        if data_queue:
            data_queue.stop()
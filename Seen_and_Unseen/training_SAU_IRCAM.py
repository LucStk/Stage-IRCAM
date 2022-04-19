import os
from pickletools import optimize
from pexpect import ExceptionPexpect
from sklearn.model_selection import train_test_split
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


longoptions = ['lock=', 'load=', 'load_SER=']
ov, ra = getopt.getopt(sys.argv[1:], "", longoptions)
ov = dict(ov)

lck = ov.get("--lock")
if lck is None:
    soft = None
    print("WARNING : No lock taken")
elif lck.lower() == "soft": soft = True
elif lck.lower() == "hard": soft = False 
else:
    print("WARNING : lock arg not recognized, No lock taken")
    soft = None

if soft is not None:
    import manage_gpus as gpl
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
    FILEPATH = r"/home/luc/Documents/STAGE_IRCAM/data/ESD_Mel/"
    os.listdir(FILEPATH)
except:
    try:
        FILEPATH = r"/data2/anasynth_nonbp/sterkers/ESD_Mel/"
        os.listdir(FILEPATH)
    except:
        try:
            FILEPATH = r"/data/anasynth_nonbp/sterkers/ESD_Mel/"
            os.listdir(FILEPATH)
        except:
            raise Exception('Data not found')

BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 30
SHUFFLE    = True
LANGAGE    = "english"

EPOCH = 100
LR = 1e-4
TEST_EPOCH = 1/2
BATCH_SIZE = 256

load_path     = ov.get('--load')
load_SER_path = ov.get('--load_SER')

#
#with tf.device('/job:foo'):
with tf.device(comp_device) :
    optimizer = tf.keras.optimizers.RMSprop(learning_rate = LR)
    auto_encodeur = Auto_Encodeur_SAU()
    discriminator = Discriminator_SAU()
    ser = SER()
    BCE = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    ################################################################
    #                         Loading Model                        #
    ################################################################

    if load_path is not None :
        try:
            auto_encodeur.load_weights(os.getcwd()+load_path)
            print('auto_encodeur load sucessfuly')
        except:
            print("auto_encodeur not load succesfully from"+os.getcwd()+load_path)

    if load_SER_path is not None:
        try:
            ser.load_weights(os.getcwd()+load_SER_path)
            print('ser load sucessfuly')
        except:
            print("ser not load succesfully from"+os.getcwd()+load_SER_path)
            raise
    else:
        raise Exception("No SER load")

    #################################################################
    #                       Préparation data                        #
    #################################################################

    train_dataloader = ESD_data_generator_ALL_SAU(FILEPATH, ser, 
                                                  batch_size=BATCH_SIZE_TRAIN,
                                                  langage=LANGAGE)

    test_dataloader  = ESD_data_generator_ALL_SAU(FILEPATH, ser, 
                                                  batch_size=BATCH_SIZE_TEST,
                                                  langage=LANGAGE,
                                                  type_='test')

    len_train_dataloader = len(train_dataloader)
    len_test_dataloader  = len(test_dataloader)

    #Utilisation data_queue
    if True:
        print("begin data_queue")
        data_queue = tf.keras.utils.OrderedEnqueuer(train_dataloader, use_multiprocessing=False, shuffle=True)
        data_queue.start()
        train_dataloader = data_queue.get()
    
    print("Data_loaders ready")

    #Création des summary
    log_dir        = "SAU_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    #Préparation enregistrement
    audio_log_dir        = "audio_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    audio_summary_writer = tf.summary.create_file_writer(audio_log_dir)
    mel_inv = Mel_inverter()
    l_mean_latent_ser = mean_SER_emotion(FILEPATH, ser, 100)
    echantillon       = emotion_echantillon(FILEPATH)

    print("Every thing ready, beging training")
    for cpt, (x,z,y) in enumerate(train_dataloader):
        if (cpt % 10 == 0): print(cpt)

        #################################################################
        #                           TRAINING                            #
        #################################################################

        x = normalisation(x)
        with tf.GradientTape() as tape_gen:
            # Apprentissage générateur
            out   = auto_encodeur(x, z)
            d_gen = discriminator(out)
            l_gen = tf.reduce_mean(BCE(tf.ones_like(d_gen),d_gen))

        #grad_gen  = tape_gen.gradient(l_gen, auto_encodeur.trainable_variables)
        #optimizer.apply_gradients(zip(grad_gen, auto_encodeur.trainable_variables))

        with tf.GradientTape() as tape_disc:
            # Apprentissage générateur
            d_true  = discriminator(x)
            d_false = discriminator(out)
            l_true  = BCE(tf.ones_like(d_true),d_true)
            l_false = BCE(tf.zeros_like(d_false),d_false)
            l_disc  = tf.reduce_mean(tf.concat((l_true,l_false),axis=0))

            

        grad_disc = tape_disc.gradient(l_disc, discriminator.trainable_variables)
        optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))
        
        acc = (np.sum(d_true >= 0.5)+ np.sum(d_false < 0.5))/(2*len(d_false))
        mdc = MDC_1D(out, x)
        with summary_writer.as_default(): 
            tf.summary.scalar('train/loss_generateur',l_gen, step=cpt)
            tf.summary.scalar('train/loss_discriminateur',l_disc, step=cpt)
            tf.summary.scalar('train/acc',acc , step=cpt)
            tf.summary.scalar('train/mdc',mdc , step=cpt)

        #################################################################
        #                           TEST                                #
        #################################################################
        if (cpt+1)%int(TEST_EPOCH*len_train_dataloader) == 0:
            print("Test Time")
            (x,z,y) = test_dataloader[cpt%len_test_dataloader]
            x = normalisation(x)
            out = auto_encodeur(x, z)
            d_gen = discriminator(out)
            l_gen = tf.reduce_mean(BCE(np.ones(d_gen.shape),d_gen))

            d_true  = discriminator(x)
            d_false = discriminator(tf.stop_gradient(out))
            l_true  = tf.reduce_mean(BCE(np.ones(d_true.shape),d_true))
            l_false = tf.reduce_mean(BCE(np.zeros(d_false.shape),d_false))

            mdc = MDC_1D(out, x)
            with summary_writer.as_default(): 
                tf.summary.scalar('test/loss_generateur',l_gen, step=cpt)
                tf.summary.scalar('test/loss_discriminateur_true',l_true, step=cpt)
                tf.summary.scalar('test/loss_discriminateur_false',l_false, step=cpt)
                tf.summary.scalar('test/mdc',mdc , step=cpt)

        #################################################################
        #                Audio-logs et sauvegarde                       #
        #################################################################

        if (cpt+1) % (10*len_test_dataloader) == 0:
            print("End batch")
            print("Creation audio sample")
            """
            Pour un échantillon neutre, effectue une EVC avec toutes les
            émotions.
            """
            l_emotion = ['Angry','Happy', 'Neutral', 'Sad', 'Surprise']
            with audio_summary_writer.as_default(): 
                x = echantillon[2] # On ne prend que le neutre
                rec_x   = mel_inv.convert(de_normalisation(x))
                tf.summary.audio('Original',rec_x, 24000, step=cpt)
                for i, emo in enumerate(l_emotion):
                    tmp = np.expand_dims(l_mean_latent_ser[i], axis = 0)
                    phi = np.repeat(tmp, x.shape[0], axis = 0)
                    out  = auto_encodeur(x, phi)
                    rec_out = mel_inv.convert(de_normalisation(out))
                    tf.summary.audio('Reconstruct '+emo,rec_out, 24000,step=cpt)

            print("save")
            auto_encodeur.save_weights(log_dir, format(cpt//len_train_dataloader))
            discriminator.save_weights(log_dir, format(cpt//len_train_dataloader))

#resize_128b_8__20220504-134323

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from utilitaires.All_model import * 
from utilitaires.utils import *
import datetime
import sys
import getopt
import numpy as np
longoptions = ['lock=', 'load_SER=', 'name=']
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
        print("No GPU Available continue on CPU")
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
BATCH_SIZE_TEST  = 100
SHUFFLE    = True
LANGAGE    = "english"

EPOCH = 100
LR    = 50
TEST_EPOCH = 1/2

load_SER_path = ov.get('--load_SER')

with tf.device(comp_device) :

    img_optim = tf.keras.optimizers.Adam(learning_rate = LR)
    ser  = SER()
    loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)
    ################################################################
    #                         Loading Model                        #
    ################################################################

    if load_SER_path is not None:
        try:
            ser.load_weights(os.getcwd()+'/logs/SER_logs/'+load_SER_path)
            print('ser load sucessfuly')
        except:
            print("ser not load succesfully from"+os.getcwd()+'/logs/SER_logs/'+load_SER_path)
            raise
    else:
        raise Exception("No SER load")


    #################################################################
    #                       Préparation data                        #
    #################################################################

    #Préparation enregistrement
    audio_log_dir        = "logs/audio_logs/Dreams/Dream" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +ov.get("--name")
    audio_summary_writer = tf.summary.create_file_writer(audio_log_dir)
    
    mel_inv = Mel_inverter()
    l_mean_latent_ser = mean_SER_emotion(FILEPATH, ser, 100)
    echantillon       = emotion_echantillon(FILEPATH)

    ###################################################################
    #                            Training                             #
    ###################################################################
    x = echantillon[2] # On ne prend que le neutre
    x = tf.expand_dims(x, 0)
    x = tf.Variable(x)
    emotion = 0
    l_emotion = ['Angry','Happy', 'Neutral', 'Sad', 'Surprise']

    for cpt in range(10000):
        if cpt % 1000 == 0:
            print(cpt)
            r = mel_inv.convert(de_normalisation(x[0]))
            with audio_summary_writer.as_default():
                tf.summary.audio("reconstruct"+str(l_emotion[emotion]),r, 24000, step=cpt)
        with tf.GradientTape() as tape:
            x = tf.Variable(x)
            y_hat = ser.call_clas(x)
            l = loss(tf.expand_dims(tf.one_hot(emotion,5),0), y_hat)
        
        grad = tape.gradient(l , x)
        x    = x + grad*LR


import os
from pickletools import optimize
from sklearn.utils import shuffle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from utilitaires.dataloader_ESD_tf import ESD_data_generator
from model import Encodeur, Decodeur
import datetime
import sys

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def dataloader(FILEPATH, batch_size=30, shuffle=True, langage = 'english', use_data_queue= False):
    data_queue = None
    train_dataloader = ESD_data_generator(FILEPATH, batch_size, shuffle, langage)
    len_train = len(train_dataloader)
    if use_data_queue:
        print("begin data_queue")
        data_queue = tf.keras.utils.OrderedEnqueuer(train_dataloader, use_multiprocessing=True, shuffle=True)
        data_queue.start()
        train_dataloader = data_queue.get()    
    test_dataloader = ESD_data_generator(FILEPATH, batch_size=400, langage=langage, type_='test',shuffle=True)
    return train_dataloader, test_dataloader, data_queue, len_train

def train(train_dataloader, test_dataloader, len_train, test = False):
    print("Training Beging")
    MEAN_DATASET = -6.0056405
    STD_DATASET  = 2.4420118
    EPOCH = 10
    LR = 1e-4
    TEST_EPOCH = 1/10

    log_dir        = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)

    mse       = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate = LR)
    encodeur  = Encodeur(); decodeur = Decodeur()
    print("Every thing is ready")
    for x,_, cpt in enumerate(train_dataloader):
        """
        PRETRAITEMENT
        """
        x = tf.transpose(x, perm = [0,2,1])#batch, lenght, n
        x = (x - MEAN_DATASET)/STD_DATASET #Normalisation
        """
        TRAIN
        """
        with tf.GradientTape() as tape:
            latent, step = encodeur(x)
            out    = decodeur(latent,step)
            x = x[:,:out.shape[1]] #Crop pour les pertes de reconstruction du decodeur 
            mask   = tf.cast(x, tf.bool)
            out    = tf.multiply(out, tf.cast(mask, tf.float32))
            loss   = mse(x,out)
            with summary_writer.as_default(): 
                tf.summary.scalar('train/loss',loss , step=cpt)

        gradients = tape.gradient(loss, encodeur.trainable_variables+decodeur.trainable_variables)
        optimizer.apply_gradients(zip(gradients, encodeur.trainable_variables + decodeur.trainable_variables))            
        """
        TEST
        """
        if test and ((cpt+1)%int(TEST_EPOCH*len_train) == 0):
            print('test_time')
            for x,_ in test_dataloader:
                x = tf.transpose(x, perm = [0,2,1])#batch, lenght, n
                x = (x - MEAN_DATASET)/STD_DATASET #Normalisation
                latent, step = encodeur(x)
                out  = decodeur(latent, step)
                x = x[:,:out.shape[1]] #Crop pour les pertes de reconstruction du decodeur
                mask = tf.cast(x, tf.bool)
                out  = tf.multiply(out, tf.cast(mask, tf.float32))
                loss = mse(x,out)
                with summary_writer.as_default(): 
                    tf.summary.scalar('test/loss',loss, step=cpt)
                break

        if (cpt+1) % len_train == 0:
            print("save")
            decodeur.save_weights(log_dir+"/decodeur_checkpoint/{}".format(cpt//len_train))
            encodeur.save_weights(log_dir+"/encodeur_checkpoint/{}".format(cpt//len_train))

if __name__ == "__main__":
    try:
        args = sys.argv[1:][0].lower()
    except:
        args = None

    if args == 'ircam':
        print("ircam connexion")
        # Get gpu
        import manage_gpus as gpl
        try:
            soft = sys.argv[1:][1].lower() == 'soft'
        except:
            soft = False
        
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
            FILEPATH = r"/data/anasynth_nonbp/sterkers/ESD_Mel/"
            os.listdir(FILEPATH)
        
        BATCH_SIZE = 30
        SHUFFLE    = True
        LANGAGE    = "english"
        USE_DATA_QUEUE = True
        train_dataloader, test_dataloader, data_queue, len_train = dataloader(FILEPATH, BATCH_SIZE, SHUFFLE, 
                                                                    LANGAGE, USE_DATA_QUEUE)
        
        with tf.device(comp_device) :
            train(train_dataloader, test_dataloader, len_train, test = True)
        if data_queue:
            data_queue.stop()

    else:
        FILEPATH = r"/home/luc/Documents/STAGE_IRCAM/data/ESD_Mel/"
        BATCH_SIZE = 30
        SHUFFLE    = True
        LANGAGE    = "english"
        USE_DATA_QUEUE = False
        train_dataloader, test_dataloader, data_queue, len_train = dataloader(FILEPATH, BATCH_SIZE, SHUFFLE, 
                                                                    LANGAGE, USE_DATA_QUEUE)
        train(train_dataloader, test_dataloader, len_train, test = True)
        if data_queue:
            data_queue.stop()
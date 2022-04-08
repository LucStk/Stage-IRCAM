import os
from pickletools import optimize
from sklearn.utils import shuffle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from utilitaires.dataloader_ESD_tf import ESD_data_generator
from utilitaires.model import *
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

def train(train_dataloader, test_dataloader, len_train, 
          test = False, 
          ircam = False, 
          load_path = None):
    print("Training Beging")
    EPOCH = 100
    LR = 1e-4
    TEST_EPOCH = 1/2

    log_dir        = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = LR)
    Model = SER()
    if load_path is not None :
        try:
            Model.load_weights(os.getcwd()+load_path)
            print('model load sucessfuly')
        except:
            print("Load not succesful from"+os.getcwd()+load_path)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)
    print("Every thing is ready")
    for cpt, (x,y) in enumerate(train_dataloader):
        print(cpt)
        y_ = tf.one_hot(y,5)
        x = normalisation(x)
        with tf.GradientTape() as tape: #Normalisation
            y_hat  = Model(x)
            l = loss(y_,y_hat)
        ymax = tf.math.argmax(y_hat, axis = 1)
        acc = np.mean(y == ymax)
        with summary_writer.as_default():
            tf.summary.scalar('train/loss',l, step=cpt)
            tf.summary.scalar('train/acc',acc, step=cpt)


        gradients = tape.gradient(l, Model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, Model.trainable_variables))            
        """
        TEST
        """
        if test and ((cpt+1)%int(TEST_EPOCH*len_train) == 0):
            print('test_time')
            for c, (x,y)  in enumerate(test_dataloader):
                y_    = tf.one_hot(y,5)
                x     = normalisation(x) 
                y_hat = Model(x)
                l = loss(y_,y_hat)
                
                acc = np.mean(y == tf.math.argmax(y_hat, axis = 1))

                with summary_writer.as_default(): 
                    tf.summary.scalar('test/loss',l, step=cpt)
                    tf.summary.scalar('test/acc',acc, step=cpt)
                if c == 2:
                    break
            test_dataloader.shuffle()

        if (cpt+1) % len_train == 0:
            print("End batch")
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
        
        BATCH_SIZE = 56
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
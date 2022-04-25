import os
from sklearn.utils import shuffle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from utilitaires.dataloader_ESD_tf import ESD_data_generator_load
from utilitaires.All_model import *
from utilitaires.utils import *
import datetime
import sys
import numpy as np
import getopt
    
longoptions = ['lock=', 'load=']
ov, ra = getopt.getopt(sys.argv[1:], "", longoptions)
ov = dict(ov)

lck = ov.get("--lock")
if lck is None:
    soft = None
    print("WARNING : No lock taken")
elif lck.lower() == "soft": soft = True
elif lck.lower() == "hard": soft = False 
else:
    print("WARNING : No lock taken")
    soft = None

comp_device ="/cpu:0"
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


BATCH_SIZE_TRAIN = 56
BATCH_SIZE_TEST  = 200
SHUFFLE    = True
LANGAGE    = "english"
USE_DATA_QUEUE = True
LR = 1e-5
TEST_EPOCH = 1/2
load_path = ov.get('--load')

with tf.device(comp_device) :
    ser    = SER()
    optimizer = tf.keras.optimizers.Adam(learning_rate = LR)
    loss      = tf.keras.losses.BinaryCrossentropy(from_logits = True)

    if load_path is not None :
        try:
            ser.load_weights(os.getcwd()+load_path)
            print('model load sucessfuly')
        except:
            print("Load not succesful from"+os.getcwd()+load_path)
            raise

    print("Data loading")
    train_dataloader = ESD_data_generator_load(FILEPATH, BATCH_SIZE_TRAIN, SHUFFLE, LANGAGE)
    test_dataloader  = ESD_data_generator_load(FILEPATH, BATCH_SIZE_TEST , SHUFFLE, LANGAGE, type_='test')
    len_train_dataloader = len(train_dataloader)
    len_test_dataloader  = len(test_dataloader)

    if True:
        print("begin data_queue")
        data_queue = tf.keras.utils.OrderedEnqueuer(train_dataloader, use_multiprocessing=False, shuffle= True)
        data_queue.start(workers = 4, max_queue_size=20)
        train_dataloader = data_queue.get()

    log_dir        = "logs/SER_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)


    print("Training Beging")

    @tf.function(input_signature = [tf.TensorSpec(shape=(BATCH_SIZE_TRAIN, None, 80), dtype=tf.float32),
                                    tf.TensorSpec(shape=(BATCH_SIZE_TRAIN), dtype=tf.float32)])
    def train(x,y):
        y = tf.one_hot(y,5)
        x  = normalisation(x)
        with tf.GradientTape() as tape:
            y_hat = ser(x)
            l     = loss(y,y_hat)

        tr_var    = ser.trainable_variables
        gradients = tape.gradient(l, tr_var)
        optimizer.apply_gradients(zip(gradients, tr_var)) 
        acc  = tf.reduce_mean(tf.cast(tf.equal(y, tf.math.argmax(y_hat, axis = 1)), dtype= tf.float64))
        
        return {"loss_SER": l, "accurcay":acc}



    @tf.function
    def test(x,y):
        y_ = tf.one_hot(y,5)
        x  = normalisation(x)
        y_hat = ser(x)
        l     = loss(y_,y_hat)
        acc   = tf.reduce_mean(tf.cast(tf.equal(y, tf.math.argmax(y_hat, axis = 1)), dtype= tf.float64))
        return {"loss_SER": l, "accurcay":acc}

    def write(metric, type = 'train'):
        with summary_writer.as_default():
            for (k, v) in metric.items():
                tf.summary.scalar(type+'/'+k,v, step=cpt)

    for cpt, (x,y) in enumerate(train_dataloader):
        print(cpt)

        metric_train = train(x,y)
        
        if ((cpt +1) % 10) == 0:
            write(metric_train, "train")
            
        if ((cpt+1)%int(TEST_EPOCH*len_train_dataloader) == 0):
            print("test")
            (x,y) = test_dataloader[cpt%len_test_dataloader]
            metric_test = test(x,y)
            write(metric_test, "test")

        if (cpt+1) % (10*len_train_dataloader) == 0:
            print("save")
            ser.save(log_dir, format(cpt//len_train_dataloader))
            ser.save_weights(log_dir, format(cpt//len_train_dataloader))


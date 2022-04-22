import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from utilitaires.SAU_fit import * 
from utilitaires.utils import *
import datetime
import sys
import getopt

longoptions = ['lock=', 'load=', 'load_SER=', 'no_metrics=']
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
LR    = 1e-5
LR_AE = 1e-4
TEST_EPOCH = 1/2

load_path     = ov.get('--load')
load_SER_path = ov.get('--load_SER')
no_metrics    = ov.get('--no_metrics') == "true"

with tf.device(comp_device) :
    tf.debugging.set_log_device_placement(True)
    #optimizer = tf.keras.optimizers.RMSprop(learning_rate = LR)
    SAU = SAU_GAN()
    ser = SER()
    ################################################################
    #                         Loading Model                        #
    ################################################################

    if load_path is not None :
        try:
            SAU_GAN.load_weights(os.getcwd()+load_path)
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
        data_queue.start(4, 20)
        train_dataloader = data_queue.get()

    print("Every thing ready, beging training")
    disc_optim = tf.keras.optimizers.Adam(learning_rate = LR)
    ae_optim   = tf.keras.optimizers.RMSprop(learning_rate = LR_AE)
    BCE = tf.keras.losses.BinaryCrossentropy()
    MSE = tf.keras.losses.MeanSquaredError()

    SAU.compile(ae_optim,disc_optim, MSE, steps_per_execution=4)
    #SAU.build(input_shape=(None,80+128))
    #print(SAU(train_dataloader[0]))
    SAU.fit(train_dataloader, epochs = 10) 


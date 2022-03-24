import os
from pickletools import optimize
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from utils.dataloader_ESD_tf import ESD_data_generator
from model import Encodeur, Decodeur
from torch.utils.tensorboard import SummaryWriter
import torch
import tensorboard
import datetime
import sys

def train(FILEPATH, use_data_queue = False, test = False):
    EPOCH = 1
    BATCH_SIZE = 30
    LR = 1e-4
    TEST_EPOCH = 1/10

    train_dataloader = ESD_data_generator(FILEPATH, BATCH_SIZE, shuffle=True, langage="english")
    if use_data_queue:
        data_queue = tf.keras.utils.SequenceEnqueuer(train_dataloader, use_multiprocessing= True)
        data_queue.start()
        train_dataloader = data_queue.get()
    
    test_dataloader = ESD_data_generator(FILEPATH, 100, type_='test',shuffle=True,langage="english")
    
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)

    mse = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate = LR)
    encodeur = Encodeur(); decodeur = Decodeur()
    cpt = 0

    for _ in range(EPOCH):
        for x,y in train_dataloader:
            cpt += 1
            x = tf.transpose(x, perm = [0,2,1])#batch, lenght, n
            with tf.GradientTape() as tape:
                latent = encodeur(x)
                out  = decodeur(latent, x.shape[1])

                mask = tf.cast(x, tf.bool)
                out  = tf.multiply(out, tf.cast(mask, tf.float32))
                loss = mse(x,out)

                writer.add_scalar("train/loss",loss.numpy(), cpt)
                writer.flush()
            gradients = tape.gradient(loss, encodeur.trainable_variables+decodeur.trainable_variables)
            optimizer.apply_gradients(zip(gradients, encodeur.trainable_variables + decodeur.trainable_variables))            
    
            if test and cpt%(int(len(train_dataloader)*TEST_EPOCH)) == 0:
                for x,y in test_dataloader:
                    latent = encodeur(x)
                    out  = decodeur(latent, x.shape[1])

                    mask = tf.cast(x, tf.bool)
                    out  = tf.multiply(out, tf.cast(mask, tf.float32))
                    loss = mse(x,out)

    if use_data_queue:
        data_queue.stop()


if __name__ == "__main__":
    try:
        args = sys.argv[1:][0].lower()
    except:
        args = None

    if args == 'ircam':
        import manage_gpus as gpl
        FILEPATH = r"/data2/anasynth_nonbp/sterkers/ESD_Mel/"
        try:
            soft = sys.argv[1:][1]
        except:
            soft = False
        
        try:
            gpu_id_locked = gpl.get_gpu_lock(gpu_device_id=-1, soft=soft)
        except gpl.NoGpuManager:
            print("no gpu manager available - will use all available GPUs", file=sys.stderr)
        except manage_gpus.NoGpuAvailable:
            # there is no GPU available for locking, continue with CPU
            comp_device = "/cpu:0" 
            os.environ["CUDA_VISIBLE_DEVICES"]=""

    else:
        FILEPATH = r"/home/luc/Documents/STAGE_IRCAM/data/ESD_Mel/"

    train(FILEPATH)
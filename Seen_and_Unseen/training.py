import os
from pickletools import optimize
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from dataloader import AttHACK_Mell_spectrogram, collate_fn
from model import Encodeur, Decodeur
from torch.utils.tensorboard import SummaryWriter
import torch
import tensorboard
import datetime
import sys

def train(FILEPATH):
    EPOCH = 1
    BATCH_SIZE = 30
    LR = 1e-3

    base = AttHACK_Mell_spectrogram(FILEPATH)
    data_loader = torch.utils.data.DataLoader(base, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)

    mse = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate = LR)
    encodeur = Encodeur()
    decodeur = Decodeur()

    cpt = 0
    for e in range(EPOCH):
        for batch in data_loader:
            cpt += 1
            x = tf.transpose(batch["data"], perm = [0,2,1])
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
        writer.flush()

if __name__ == "__main__":
    if sys.argv[1:][0].lower() == 'ircam':
        import manage_gpus as gpl
        FILEPATH = r"/net/as-sdb/data/speech/corpus/Att-HACK"
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
        FILEPATH = r"/home/luc/Documents/STAGE_IRCAM/data/melspcs/"

    train(FILEPATH)
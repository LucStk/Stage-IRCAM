import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from utilitaires.All_model import * 
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
LR_AE = 1e-4
TEST_EPOCH = 1/2

load_path     = ov.get('--load')
load_SER_path = ov.get('--load_SER')
no_metrics    = ov.get('--no_metrics') == "true"

with tf.device(comp_device) :
    ae_optim   = tf.keras.optimizers.RMSprop(learning_rate = LR_AE)
    ae_optim   = tf.keras.mixed_precision.LossScaleOptimizer(ae_optim)
    
    auto_encodeur = Auto_Encodeur_SAU()
    #auto_encodeur.compile()

    ser = SER()
    BCE = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    MSE = tf.keras.losses.MeanSquaredError()
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


    #Création des summary
    log_dir        = "logs/AE_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    #Préparation enregistrement
    audio_log_dir        = "logs/audio_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    audio_summary_writer = tf.summary.create_file_writer(audio_log_dir)
    
    #
    mel_inv = Mel_inverter()
    l_mean_latent_ser = mean_SER_emotion(FILEPATH, ser, 100)
    echantillon       = emotion_echantillon(FILEPATH)


    if True:
        print("begin data_queue")
        data_queue = tf.keras.utils.OrderedEnqueuer(train_dataloader, 
                                                    use_multiprocessing=False, 
                                                    shuffle=True)
        data_queue.start(4, 20)
        train_dataloader = data_queue.get()

    print("Every thing ready, beging training")

    @tf.function
    def train(input):
        x = input[:,:80]
        z = input[:,80:]
        x = normalisation(x)
        with tf.GradientTape() as tape_gen:
            out   = auto_encodeur(x, z)
            l_gen = MSE(x, out)
            l_gen = ae_optim.get_scaled_loss(l_gen)

        tr_var   = auto_encodeur.trainable_variables
        grad_gen = tape_gen.gradient(l_gen, tr_var)
        grad_gen = ae_optim.get_unscaled_gradients(grad_gen)
        ae_optim.apply_gradients(zip(grad_gen, tr_var))



    """
        with summary_writer.as_default(): 
            tf.summary.scalar('train/loss_generateur',l_gen, step=cpt)
            
    @tf.function
    def test(cpt):
        print("Test Time")
        input = test_dataloader[cpt%len_test_dataloader]
        x = input[:,:80]
        z = input[:,80:]

        x = normalisation(x)
        out   = auto_encodeur(x, z)
        l_gen = MSE(x, out)

        with summary_writer.as_default(): 
            tf.summary.scalar('test/loss_generateur',l_gen, step=cpt)
    
    @tf.function
    def create_audio(cpt):
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


    ###################################################################
    #                            Training                             #
    ###################################################################
    
    import contextlib
    @contextlib.contextmanager
    def options(options):
        old_opts = tf.config.optimizer.get_experimental_options()
        tf.config.optimizer.set_experimental_options(options)
        try:
            yield
        finally:
            tf.config.optimizer.set_experimental_options(old_opts)

    with options({'constant_folding': True}):
    """
    for cpt, x in enumerate(train_dataloader):
        train(x)

        """
        if (cpt+1)%int(TEST_EPOCH*len_train_dataloader) == 0:
            print("test ",cpt)
            test(cpt)
        
        if (cpt+1) % (2*len_test_dataloader) == 0:
            create_audio(cpt)
        """
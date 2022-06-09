#%%
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import tensorflow as tf
from utilitaires.All_model import * 
from utilitaires.utils import *
import numpy as np

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

BATCH_SIZE_TEST = 100
SHUFFLE         = True
LANGAGE         = "english"

load_SER_path = "resize_128b_8_3_20220511-155842"
#load_SER_path = "resize_128b_16__20220504-132520"
#load_SER_path = "resize_128b_4__20220504-135654"

ser  = SER()
file = os.getcwd()[:-5]+'/logs/SER_logs/'+ load_SER_path

ser.load_weights(file)
test_dataloader = ESD_data_generator_load(FILEPATH, BATCH_SIZE_TEST , SHUFFLE, LANGAGE, type_='test')

out  = []
out2 = []
Y    = []
for i in range(10):
    (x,y) = test_dataloader[0]
    x     = normalisation(x)

    out  += [ser.call_clas(x).numpy()]
    out2 += [ser.call_latent(x).numpy()]
    Y += [y]
out = np.concatenate(out)
out2 = np.concatenate(out2)
y = np.concatenate(Y)
    



#%% Affichage
indice  = range(5)
indices = [i in indice for i in y]
out_     = out[indices]
out2_    = out2[indices]


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

T_SNE = TSNE() #(perplexity = 30, learning_rate=100)
tr = T_SNE.fit_transform(out_,y)

colors = ["blue", "red", "green", "yellow", "black"]
#Dans l'ordre : Angry, Happy, Neutral, Sad, Surprise

for i in indice:
    tmp = tr[np.where(y[indices] == i)]
    plt.scatter(tmp[:,0], tmp[:,1], c=colors[i])   
plt.show()

# latent
tr = T_SNE.fit_transform(out2_,y)

colors = ["blue", "red", "green", "yellow", "black"]
#Dans l'ordre : Angry, Happy, Neutral, Sad, Surprise

for i in indice:
    tmp = tr[np.where(y[indices] == i)]
    plt.scatter(tmp[:,0], tmp[:,1], c=colors[i])   
plt.show()

     #%%
mat = np.zeros((5,5))
yhat = tf.math.argmax(out, axis = 1,output_type=tf.dtypes.int32)
for i,j in zip(y,yhat):
    mat[i][j] += 1


plt.imshow(mat)
plt.show()
mat = np.log(mat+0.01)
plt.imshow(mat)
plt.show()

# %%

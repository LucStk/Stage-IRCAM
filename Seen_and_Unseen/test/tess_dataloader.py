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
#%%
ser  = SER()
file = os.getcwd()[:-5]+'/logs/SER_logs/'+ load_SER_path
ser.load_weights(file)

test_dataloader  = ESD_data_generator_ALL_SAU(FILEPATH, ser, 
                                            batch_size=BATCH_SIZE_TEST,
                                            langage=LANGAGE,
                                            type_='test')
#%%
test_dataloader.shuffle()

x,z,y = test_dataloader[0]
indice  = range(5)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

T_SNE = TSNE() #(perplexity = 30, learning_rate=100)
tr = T_SNE.fit_transform(z,y)

colors = ["blue", "red", "green", "yellow", "black"]
#Dans l'ordre : Angry, Happy, Neutral, Sad, Surprise

for i in range(5):
    tmp = tr[np.where(y == i)]
    plt.scatter(tmp[:,0], tmp[:,1], c=colors[i])   
plt.show()




# %%
from pathlib import Path

def filtre_lg(l, lg): # Selection english or chinese audio or both
    dict_lg = {"english": "([0]{2}(?:[1][1-9]|[2][0]))", 
                "chinese" :"([0]{2}(?:[0]\d|[1][0]))", 
                None : ".*"}
    ret = []
    for i in l:
        i = i.__str__() #Only work if the directory is named ESD
        if re.match('(?:.*\/ESD_Mel\/'+dict_lg[lg]+')',i):
            ret += [i]
    return ret

p = Path(FILEPATH)
ser  = SER()
file = os.getcwd()[:-5]+'/logs/SER_logs/'+ load_SER_path
ser.load_weights(file)

list_emotions = ['Angry', "Happy", "Neutral", "Sad", "Surprise"]
ret    = []
b_size = 100

dataname = filtre_lg(p.glob('**/{}/*.p'.format('test')), "english")
x    = np.empty(len(dataname), object)
x[:] = [pd.read_pickle(f) for f in dataname]
y = np.array([list_emotions.index(re.findall("((?:\w|\.)+)", l)[-3]) for l in dataname])


for deb,end in zip(range(0,len(x)+b_size,b_size), range(b_size,len(x)+b_size,b_size)):
    x_1 = auto_padding(x[deb:end])
    x_2 = tf.transpose(x_1, (0,2,1))
    x_3 = normalisation(x_2)
    x_4 = ser.call_clas(x_3)
    ret.append(x_4)
    print('|', end="", flush=True)

print("latent created")    
z  = tf.concat(ret, axis = 0)

# %% VISUALISATION
indices  = np.arange(len(y))
np.random.shuffle(indices)
indices = indices[:200]

z_ = tf.gather(z, indices=indices)
y_ = tf.gather(y, indices=indices)

#%%Affichage
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

T_SNE = TSNE() #(perplexity = 30, learning_rate=100)
tr = T_SNE.fit_transform(z_,y_)

colors = ["blue", "red", "green", "yellow", "black"]
#Dans l'ordre : Angry, Happy, Neutral, Sad, Surprise

for i in range(5):
    tmp = tr[np.where(y_ == i)]
    plt.scatter(tmp[:,0], tmp[:,1], c=colors[i])   
plt.show()

# %%
#%%

test_dataloader = ESD_data_generator_load(FILEPATH, 100, False, LANGAGE, type_='test')
(xo,yo) = test_dataloader[0] 




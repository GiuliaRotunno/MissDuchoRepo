'''
#script to test the functionality of zeros, random and mean methods
from ducho.imputation.zeros import ZerosImputator
from ducho.imputation.random import RandomImputator
from ducho.imputation.mean import MeanImputator

list_of_modalities = ['visual','textual']
#ZerosImputator(list_of_modalities).impute()    
RandomImputator(list_of_modalities, random_seed=42).impute()
#MeanImputator(list_of_modalities).impute()      #takes a few minutes, but works


# verifica contentuto file numpy 
import numpy as np

#data = np.load('local/data/demo_baby/visual_embeddings_zeros/B002KSCA8U.npy')
#data1 = np.load('local/data/demo_baby/textual_embeddings_zeros/B00AIVN89G.npy')
#print("Tipo di dati:", type(data), "Forma dell'array:", data.shape, "Tipo degli elementi:", data.dtype, "Contenuto:", data[:5])             
#print("Tipo di dati:", type(data1), "Forma dell'array:", data1.shape, "Tipo degli elementi:", data1.dtype, "Contenuto:", data1[:5])

data = np.load('local/data/demo_baby/visual_embeddings_random/B002KSCA8U.npy')
data1 = np.load('local/data/demo_baby/textual_embeddings_random/B00AIVN89G.npy')
print("Tipo di dati:", type(data), "Forma dell'array:", data.shape, "Tipo degli elementi:", data.dtype, "Contenuto:", data[:5])             #0.0954  0.01
print("Tipo di dati:", type(data1), "Forma dell'array:", data1.shape, "Tipo degli elementi:", data1.dtype, "Contenuto:", data1[:5])         #4.21  -2.11

#data = np.load('local/data/demo_baby/visual_embeddings_mean/B002KSCA8U.npy')
#data1 = np.load('local/data/demo_baby/textual_embeddings_mean/B00AIVN89G.npy')
#print("Tipo di dati:", type(data), "Forma dell'array:", data.shape, "Tipo degli elementi:", data.dtype, "Contenuto:", data[:5])               
#print("Tipo di dati:", type(data1), "Forma dell'array:", data1.shape, "Tipo degli elementi:", data1.dtype, "Contenuto:", data1[:5])          

#PYTHONPATH=. python demos/demo_baby/run.py 



import numpy as np
data = np.load('imputed/local/data/demo_giulia/visual_embeddings/torch/ResNet50/avgpool/random/B002KSCA8U.npy')
data1 = np.load('imputed/local/data/demo_giulia/visual_embeddings/torch/ResNet50/avgpool/random/B0041SRGG8.npy')
print("Tipo di dati0:", type(data), "Forma dell'array:", data.shape, "Tipo degli elementi:", data.dtype, "Contenuto:", data)         
print("Tipo di dati1:", type(data1), "Forma dell'array:", data1.shape, "Tipo degli elementi:", data1.dtype, "Contenuto:", data1) 

import numpy as np
from ducho.imputation.random import RandomImputator
RandomImputator(missing_items=['B002KSCA8U', 'B0041SRGG8'], full_output_path='imputed/local/data/demo_giulia/visual_embeddings/torch/ResNet50/avgpool/random_twin', embedding_dim=1, seed=42).impute()
datatwin= np.load('imputed/local/data/demo_giulia/visual_embeddings/torch/ResNet50/avgpool/random_twin/random/B002KSCA8U.npy')
data1twin = np.load('imputed/local/data/demo_giulia/visual_embeddings/torch/ResNet50/avgpool/random_twin/random/B0041SRGG8.npy')
print("Tipo di dati 0:", type(datatwin), "Forma dell'array:", datatwin.shape, "Tipo degli elementi:", datatwin.dtype, "Contenuto:", datatwin)         
print("Tipo di dati 1:", type(data1twin), "Forma dell'array:", data1twin.shape, "Tipo degli elementi:", data1twin.dtype, "Contenuto:", data1twin) 
'''
'''
import numpy as np
from ducho.imputation.random import RandomImputator
RandomImputator(missing_items=['B002KSCA8U', 'B0041SRGG8'], full_output_path='imputed/local/data/demo_giulia/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1/random_twin', embedding_dim=1, seed=42).impute()
datatwin= np.load('imputed/local/data/demo_giulia/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1/random_twin/random/B002KSCA8U.npy')
data1twin = np.load('imputed/local/data/demo_giulia/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1/random_twin/random/B0041SRGG8.npy')
print("Tipo di dati 0:", type(datatwin), "Forma dell'array:", datatwin.shape, "Tipo degli elementi:", datatwin.dtype, "Contenuto:", datatwin)         
print("Tipo di dati 1:", type(data1twin), "Forma dell'array:", data1twin.shape, "Tipo degli elementi:", data1twin.dtype, "Contenuto:", data1twin) 


import numpy as np
emb = np.load('imputed/local/data/demo_giulia/visual_embeddings/torch/ResNet50/avgpool/neighmean/top_k_1_indexed/4.npy')
print(emb)
print(emb.shape)


import numpy as np
data = np.load('imputed/local/data/demo_giulia/visual_embeddings_textual_embeddings_concat/transformers/openai/clip-vit-base-patch16/1/mean/B000M41BRA.npy')
data1 = np.load('imputed/local/data/demo_giulia/visual_embeddings_textual_embeddings_concat/transformers/openai/clip-vit-base-patch16/1/mean/B0050O92FU.npy')
print("Tipo di dati0:", type(data), "Forma dell'array:", data.shape, "Tipo degli elementi:", data.dtype, "Contenuto:", data)         
print("Tipo di dati1:", type(data1), "Forma dell'array:", data1.shape, "Tipo degli elementi:", data1.dtype, "Contenuto:", data1) 

#B000M41BRA	http://ecx.images-amazon.com/images/I/51FH4tsRKrL._SY300_.jpg
#B0050O92FU	http://ecx.images-amazon.com/images/I/41ZQovKqKhL._SY300_.jpg

from ducho.imputation.vigan_imputation.networks import *
visual_dim = 2048  # ResNet50
textual_dim = 768  # MPNet
AE = define_AE(visual_dim, textual_dim, gpu_ids=[])
G_A = define_G(visual_dim, textual_dim, gpu_ids=[])
G_B = define_G(textual_dim, visual_dim, gpu_ids=[]) 
D_A = define_D(textual_dim, gpu_ids=[])
D_B = define_D(visual_dim, gpu_ids=[])
print("Reti create con successo!")
print_network(AE)

import torch
from ducho.imputation.vigan_imputation.viganmodel import VIGANModel
visual_dim = 2048
textual_dim = 768
model = VIGANModel(
    input_A_dim=visual_dim,
    input_B_dim=textual_dim,
    gpu_ids=[],  
    use_lsgan=True
)
model.setup_optimizers(lr=0.0002, beta1=0.5)
batch_size = 32
real_A = torch.randn(batch_size, visual_dim)
real_B = torch.randn(batch_size, textual_dim)
model.set_input(real_A, real_B)
print("Modello VIGAN creato con successo!")

import numpy as np
datav = np.load('local/data/demo_giulia/visual_embeddings/torch/ResNet50/avgpool/097293751X.npy')
datat = np.load('local/data/demo_giulia/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1/097293751X.npy')
print("Tipo di dati0:", type(datav), "Forma dell'array:", datav.shape, "Tipo degli elementi:", datav.dtype, "Contenuto:", datav)         
print("Tipo di dati1:", type(datat), "Forma dell'array:", datat.shape, "Tipo degli elementi:", datat.dtype, "Contenuto:", datat) 


from ducho.imputation.vigan_imputation.vigan_imputator import ViganImputator
missing_visual_file = 'missing_visual.tsv'
missing_textual_file = 'missing_textual.tsv'
dataset_path = './local/data/demo_giulia'
visual_dim = 2048
textual_dim = 768
gpu_ids=[]
print(f"Applying vigan imputation.")
imputer = ViganImputator(dataset_path, missing_visual_file, missing_textual_file, visual_dim, textual_dim, gpu_ids)
imputer.impute()
print(f"Vigan imputation completed.")   

import numpy as np
datav = np.load('imputed/local/data/demo_giulia/visual_embeddings/torch/ResNet50/avgpool/vigan/B002KSCA8U.npy')
datat = np.load('imputed/local/data/demo_giulia/textual_embeddings/sentence-transformers/sentence-transformers/all-mpnet-base-v2/1/vigan/B000M41BRA.npy')
print("Tipo di dati0:", type(datav), "Forma dell'array:", datav.shape, "Tipo degli elementi:", datav.dtype, "Contenuto:", datav)         
print("Tipo di dati1:", type(datat), "Forma dell'array:", datat.shape, "Tipo degli elementi:", datat.dtype, "Contenuto:", datat) 

import numpy as np
#datav = np.load('local/data/demo_giulia/imputed/crossmodal/visual_embeddings/smil/B002KSCA8U.npy')
datat = np.load('local/data/demo_giulia/test/imputed_embeddings/neighmean/top_k_3/B00003TL7P.npy')
#print("Tipo di dati0:", type(datav), "Forma dell'array:", datav.shape, "Tipo degli elementi:", datav.dtype, "Contenuto:", datav)         
print("Tipo di dati1:", type(datat), "Forma dell'array:", datat.shape, "Tipo degli elementi:", datat.dtype, "Contenuto:", datat) 

with open("output_completo.txt", "w") as f:
    np.set_printoptions(threshold=np.inf)  # mostra tutto senza troncare
    f.write(str(datat))

import numpy as np
datav = np.load('local/data/demo_giulia/imputed/crossmodal/visual_embeddings/autoenc/B002KSCA8U.npy')
datav1 = np.load('local/data/demo_giulia/imputed/crossmodal/visual_embeddings/autoenc/B0041SRGG8.npy')
print("Tipo di dati0:", type(datav), "Forma dell'array:", datav.shape, "Tipo degli elementi:", datav.dtype, "Contenuto:", datav)         
print("Tipo di dati1:", type(datav1), "Forma dell'array:", datav1.shape, "Tipo degli elementi:", datav1.dtype, "Contenuto:", datav1) 

datat = np.load('local/data/demo_giulia/imputed/crossmodal/textual_embeddings/autoenc/B000M41BRA.npy')
datat1 = np.load('local/data/demo_giulia/imputed/crossmodal/textual_embeddings/autoenc/B0050O92FU.npy')
print("Tipo di dati0:", type(datat), "Forma dell'array:", datat.shape, "Tipo degli elementi:", datat.dtype, "Contenuto:", datat)         
print("Tipo di dati1:", type(datat1), "Forma dell'array:", datat1.shape, "Tipo degli elementi:", datat1.dtype, "Contenuto:", datat1) 
'''

'''

with open("local/data/demo_office/final_meta.tsv", "r", encoding="utf-8") as fin, \
     open("local/data/demo_office/items.tsv", "w", encoding="utf-8") as fout:
    for line in fin:
        fout.write(line.split("\t")[0] + "\n")
'''






'''
import os
import shutil

os.makedirs('./local/data/demo_office/aval_emb_neighmean_textual', exist_ok=True) 
with open('./local/data/demo_office/test_items_neighmean_textual.tsv', "r", encoding="utf-8") as f: 
    nomi_da_escludere = {line.strip().split("\t")[0] + ".npy" for line in f if line.strip()}

for nome in os.listdir('./local/data/demo_office/extracted/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1'):
    src = os.path.join('./local/data/demo_office/extracted/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1', nome)
    dst = os.path.join('./local/data/demo_office/aval_emb_neighmean_textual', nome)

    if os.path.isfile(src) and nome not in nomi_da_escludere:
        shutil.copy2(src, dst)
'''
'''
import os
import shutil

os.makedirs('./local/data/demo_office/test_emb_neighmean_textual', exist_ok=True)  
with open('./local/data/demo_office/test_items_neighmean_textual.tsv', "r", encoding="utf-8") as f: 
    nomi_da_escludere = {line.strip().split("\t")[0] + ".npy" for line in f if line.strip()}

for nome in os.listdir('./local/data/demo_office/extracted/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1'):
    src = os.path.join('./local/data/demo_office/extracted/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1', nome)
    dst = os.path.join('./local/data/demo_office/test_emb_neighmean_textual', nome)

    if os.path.isfile(src) and nome in nomi_da_escludere:
        shutil.copy2(src, dst)


numero_elementi = len(os.listdir('./local/data/demo_office/test_emb_neighmean_textual'))
print(numero_elementi)
'''

'''
import numpy as np
import os

### funzioni per il calcolo degli errori di ricostruzione
def compute_reconstruction_errors(real_embeddings, imputed_embeddings, eps=1e-8):
      mse = []
      cosine_sim = []

      for real_emb, imp_emb in zip(real_embeddings, imputed_embeddings):
         
         real_emb = np.asarray(real_emb).reshape(-1)
         imp_emb = np.asarray(imp_emb).reshape(-1)

         #MSE error
         mse.append(((imp_emb - real_emb)**2).mean())
         
         # Cosine similarity
         numerator = np.dot(real_emb, imp_emb)
         denominator = (np.linalg.norm(real_emb) * np.linalg.norm(imp_emb) + eps)
         cosine_sim.append(numerator / denominator)

      return np.mean(mse),np.mean(cosine_sim)

def compute_reconstruction_errors_for_modality(input_path, result_path):  
        
        real_embeddings = [np.load(os.path.join(input_path, file)) for file in os.listdir(input_path)]
        imputed_embeddings = [np.load(os.path.join(result_path, file)) for file in os.listdir(result_path)]
        return compute_reconstruction_errors(real_embeddings, imputed_embeddings)

mse, cosine_sim = compute_reconstruction_errors_for_modality('./local/data/demo_office/test_emb_neighmean_textual', './local/data/demo_office/imputed_emb_heat_textual/heat')
print(f"Test result heat: mse={mse}, cosine similarity={cosine_sim}")
'''

'''
from math import ceil
input_file = "./local/data/demo_office/test_items_graphautoenc.tsv"
output_file_1 = "./local/data/demo_office/test_items_graphautoenc_visual.tsv"
output_file_2 = "./local/data/demo_office/test_items_graphautoenc_textual.tsv"

with open(input_file, "r", encoding="utf-8") as f:
    righe = f.readlines()
header = righe[0]
dati = righe[1:]
meta = ceil(len(dati) / 2)
prima_meta = dati[:meta]
seconda_meta = dati[meta:]

with open(output_file_1, "w", encoding="utf-8") as f:
    f.write(header)
    f.writelines(prima_meta)

with open(output_file_2, "w", encoding="utf-8") as f:
    f.write(header)
    f.writelines(seconda_meta)
'''

'''
import os
import shutil

os.makedirs('./local/data/demo_office/aval_emb_graphautoenc_textual', exist_ok=True) 
with open('./local/data/demo_office/test_items_graphautoenc_textual.tsv', "r", encoding="utf-8") as f: 
    nomi_da_escludere = {line.strip().split("\t")[0] + ".npy" for line in f if line.strip()}

for nome in os.listdir('./local/data/demo_office/extracted/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1'):
    src = os.path.join('./local/data/demo_office/extracted/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1', nome)
    dst = os.path.join('./local/data/demo_office/aval_emb_graphautoenc_textual', nome)

    if os.path.isfile(src) and nome not in nomi_da_escludere:
        shutil.copy2(src, dst)

os.makedirs('./local/data/demo_office/aval_emb_graphautoenc_visual', exist_ok=True) 
with open('./local/data/demo_office/test_items_graphautoenc_visual.tsv', "r", encoding="utf-8") as f: 
    nomi_da_escludere = {line.strip().split("\t")[0] + ".npy" for line in f if line.strip()}

for nome in os.listdir('./local/data/demo_office/extracted/visual_embeddings/torch/ResNet50/avgpool'):
    src = os.path.join('./local/data/demo_office/extracted/visual_embeddings/torch/ResNet50/avgpool', nome)
    dst = os.path.join('./local/data/demo_office/aval_emb_graphautoenc_visual', nome)

    if os.path.isfile(src) and nome not in nomi_da_escludere:
        shutil.copy2(src, dst)



os.makedirs('./local/data/demo_office/test_emb_graphautoenc_textual', exist_ok=True)  
with open('./local/data/demo_office/test_items_graphautoenc_textual.tsv', "r", encoding="utf-8") as f: 
    nomi_da_escludere = {line.strip().split("\t")[0] + ".npy" for line in f if line.strip()}

for nome in os.listdir('./local/data/demo_office/extracted/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1'):
    src = os.path.join('./local/data/demo_office/extracted/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1', nome)
    dst = os.path.join('./local/data/demo_office/test_emb_graphautoenc_textual', nome)

    if os.path.isfile(src) and nome in nomi_da_escludere:
        shutil.copy2(src, dst)


numero_elementi = len(os.listdir('./local/data/demo_office/test_emb_graphautoenc_textual'))
print(numero_elementi)

import os
import shutil

os.makedirs('./local/data/demo_office/test_emb_graphautoenc_visual', exist_ok=True)  
with open('./local/data/demo_office/test_items_graphautoenc_visual.tsv', "r", encoding="utf-8") as f: 
    nomi_da_escludere = {line.strip().split("\t")[0] + ".npy" for line in f if line.strip()}

for nome in os.listdir('./local/data/demo_office/extracted/visual_embeddings/torch/ResNet50/avgpool'):
    src = os.path.join('./local/data/demo_office/extracted/visual_embeddings/torch/ResNet50/avgpool', nome)
    dst = os.path.join('./local/data/demo_office/test_emb_graphautoenc_visual', nome)

    if os.path.isfile(src) and nome in nomi_da_escludere:
        shutil.copy2(src, dst)

numero_elementi = len(os.listdir('./local/data/demo_office/test_emb_graphautoenc_visual'))
print(numero_elementi)



import numpy as np
import os

### funzioni per il calcolo degli errori di ricostruzione
def compute_reconstruction_errors(real_embeddings, imputed_embeddings, eps=1e-8):
      mse = []
      cosine_sim = []

      for real_emb, imp_emb in zip(real_embeddings, imputed_embeddings):
         
         real_emb = np.asarray(real_emb).reshape(-1)
         imp_emb = np.asarray(imp_emb).reshape(-1)

         #MSE error
         mse.append(((imp_emb - real_emb)**2).mean())
         
         # Cosine similarity
         numerator = np.dot(real_emb, imp_emb)
         denominator = (np.linalg.norm(real_emb) * np.linalg.norm(imp_emb) + eps)
         cosine_sim.append(numerator / denominator)

      return np.mean(mse),np.mean(cosine_sim)

def compute_reconstruction_errors_for_modality(input_path, result_path):  
        
        real_embeddings = [np.load(os.path.join(input_path, file)) for file in os.listdir(input_path)]
        imputed_embeddings = [np.load(os.path.join(result_path, file)) for file in os.listdir(result_path)]
        return compute_reconstruction_errors(real_embeddings, imputed_embeddings)

mse, cosine_sim = compute_reconstruction_errors_for_modality('./local/data/demo_office/test_emb_graphautoenc_visual', './local/data/demo_office/imputed_emb_vargraphautoenc_visual/vargraphautoenc')
print(f"Test result vargraphautoenc visual: mse={mse}, cosine similarity={cosine_sim}")
mse, cosine_sim = compute_reconstruction_errors_for_modality('./local/data/demo_office/test_emb_graphautoenc_textual', './local/data/demo_office/imputed_emb_vargraphautoenc_textual/vargraphautoenc')
print(f"Test result vargraphautoenc textual: mse={mse}, cosine similarity={cosine_sim}")

import torch
adjF = torch.load("adj_featprop.pt")
adjP = torch.load("adj_perspagerank.pt")
adjH = torch.load("adj_heat.pt")

print(adjF.to_dense()[:7, :7])

adjP_t = torch.from_numpy(adjP)
adjH_t = torch.from_numpy(adjH)

print(adjP_t.to_dense()[:7, :7])
print(adjH_t.to_dense()[:7, :7])
'''

'''
import pandas as pd
import numpy as np

input_file = "./Graph-Missing-Modalities/data/Office_Products/dropped/final_meta.tsv"
output_file = "./Graph-Missing-Modalities/data/Office_Products/dropped/dropped_meta.tsv"
missing_items_file = "./Graph-Missing-Modalities/data/Office_Products/dropped/missing_items_30.tsv"
drop_fraction = 0.3
random_seed = 42  # per riproducibilità

df = pd.read_csv(input_file, sep="\t", header=None)
np.random.seed(random_seed)
mask = np.random.rand(len(df)) < drop_fraction
df_removed = df[mask]
df_remaining = df[~mask]
df_remaining.to_csv(output_file, sep="\t", index=False, header=False)
df_removed.to_csv(missing_items_file, sep="\t", index=False, header=False)


import pandas as pd
reviews_file = "./Graph-Missing-Modalities/data/Office_Products/dropped/final_reviews.tsv"
missing_items_file = "./Graph-Missing-Modalities/data/Office_Products/dropped/missing_items_30.tsv"  # il file con gli item da rimuovere
output_file = "./Graph-Missing-Modalities/data/Office_Products/dropped/dropped_reviews.tsv"

df_reviews = pd.read_csv(reviews_file, sep="\t", header=None)
missing_items = pd.read_csv(missing_items_file, sep="\t", header=None)[0].tolist()
df_filtered = df_reviews[~df_reviews[1].isin(missing_items)]
df_filtered.to_csv(output_file, sep="\t", index=False, header=False)
'''
'''
import os
import pandas as pd

missing_items_file = "./Graph-Missing-Modalities/data/Office_Products/dropped/missing_items_30.tsv"
textual_dir = "./Graph-Missing-Modalities/data/Office_Products/dropped/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1"
visual_dir = "./Graph-Missing-Modalities/data/Office_Products/dropped/visual_embeddings"

missing_items = pd.read_csv(missing_items_file, sep="\t", header=None)[0].tolist()
missing_items_set = set(missing_items)  # più veloce per il lookup
def remove_missing_files(folder):
    removed_count = 0
    for filename in os.listdir(folder):
        if filename.endswith(".npy"):
            item_id = os.path.splitext(filename)[0]  # rimuove .npy
            if item_id in missing_items_set:
                file_path = os.path.join(folder, filename)
                os.remove(file_path)
                removed_count += 1
    return removed_count
removed_textual = remove_missing_files(textual_dir)
removed_visual = remove_missing_files(visual_dir)

print(f"File rimossi da '{textual_dir}': {removed_textual}")
print(f"File rimossi da '{visual_dir}': {removed_visual}")

'''

'''
import os
import numpy as np

cartella_F = "./local/data/demo_vbpr/test_imputed_visual_embeddings/featprop"
cartella_P = "./local/data/demo_vbpr/test_imputed_visual_embeddings/heat"

# prendo gli id comuni
ids = sorted(
    set(f[:-4] for f in os.listdir(cartella_F) if f.endswith(".npy"))
    & set(f[:-4] for f in os.listdir(cartella_P) if f.endswith(".npy"))
)
# solo i primi 10
#ids = ids[:10]

for id_ in ids:
    emb_F = np.load(os.path.join(cartella_F, f"{id_}.npy"))
    emb_P = np.load(os.path.join(cartella_P, f"{id_}.npy"))

    uguali = np.allclose(emb_F, emb_P, atol=1e-6)
    if uguali == False:
        print(f"{id_}: embedding uguali? {uguali}")
'''
'''
count = 0

with open("./local/data/demo_paper/sports/original_meta.tsv", "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("B00"):
            count += 1

print("Righe che iniziano con B00:", count)
'''
'''
import os

folder1 = "./local/data/demo_paper/sports/extracted/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1"
folder2 = "./local/data/demo_paper/sports/extracted/visual_embeddings/torch/ResNet50/avgpool"

files1 = set(os.listdir(folder1))
files2 = set(os.listdir(folder2))

solo_in_1 = files1 - files2
solo_in_2 = files2 - files1
in_entrambe = files1 & files2

print("\nFile solo in vis:")
for f in solo_in_2:
    print(f)

print("\nTotali:")
print("Solo text:", len(solo_in_1))
print("Solo vis:", len(solo_in_2))
print("In entrambe:", len(in_entrambe))
'''
'''
input_file = "./local/data/demo_paper/sports/meta.tsv"
output_file = "./local/data/demo_paper/sports/items.tsv"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        first_column = line.strip().split("\t")[0]
        if line.startswith("B00"):
            outfile.write(first_column + "\n")
'''

import os

cartella1 = "./local/data/demo_elliot/baby/imputed/visual_embeddings/mp10/zeros"
cartella2 = "./local/data/demo_paper/baby/imputed/visual_embeddings/mp10/zeros/0.1"

files1 = set(os.listdir(cartella1))
files2 = set(os.listdir(cartella2))

# File presenti in entrambe
comuni = files1.intersection(files2)
# File solo in cartella1
solo1 = files1 - files2
# File solo in cartella2
solo2 = files2 - files1
print("File in entrambe le cartelle:")
print(len(comuni))
print("\nFile solo in cartella1:")
print(len(solo1))
print("\nFile solo in cartella2:")
print(len(solo2))


#Vediamo se le ricostruzioni sono giuste -> Script o nuova funzionalità di Ducho (simula l'assenza di modalità) 
#Facciamo finta che certi item abbiano missing modalities, poi le ricostruiamo e poi vediamo la distanza, tramite mean square error vediamo se la ricostruzione è giusta.
import os
import numpy as np
import pandas as pd
from numpy.linalg import norm
from ducho.runner.Runner import MultimodalFeatureImputator
import shutil 

#questo script considera una sola modalità ('neighmean' 'featprop' 'perspagerank' 'heat')
#FUNZIONE COSINE SIMILARITY
def cosine_similarity(a, b):
    a = np.squeeze(a)
    b = np.squeeze(b)
    na = norm(a)
    nb = norm(b)
    if na == 0 or nb == 0:
        return None
    return np.dot(a, b) / (na * nb)

#CARICHIAMO LISTA ITEM
tsv_file = "local/data/demo_giulia/test/items.tsv"   #file tsv di item con tutte le modalità presenti
df = pd.read_csv(tsv_file, sep="\t")                 #lettura file di item
items = df["item_name"].tolist()                     #lista item

#CREIAMO DIZIONARIO ITEM-EMBEDDING CARICANDO EMBEDDING DEGLI ITEM
embeddings_folder = "local/data/demo_giulia/test/visual_embeddings"   #cartella embedding estratti visuali
#embeddings_folder = "local/data/demo_giulia/test/textual_embeddings"   #cartella embedding estratti testuali
X = {item: np.load(os.path.join(embeddings_folder, f"{item}.npy")) for item in items}  #dizionario {item: embedding}  cioè  {'item': array([[ 0, 1, 2, ...]], dtype=float32), ... }

#CREIAMO DIZIONARIO ITEM-EMBEDDING CON EMBEDDING MISSING ARTIFICIALI (MASCHERANDO con None)
#LOOP SU DIVERSI LIVELLI DI MISSING (mask_ratio)
mask_levels = [0.1, 0.5, 0.8]  
results = []                             

for mask_ratio in mask_levels:

    mask = np.random.rand(len(items)) < mask_ratio #crea mask
    if not mask.any():  # se nessun elemento è True, forza almeno uno
        idx = np.random.randint(len(items))  # scegli un indice a caso
        mask[idx] = True
    X_masked = {item: (None if mask[i] else X[item].copy()) for i, item in enumerate(items)}  #crea dizionario masked

    missing_file = 'local/data/demo_giulia/test/missing_file.tsv'  #crea missing_file per quel ratio
    with open(missing_file, "w") as f:
        for i, item in enumerate(items):
            if mask[i]:
                f.write(f"{item}\n")
    
    non_masked_embeddings_folder = "local/data/demo_giulia/test/non_masked_embeddings" #crea cartella con embedding non mascherati per quel ratio
    os.makedirs(non_masked_embeddings_folder, exist_ok=True)
    for i, item in enumerate(items):
        if not mask[i]:
            np.save(os.path.join(non_masked_embeddings_folder, f"{item}.npy"), X[item])

    imputator_obj = MultimodalFeatureImputator(config_file_path='./demos/demo_giulia/config_test.yml')  #imputazione
    imputator_obj.execute_imputations()
   
    #CREIAMO DIZIONARIO ITEM-EMBEDDING CON EMBEDDING MISSING IMPUTATI
    method_name = 'heat' #da cambiare ['neighmean' 'featprop' 'perspagerank' 'heat']
    imputed_embeddings_folder = "local/data/demo_giulia/test/imputed_embeddings/" + f"{method_name}"  #crea cartella con embedding IMPUTATI per quel ratio
    os.makedirs(imputed_embeddings_folder, exist_ok=True)

    X_imp = {}
    for item, emb in X_masked.items():
        if emb is None:
            X_imp[item] = np.load(os.path.join(imputed_embeddings_folder, f"{item}.npy"))
        else:
            X_imp[item] = emb.copy()

    #pulizia file 
    shutil.rmtree('local/data/demo_giulia/test/imputed_embeddings')
    shutil.rmtree('local/data/demo_giulia/test/non_masked_embeddings')
    shutil.rmtree('local/data/demo_giulia/test/non_masked_embeddings_indexed')
    os.remove('local/data/demo_giulia/test/IMPUTAZIONE.txt')
    os.remove('local/data/demo_giulia/test/missing_file.tsv')
    os.remove('local/data/demo_giulia/test/missing_visual_indexed.tsv')
    os.remove('local/data/demo_giulia/test/train_indexed.tsv')

    #CALCOLIAMO MSE E COSINE SIMILARITY TRA GLI EMBEDDING REALI E IMPUTATI DEGLI ITEM MASCHERATI
    mse_vals = []
    cs_vals = []
    for i, item in enumerate(items):
        if mask[i]:
            mse_vals.append(((X_imp[item] - X[item])**2).mean())
            cs_vals.append(cosine_similarity(X_imp[item], X[item]))
    mse_final = np.mean(mse_vals)
    cs_final = np.mean([c for c in cs_vals if c is not None])
    results.append((mask_ratio, mse_final, cs_final))

#SALVA RISULTATI FINALI
out_file = "local/data/demo_giulia/test/results_masking.tsv"
table_header = f"{'mask_ratio':<12} {'mse':<12} {'cosine_sim':<12}"
with open(out_file, "w") as f:
    f.write(f"method: {method_name}" + "\n")
    f.write(table_header + "\n")
    for mask_ratio, mse_final, cs_final in results:
        line = f"{mask_ratio:<12.2f} {mse_final:<12.4f} {cs_final:<12.4f}"
        f.write(line + "\n")
print("\nRisultati salvati in:", out_file)







'''
old 
#CARICHIAMO LISTA ITEM
tsv_file = "local/data/demo_giulia/test/items.tsv"   #file tsv di item con tutte le modalità presenti
df = pd.read_csv(tsv_file, sep="\t")                 #lettura file di item
items = df["item_name"].tolist()                     #lista item

#CREIAMO DIZIONARIO ITEM-EMBEDDING CARICANDO EMBEDDING DEGLI ITEM
embeddings_folder = "local/data/demo_giulia/test/visual_embeddings"   #cartella embedding estratti visuali
#embeddings_folder = "local/data/demo_giulia/test/textual_embeddings"   #cartella embedding estratti testuali
X = {item: np.load(os.path.join(embeddings_folder, f"{item}.npy")) for item in items}  #dizionario {item: embedding}  cioè  {'item': array([[ 0, 1, 2, ...]], dtype=float32), ... }

#CREIAMO DIZIONARIO ITEM-EMBEDDING CON EMBEDDING MISSING ARTIFICIALI (MASCHERANDO con None)
mask_ratio = 0.3
mask = np.random.rand(len(items)) < mask_ratio   #es.[False  True False  True False False  True False False False] -> item 2 4 7 avranno embedding mancante
X_masked = {}
for i, item in enumerate(items):
    if mask[i]:
        X_masked[item] = None             #dizionario masked avrà {'item': None }
    else:
        X_masked[item] = X[item].copy()   #dizionario masked avrà {'item': array([[ 0, 1, 2, ...]], dtype=float32), ... }

#CREARIAMO FILE CON ITEM MASCHERATI (missing_file)
missing_file = "local/data/demo_giulia/test/missing_file.tsv"      #file tsv con item con modalità mancante (mascherata)
masked_items = [item for i, item in enumerate(items) if mask[i]]
with open(missing_file, "w") as f:
    for item in masked_items:
        f.write(f"{item}\n")

#CREIAMO CARTELLA CON EMBEDDING NON MASCHERATI
non_masked_embeddings_folder = "local/data/demo_giulia/test/non_masked_embeddings"   #cartella con embedding utili all'imputazione (input_path)
os.makedirs(non_masked_embeddings_folder, exist_ok=True)
for i, item in enumerate(items):
    if not mask[i]:  # item NON mascherato
        out_path = os.path.join(non_masked_embeddings_folder, f"{item}.npy")
        np.save(out_path, X[item])  # salva embedding originale

#IMPUTIAMO GLI EMBEDDING PER GLI ITEM MASCHERATI (con i metodi del run.py)
imputator_obj = MultimodalFeatureImputator(config_file_path='./demos/demo_giulia/config_test.yml')
imputator_obj.execute_imputations() 

#CREIAMO DIZIONARIO ITEM-EMBEDDING CON EMBEDDING MISSING IMPUTATI
imputed_embeddings_folder = "local/data/demo_giulia/test/imputed_embeddings/neighmean/top_k_3"   #
X_imp = {}
for item, emb in X_masked.items():
        if emb is None:  # CARICARE embedding imputato da file
            imp_file = os.path.join(imputed_embeddings_folder, f"{item}.npy")
            X_imp[item] = np.load(imp_file)
        else:
            X_imp[item] = emb.copy()  # Item NON mascherato (copiamo embedding originale)

#CALCOLIAMO MSE E COSINE SIMILARITY TRA GLI EMBEDDING REALI E IMPUTATI DEGLI ITEM MASCHERATI
mse = [ ((X_imp[item] - X[item])**2).mean() for i, item in enumerate(items) if mask[i] ] 
print("MSE:", np.mean(mse)) 

def cosine_similarity(a, b):
    a = np.squeeze(a)
    b = np.squeeze(b)
    na = norm(a)
    nb = norm(b)
    if na == 0 or nb == 0: # emb nullo → no
        return 'no'  
    return np.dot(a, b) / (na * nb)

cosine_scores = [cosine_similarity(X_imp[item], X[item]) for i, item in enumerate(items) if mask[i]]
print("CS:", np.mean(cosine_scores))

#provare con la percentuale
#NEIGHMEAN TESTUALE FUNZIONA :) (TOPK=3)   MSE: 0.0012422447  CS: 0.35045633
#NEIGHMEAN VISUAL FUNZIONA :)   (TOPK=3)  MSE: 0.097015545 CS: 0.2668573
'''
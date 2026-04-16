#Vediamo se le ricostruzioni sono giuste -> Script o nuova funzionalità di Ducho (simula l'assenza di modalità) 
#Facciamo finta che certi item abbiano missing modalities, poi le ricostruiamo e poi vediamo la distanza, tramite mean square error vediamo se la ricostruzione è giusta.
import os
import numpy as np
import pandas as pd
from numpy.linalg import norm
from ducho.runner.Runner import MultimodalFeatureImputator
import shutil 

#lo script considera una sola modalità ('neighmean' 'featprop' 'perspagerank' 'heat')
#oppure considera 2 modalità ('autoenc', 'graphautoenc', 'vigan', 'smil', 'mmin')

#FUNZIONE COSINE SIMILARITY
def cosine_similarity(a, b): 
    a = np.squeeze(a)
    b = np.squeeze(b)
    na = norm(a)
    nb = norm(b)
    if na == 0 or nb == 0:
        return None
    sim = np.dot(a, b) / (na * nb)
    scaled_sim = (sim + 1) / 2   # porta da [-1,1] a [0,1]
    return scaled_sim

#CARICHIAMO LISTA ITEM
tsv_file = "local/data/demo_giulia/test/items.tsv"   #file tsv di item con tutte le modalità presenti
df = pd.read_csv(tsv_file, sep="\t")                 #lettura file di item
items = df["item_name"].tolist()                     #lista item

#CREIAMO DIZIONARIO ITEM-EMBEDDINGS CARICANDO EMBEDDINGS DEGLI ITEM
visual_folder = "local/data/demo_giulia/test/visual_embeddings"  #cartella embedding estratti visuali
X_visual = {item: np.load(os.path.join(visual_folder, f"{item}.npy")) for item in items}
textual_folder = "local/data/demo_giulia/test/textual_embeddings"   #cartella embedding estratti testuali
X_textual = {item: np.load(os.path.join(textual_folder, f"{item}.npy")) for item in items}
X = {item: {"visual": X_visual[item], "textual": X_textual[item]} for item in items}  #dizionario {'item': {'visual': visual_embedding, 'textual': textual_embedding}}  

#CREIAMO DIZIONARIO ITEM-EMBEDDINGS CON EMBEDDINGS MISSING ARTIFICIALI (MASCHERANDO con None)
#LOOP SU DIVERSI LIVELLI DI MISSING (mask_ratio)
mask_levels_visual  = [0.1, 0.5, 0.8]  # % missingness non unica per entrambe le modalità, ma singola per ciascuna di esse
mask_levels_textual = [0.2, 0.6, 0.9]  
results = []

for mask_ratio_visual, mask_ratio_textual in zip(mask_levels_visual, mask_levels_textual):

    mask_visual  = np.random.rand(len(items)) < mask_ratio_visual #crea mask visual per gli item
    mask_textual = np.random.rand(len(items)) < mask_ratio_textual  #crea mask testuale per gli item
    if not mask_visual.any():   
        mask_visual[np.random.randint(len(items))] = True  # Garantisce almeno un masked per modalità
    if not mask_textual.any():
        mask_textual[np.random.randint(len(items))] = True

    X_masked = { item: {"visual": None if mask_visual[i] else X[item]["visual"].copy(),
                        "textual": None if mask_textual[i] else X[item]["textual"].copy()}
                 for i, item in enumerate(items)}   #crea dizionario masked 

    missing_visual_file = 'local/data/demo_giulia/test/missing_visual.tsv'  #crea missing_visual_file per quel ratio
    with open(missing_visual_file, "w") as f:
        for i, item in enumerate(items):
            if mask_visual[i]:
                f.write(f"{item}\n")

    missing_textual_file = 'local/data/demo_giulia/test/missing_textual.tsv'  #crea missing_textual_file per quel ratio
    with open(missing_textual_file, "w") as f:
        for i, item in enumerate(items):
            if mask_textual[i]:
                f.write(f"{item}\n")
    
    non_masked_visual_embeddings_folder = "local/data/demo_giulia/test/non_masked_visual_embeddings" #crea cartella con visual embedding non mascherati per quel ratio
    os.makedirs(non_masked_visual_embeddings_folder, exist_ok=True)
    for i, item in enumerate(items):
        if not mask_visual[i]:
            np.save(os.path.join(non_masked_visual_embeddings_folder, f"{item}.npy"), X[item]["visual"])

    non_masked_textual_embeddings_folder = "local/data/demo_giulia/test/non_masked_textual_embeddings" #crea cartella con textual embedding non mascherati per quel ratio
    os.makedirs(non_masked_textual_embeddings_folder, exist_ok=True)
    for i, item in enumerate(items):
        if not mask_textual[i]:
            np.save(os.path.join(non_masked_textual_embeddings_folder, f"{item}.npy"), X[item]["textual"])

    imputator_obj = MultimodalFeatureImputator(config_file_path='./demos/demo_giulia/config_test.yml') #imputazione
    imputator_obj.execute_imputations()

    #CREIAMO DIZIONARIO ITEM-EMBEDDINGS CON EMBEDDINGS MISSING IMPUTATI
    method_name = "autoenc" #da cambiare ['neighmean', 'featprop', 'perspagerank', 'heat', 'autoenc', 'graphautoenc', 'vigan', 'smil', 'mmin']
    imputed_visual_embeddings_folder = "local/data/demo_giulia/test/imputed/visual_embeddings/" + f"{method_name}" #crea cartella con visual embedding IMPUTATI per quel ratio
    os.makedirs(imputed_visual_embeddings_folder, exist_ok=True)
    imputed_textual_embeddings_folder = "local/data/demo_giulia/test/imputed/textual_embeddings/" + f"{method_name}" #crea cartella con textual embedding IMPUTATI per quel ratio
    os.makedirs(imputed_textual_embeddings_folder, exist_ok=True)
    
    X_imp = {}
    for item in items:
        X_imp[item] = {"visual": X[item]["visual"] if X_masked[item]["visual"] is not None else np.load(os.path.join(imputed_visual_embeddings_folder, f"{item}.npy")),
                       "textual": X[item]["textual"] if X_masked[item]["textual"] is not None else np.load(os.path.join(imputed_textual_embeddings_folder, f"{item}.npy"))}

    #pulizia file
    shutil.rmtree('local/data/demo_giulia/test/imputed')
    shutil.rmtree('local/data/demo_giulia/test/non_masked_visual_embeddings')
    shutil.rmtree('local/data/demo_giulia/test/non_masked_textual_embeddings')
    os.remove('local/data/demo_giulia/test/IMPUTAZIONE.txt')
    os.remove('local/data/demo_giulia/test/missing_visual.tsv')
    os.remove('local/data/demo_giulia/test/missing_textual.tsv')
    if method_name in ['neighmean', 'featprop', 'perspagerank', 'heat', 'graphautoenc']:
            shutil.rmtree('local/data/demo_giulia/test/non_masked_visual_embeddings_indexed')
            shutil.rmtree('local/data/demo_giulia/test/non_masked_textual_embeddings_indexed')
            os.remove('local/data/demo_giulia/test/missing_visual_indexed.tsv')
            os.remove('local/data/demo_giulia/test/missing_textual_indexed.tsv')
            os.remove('local/data/demo_giulia/test/train_indexed.tsv')

    #CALCOLIAMO MSE E COSINE SIMILARITY TRA GLI EMBEDDING REALI E IMPUTATI DEGLI ITEM MASCHERATI PER OGNI MODALITÀ
    #metriche mediate sugli embedding missing artificiali, non su tutti
    mse_vis = []
    cs_vis = []
    mse_txt = []
    cs_txt = []
    for i, item in enumerate(items):

        if mask_visual[i]:
            real = X[item]["visual"]
            imp  = X_imp[item]["visual"]
            mse_vis.append(((imp - real)**2).mean())
            cs_vis.append(cosine_similarity(imp, real))
        
        if mask_textual[i]:  
           real = X[item]["textual"]
           imp  = X_imp[item]["textual"]
           mse_txt.append(((imp - real)**2).mean())
           cs_txt.append(cosine_similarity(imp, real))
    
    results.append(mask_ratio_visual, mask_ratio_textual, np.mean(mse_vis) if mse_vis else None,  np.mean([c for c in cs_vis if c is not None]) if cs_vis else None,
                                                          np.mean(mse_txt) if mse_txt else None, np.mean([c for c in cs_txt if c is not None]) if cs_txt else None)

#SALVA RISULTATI FINALI
out_file = f"local/data/demo_giulia/test/results_masking_{method_name}.tsv"
with open(out_file, "w") as f:
    f.write(f"method: {method_name}\n")
    f.write(f"{'mask_visual_ratio':<10} {'mask_textual_ratio':<10} {'mse_vis':<12} {'cs_vis':<12} {'mse_txt':<12} {'cs_txt':<12}\n")
    # Ogni elemento in results è: (mask_visual_ratio, mask_textual_ratio, mse_vis, cs_vis, mse_txt, cs_txt)
    for mask_visual_ratio, mask_textual_ratio, mse_v, cs_v, mse_t, cs_t in results:
        line = (f"{mask_visual_ratio:<10.2f} {mask_textual_ratio:<10.2f} {mse_v:<12.4f} {cs_v:<12.4f} {mse_t:<12.4f} {cs_t:<12.4f}")
        f.write(line + "\n")
print("\nRisultati salvati in:", out_file)
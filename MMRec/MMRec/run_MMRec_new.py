import os
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" #for reproducibility

import numpy as np
import pandas as pd
import argparse
from utils.quick_start import quick_start

parser = argparse.ArgumentParser(description="Prepare for MMRec.")
parser.add_argument('--data', type=str, default='Baby')
parser.add_argument('--model', type=str, default='LGMRec')
parser.add_argument('--method', type=str, default='random')
parser.add_argument('--mp', type=int, default=10, help='missing percentage (10, 50, 90)')
args = parser.parse_args()

root = f'./data/{args.data}'

train = pd.read_csv(f'{root}/train_final_indexed.tsv', sep='\t', header=None)
val = pd.read_csv(f'{root}/val_final_indexed.tsv', sep='\t', header=None)
test = pd.read_csv(f'{root}/test_final_indexed.tsv', sep='\t', header=None)
df = pd.concat([train, val, test])

num_items = df[1].nunique()

visual_embeddings = np.empty((num_items, 2048))
textual_embeddings = np.empty((num_items, 768))

visual_dir = f'{root}/visual_embeddings_complete_indexed/mp{args.mp}/{args.method}'
textual_dir = f'{root}/textual_embeddings_complete_indexed/mp{args.mp}/{args.method}'

for it in range(num_items):
    visual_embeddings[it] = np.load(f'{visual_dir}/{it}.npy')
    textual_embeddings[it] = np.load(f'{textual_dir}/{it}.npy')

config_dict = {
    'gpu_id': 0,
}

quick_start(model=args.model, dataset=args.data, config_dict=config_dict, save_model=True,
            visual_embeddings=visual_embeddings, textual_embeddings=textual_embeddings,
            extractor='RNet50+SBert')

#python run_MMRec_new.py --data Baby --model LGMRec --method zeros --mp 10
#python run_MMRec_new.py --data Baby --model PGL --method zeros --mp 10

#python run_MMRec_new.py --data Beauty --model LGMRec --method zeros --mp 10
#python run_MMRec_new.py --data Beauty --model PGL --method zeros --mp 10

#python run_MMRec_new.py --data Office --model LGMRec --method zeros --mp 10
#python run_MMRec_new.py --data Office --model PGL --method zeros --mp 10

#python run_MMRec_new.py --data Sports --model LGMRec --method zeros --mp 10
#python run_MMRec_new.py --data Sports --model PGL --method zeros --mp 10

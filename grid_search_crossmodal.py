import numpy as np
import pandas as pd
import os
from itertools import product
from loguru import logger

from ducho.imputation.graphautoenc import GraphautoencImputator
from ducho.imputation.vargraphautoenc import VargraphautoencImputator


class GridSearchCrossModalImputator:

    def __init__(self, dataset_path, input_paths, output_paths, train_path, seed=42):
        
        self.dataset_path = dataset_path
        self.input_paths = input_paths
        self.output_paths = output_paths
        self.train_path = train_path
        self.seed = seed

        self.test_percentage = 0.30 # 30% rimosso come test finale
        self.validation_percentage = [0.2]  # 20% del rimanente per validation
        self.results = []

        # Definizione dei range per ogni metodo
        self.param_grids = {"graphautoenc": {"top_k": list(range(20, 31, 10))},    # {"top_k": list(range(10, 101, 10))}, 
                            "vargraphautoenc": {"top_k": list(range(20, 31, 10))}}  # {"top_k": list(range(10, 101, 10))}, 
        # Mapping tra nome metodo e classe imputator
        self.imputator_classes = {"graphautoenc": GraphautoencImputator, "vargraphautoenc": VargraphautoencImputator}

   
    def _load_items_list(self, items_file="items.tsv"):  #Carica la lista completa degli items dal file.
        with open(os.path.join(self.dataset_path, items_file), "r", encoding="utf-8") as f:
            return [line.strip() for line in f]

    def _split_items_for_test(self, items_file="items.tsv"):
        items = self._load_items_list(items_file)
        np.random.seed(self.seed)
        shuffled = np.random.permutation(items)
        # 30% per test finale
        n_test = int(len(items) * self.test_percentage)
        test_items = shuffled[:n_test].tolist()
        train_val_items = shuffled[n_test:].tolist()

        logger.info(f"Split completato:")
        logger.info(f"  - Test items (30%): {len(test_items)}")
        logger.info(f"  - Train/Val items (70%): {len(train_val_items)}")
        return test_items, train_val_items

    def _generate_param_combinations(self, method_name):   # Genera tutte le combinazioni di parametri per un metodo specifico.
        
        grid = self.param_grids[method_name]
        keys = list(grid.keys())
        values = list(grid.values())
        return [dict(zip(keys, v)) for v in product(*values)]

    def _run_single_configuration(self, method_name, params):
       
        # Ottieni la classe imputator appropriata
        ImputatorClass = self.imputator_classes[method_name]
        # Crea l'istanza dell'imputator con i parametri
        imputer = ImputatorClass(dataset_path=self.dataset_path, missing_visual_file=None, missing_textual_file=None, input_paths=self.input_paths, output_paths=self.output_paths,
                                 train_path=self.train_path, top_k=params["top_k"], seed=self.seed, missing_percentage_visual=self.validation_percentage, missing_percentage_textual=self.validation_percentage, items_file="items.tsv")
        errors = imputer.impute()
        # errors = {
        #   'visual': [(percentage, mse, cosine)],
        #   'textual': [(percentage, mse, cosine)]
        # }
        (_, mse_visual, cos_visual) = errors["visual"][0]
        (_, mse_textual, cos_textual) = errors["textual"][0]
        mse_mean = (mse_visual + mse_textual) / 2
        cosine_mean = (cos_visual + cos_textual) / 2

        return {"mse_visual": mse_visual, "mse_textual": mse_textual, "mse_mean": mse_mean,
                "cosine_visual": cos_visual, "cosine_textual": cos_textual, "cosine_mean": cosine_mean}

    def search_best_params(self, method_name, metric="mse_mean", items_file="items.tsv"):  #Esegue la grid search per un metodo specifico.

        logger.info(f"GRID SEARCH per metodo: {method_name.upper()}")
        # Crea split di test
        test_items, _ = self._split_items_for_test(items_file)
        # Genera tutte le combinazioni di parametri
        param_combinations = self._generate_param_combinations(method_name)

        best_score = float("inf")
        best_params = None
        best_metrics = None
        
        # Testa ogni combinazione
        for i, params in enumerate(param_combinations, 1):
            logger.info(f"[{i}/{len(param_combinations)}] Test: {params}")
            try:
                metrics = self._run_single_configuration(method_name, params)
                score = metrics[metric]

                self.results.append({"method": method_name, **params, **metrics})
                logger.info(f"MSE mean: {metrics['mse_mean']:.6f}")
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    best_metrics = metrics.copy()
                    logger.success("Nuovo miglior setting trovato!")
            except Exception as e:
                logger.error(f"Errore: {e}")
                import traceback
                traceback.print_exc()

        logger.success(f"MIGLIOR CONFIGURAZIONE per {method_name}")
        logger.info(f"Parametri: {best_params}")
        logger.info(f"MSE VISUAL: {best_metrics['mse_visual']:.6f}")
        logger.info(f"Cosine Similarity VISUAL: {best_metrics['cosine_visual']:.6f}")
        logger.info(f"MSE TEXTUAL: {best_metrics['mse_textual']:.6f}")
        logger.info(f"Cosine Similarity TEXTUAL: {best_metrics['cosine_textual']:.6f}")
        
        return best_params, best_metrics, test_items

    def get_results_dataframe(self) -> pd.DataFrame:   #Converte i risultati in un DataFrame pandas per analisi.
        df = pd.DataFrame(self.results)
        return df.sort_values("mse_mean")

    def print_top_configurations(self, n: int = 10): #Stampa le top N configurazioni per MSE mean.
        df = self.get_results_dataframe()
        logger.info(f"TOP {n} CONFIGURAZIONI (ordinate per MSE mean)")
        print(df.head(n).to_string(index=False))
        logger.info("")






def execute_grid_search_crossmodal(method_name):  #Esegue grid search per trovare i parametri ottimali per i metodi d'imputazione crossmodali a grafo
    gs = GridSearchCrossModalImputator(dataset_path="./local/data/demo_office",
                                       input_paths=["./local/data/demo_office/extracted/visual_embeddings/torch/ResNet50/avgpool", "./local/data/demo_office/extracted/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1"],
                                       output_paths=["./local/data/demo_office/imputed/crossmodal/visual_embeddings","./local/data/demo_office/imputed/crossmodal/textual_embeddings"],
                                       train_path="train.tsv", seed=42)
    best_params, best_metrics, test_items = gs.search_best_params(method_name=method_name, metric="mse_mean")
    gs.print_top_configurations(10)
    with open(f"./local/data/demo_office/test_items_{method_name}.tsv", "w") as f:      #eccolooooooooooooo
        for item in test_items:
            f.write(f"{item}\n")
    return best_params, best_metrics

best_params, best_metrics = execute_grid_search_crossmodal("vargraphautoenc")
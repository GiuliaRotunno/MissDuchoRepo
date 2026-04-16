import numpy as np
import pandas as pd
from itertools import product
from typing import Dict, List, Tuple, Any
import os
from loguru import logger

from ducho.imputation.neighmean import NeighmeanImputator
from ducho.imputation.featprop import FeatpropImputator
from ducho.imputation.perspagerank import PerspagerankImputator
from ducho.imputation.heat import HeatImputator

# Rimuove 30% degli item come test set finale.
# Sul rimanente 70% simula missingness del 20% per la validation.
class GridSearchImputator:
       
    def __init__(self, dataset_path, input_paths, output_paths, train_path, modality, seed=42):
      
        self.dataset_path = dataset_path
        self.input_paths = input_paths
        self.output_paths = output_paths
        self.train_path = train_path             
        self.modality = modality
        self.seed = seed
        self.results = []
        
        self.test_percentage = 0.30  # 30% rimosso come test finale
        self.validation_percentage = [0.2]  # 20% del rimanente per validation
        
        # Definizione dei range per ogni metodo
        self.param_grids = {'neighmean': {'top_k': list(range(10, 101, 10))},
                            'featprop': {'top_k': list(range(10, 101, 10)), 'layers': list(range(1, 21, 1))},
                            'perspagerank': {'top_k': list(range(10, 101, 10)),'layers': list(range(1, 21, 1)),'alpha': [0.1]},
                            'heat': {'top_k': list(range(10, 101, 10)),'layers': list(range(1, 21, 1)), 'time': [5.0]}}
        # Mapping tra nome metodo e classe imputator
        self.imputator_classes = {'neighmean': NeighmeanImputator, 'featprop': FeatpropImputator, 'perspagerank': PerspagerankImputator,'heat': HeatImputator}
    
    def _create_items_file(self, items_list, percentage, method_name, suffix='validation'):  # Crea un file temporaneo con la lista degli items missing.
   
        filename = f"artificial_missing_file_{percentage}_{method_name}_{suffix}.tsv"  #suffix = validation/test
        filepath = os.path.join(self.dataset_path, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in items_list:
                f.write(f"{item}\n")
        return filename
    
    def _load_items_list(self, items_file='items.tsv'):  #Carica la lista completa degli items dal file.
    
        filepath = os.path.join(self.dataset_path, items_file)
        items = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                items.append(line.strip())
        return items
    
    def _split_items_for_test(self, items_file='items.tsv'):  

        all_items = self._load_items_list(items_file)
        n_items = len(all_items)
        np.random.seed(self.seed)
        shuffled_items = np.random.permutation(all_items)
        # 30% per test finale
        n_test = int(n_items * self.test_percentage)
        test_items = shuffled_items[:n_test].tolist()
        train_val_items = shuffled_items[n_test:].tolist()

        logger.info(f"Split completato:")
        logger.info(f"  - Test items (30%): {len(test_items)}")
        logger.info(f"  - Train_val items (70%): {len(train_val_items)}")
        return test_items, train_val_items
    
    def _generate_param_combinations(self, method_name: str) -> List[Dict[str, Any]]:  # Genera tutte le combinazioni di parametri per un metodo specifico.
        
        param_grid = self.param_grids[method_name]
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = []
        for values in product(*param_values):
            param_dict = dict(zip(param_names, values))
            combinations.append(param_dict)
        return combinations
    
    def _run_single_configuration(self, method_name: str, params: Dict[str, Any]) -> Dict[str, float]:  #Esegue l'imputazione con una specifica configurazione di parametri.
        
        # Ottieni la classe imputator appropriata
        imputator_class = self.imputator_classes[method_name]
        # Crea l'istanza dell'imputator con i parametri
        if method_name == 'neighmean':
            imputer = imputator_class(dataset_path=self.dataset_path, missing_file=None, input_paths=self.input_paths, output_paths=self.output_paths, train_path=self.train_path,
                                      top_k=params['top_k'], modality=self.modality, missing_percentage=self.validation_percentage, items_file='items.tsv', seed=self.seed)
        elif method_name == 'featprop':
            imputer = imputator_class(dataset_path=self.dataset_path, missing_file=None, input_paths=self.input_paths, output_paths=self.output_paths, train_path=self.train_path,
                                      top_k=params['top_k'], layers=params['layers'], modality=self.modality, missing_percentage=self.validation_percentage, items_file='items.tsv', seed=self.seed)
        elif method_name == 'perspagerank':
            imputer = imputator_class(dataset_path=self.dataset_path, missing_file=None, input_paths=self.input_paths, output_paths=self.output_paths, train_path=self.train_path,
                                      top_k=params['top_k'], layers=params['layers'], alpha=params['alpha'], modality=self.modality, missing_percentage=self.validation_percentage, items_file='items.tsv', seed=self.seed)
        elif method_name == 'heat':
            imputer = imputator_class(dataset_path=self.dataset_path, missing_file=None, input_paths=self.input_paths, output_paths=self.output_paths, train_path=self.train_path,
                                      top_k=params['top_k'], layers=params['layers'], time=params['time'], modality=self.modality, missing_percentage=self.validation_percentage, items_file='items.tsv', seed=self.seed)
        
        # Esegui imputazione e ottieni errori
        errors = imputer.impute()
        # Gli errori sono nel formato {percentage: (mse, cosine_sim)}, prendiamo il primo (e unico) valore dal dizionario
        percentage, (mse, cosine_sim) = list(errors.items())[0]
        return {'mse': mse, 'cosine_similarity': cosine_sim}
    
    def search_best_params(self, method_name: str, metric: str = 'mse', items_file: str = 'items.tsv') -> Tuple[Dict[str, Any], Dict[str, float]]:   #Esegue la grid search per un metodo specifico.
        
        logger.info(f"GRID SEARCH per metodo: {method_name.upper()}")    
        # Crea split di test
        test_items, _ = self._split_items_for_test(items_file)
        # Genera tutte le combinazioni di parametri
        param_combinations = self._generate_param_combinations(method_name)
        logger.info(f"Numero totale di configurazioni da testare: {len(param_combinations)}")
        
        best_params = None
        best_metrics = None
        best_score = float('inf') if metric == 'mse' else float('-inf')
        
        # Testa ogni combinazione
        for i, params in enumerate(param_combinations, 1):
            logger.info(f"\n[{i}/{len(param_combinations)}] Testando: {params}")
            try:
                # Esegui imputazione e ottieni metriche
                metrics = self._run_single_configuration(method_name, params)
                current_score = metrics[metric]
                logger.info(f"  MSE: {metrics['mse']:.6f}")
                logger.info(f"  Cosine Similarity: {metrics['cosine_similarity']:.6f}")
                # Salva risultato
                self.results.append({'method': method_name, 'params': params.copy(), 'metrics': metrics.copy()})
                # Aggiorna il migliore se necessario
                is_better = (current_score < best_score) if metric == 'mse' else (current_score > best_score)
                if is_better:
                    best_score = current_score
                    best_params = params.copy()
                    best_metrics = metrics.copy()
                    logger.success(f"Nuovo miglior setting trovato!")
            except Exception as e:
                logger.error(f"Errore con questa configurazione: {e}")
                import traceback
                traceback.print_exc()
                continue
    
        logger.success(f"MIGLIOR CONFIGURAZIONE per {method_name}:")
        logger.info(f"Parametri: {best_params}")
        logger.info(f"MSE: {best_metrics['mse']:.6f}")
        logger.info(f"Cosine Similarity: {best_metrics['cosine_similarity']:.6f}")
        
        return best_params, best_metrics, test_items
    
    def get_results_dataframe(self) -> pd.DataFrame:   #Converte i risultati in un DataFrame pandas per analisi.
      
        rows = []
        for result in self.results:
            row = {
                'method': result['method'],
                **result['params'],
                'mse': result['metrics']['mse'],
                'cosine_similarity': result['metrics']['cosine_similarity']
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        return df.sort_values('mse')
    
    def print_top_configurations(self, n: int = 10):   #Stampa le top N configurazioni per MSE.
      
        df = self.get_results_dataframe()
        logger.info(f"TOP {n} CONFIGURAZIONI (ordinate per MSE)")
        print(df.head(n).to_string(index=False))
        logger.info("")




def execute_grid_search(method_name, modality):  #Esegue grid search per trovare i parametri ottimali per i metodi d'imputazione a grafo
        
        logger.info(f'Avvio Grid Search per {method_name} su {modality}')
        gs = GridSearchImputator(dataset_path='./local/data/demo_office', input_paths='./local/data/demo_office/extracted/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1', output_paths='./local/data/demo_office/imputed/unimodal/textual_embeddings', train_path='train.tsv', modality=modality, seed=42)
        best_params, best_metrics, test_items = gs.search_best_params(method_name, metric='mse', items_file='items.tsv')
        gs.print_top_configurations(10)
        return best_params, best_metrics, test_items
    

best_params, best_metrics, test_items = execute_grid_search(method_name='heat', modality='textual')
print(best_params)
print(best_metrics)
filepath="./local/data/demo_office/test_items_heat_textual.tsv"
with open(filepath, "w", encoding="utf-8") as f:
    for item in test_items:
        f.write(f"{item}\n")

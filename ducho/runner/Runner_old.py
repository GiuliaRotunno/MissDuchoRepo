import os 
import time
# Hiding TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# Hiding Transformers' tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from loguru import logger
from alive_progress import alive_bar
import torch
import platform
import tensorflow as tf
import importlib
import datetime
import multiprocessing
from ducho.config.Config import Config
from ducho.multimodal.visual.VisualDataset import VisualDataset
from ducho.multimodal.multiple.visual_textual.VisualTextualDataset import VisualTextualDataset
from ducho.internal.utils.json2dotnotation import banner
#importare gli imputatori
from ducho.imputation.zeros import ZerosImputator
from ducho.imputation.random import RandomImputator
from ducho.imputation.mean import MeanImputator
from ducho.imputation.neighmean import NeighmeanImputator
from ducho.imputation.perspagerank import PerspagerankImputator
from ducho.imputation.heat import HeatImputator
from ducho.imputation.featprop import FeatpropImputator
from ducho.imputation.autoenc import AutoencImputator
from ducho.imputation.graphautoenc import GraphautoencImputator
from ducho.imputation.vigan_imputation.vigan_imputator import ViganImputator 


def camel_case(s):
    return ''.join(x for x in s.title() if not x.isspace()).replace('_', '')


def _execute_extraction_from_models_list(models, extractor_class, gpu, dataset):
    """
    Execute the extraction process for each model and each layer of the model.
    It also displays the extraction graphically.

    Args:
        models (List[Dict[str, Any]]): List of dictionaries containing model data (see Config.get_models).
        extractor_class (Type[Extractor]): Class representing the extractor.
        gpu (str): List of GPU identifiers.
        dataset (Type[Dataset]): Class representing the dataset.

    Return None
    """
    change_mean_std = False

    for model in models:
        logger.info(f'Extraction model: {model["model_name"]}')

        if 'fusion' in model.keys():
            if type(model['fusion']) == list:
                raise ValueError(f'Fusion field cannot be a list!')
            if model['fusion'] not in ['sum', 'concat', 'mul', 'mean']:
                raise NotImplementedError(
                    f'Fusion {model["fusion"]} is not implemented yet! Please select among: ["sum", "concat", "mul", "mean"]')

        extractor = extractor_class(gpu=gpu)

        # set framework
        extractor.set_framework(model['backend'])
        dataset.set_framework(model['backend'])

        # set model
        extractor.set_model(model)
        dataset.set_model(model['model_name'])

        # set preprocessing flag
        dataset.set_preprocessing_flag(model['preprocessing_flag'])
        if isinstance(dataset, VisualDataset):
            if 'preprocessing' in model.keys():  # preprocessing
                if not model['preprocessing'] in ['zscore', 'minmax', None]:
                    raise ValueError("Normalization must be 'minmax', 'zscore' or 'None'")
                dataset.set_preprocessing_type(model['preprocessing'])

            if 'mean' in model.keys() and 'std' in model.keys():
                if 'preprocessing' in model.keys():
                    if model['preprocessing'] == 'zscore':
                        dataset.set_mean_std(mean=model['mean'],
                                             std=model['std']
                                             )
                        change_mean_std = True
                    else:
                        raise ValueError("Mean and std values can be set only for zscore normalization!")
                else:
                    raise ValueError(
                        "Mean and std values can be set only for zscore normalization but by the default normalization is None. Please, specify if you want zscore normalization!")

        if isinstance(dataset, VisualTextualDataset):
            dataset.set_model_name(model['model_name'])

        # assess if the extractor is visual or visual-textual in order to properly set the dataset's image processor. 
        if (isinstance(dataset, VisualDataset) or isinstance(dataset, VisualTextualDataset)) and 'transformers' in model['backend']:
            dataset.set_image_processor(extractor._image_processor)

        # execute extractions
        for model_layer in model['output_layers']:
            logger.info(f'Extraction layer: {model["model_name"]}.{model_layer}')

            # set output layer
            extractor.set_output_layer(model_layer)

            start_time = time.time()

            if 'tensorflow' not in model['backend']:
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=int(model['batch_size']) if 'batch_size' in model.keys() else 1,
                    shuffle=False,
                    sampler=None,
                    num_workers=multiprocessing.cpu_count(),
                    pin_memory=True
                    )
                
                with alive_bar(len(dataloader)) as t:
                    # for evey item do the extraction
                    for index, batch in enumerate(dataloader):
                        # do the extraction
                        extractor_output = extractor.extract_feature(batch)
                        # create the npy file with the extraction output
                        dataset.create_output_file(
                            batch, 
                            extractor_output, 
                            model_layer,
                            fusion=model['fusion'] if 'fusion' in model.keys() else None
                            )
                        # update the progress bar
                        t()
            else:
                with alive_bar(total=dataset.__len__()) as t:
                    # for evey item do the extraction                    
                    for index in range(dataset.__len__()):
                        # retrieve the item (preprocessed) from dataset
                        batch = dataset.__getitem__(index)
                        # do the extraction
                        extractor_output = extractor.extract_feature(batch)
                        # create the npy file with the extraction output
                        dataset.create_output_file(
                            batch, 
                            extractor_output, 
                            model_layer,
                            fusion=model['fusion'] if 'fusion' in model.keys() else None
                            )
                        # update the progress bar
                        t()

            if change_mean_std:
                dataset._reset_mean_std()
                change_mean_std = False

            end_time = time.time()
            logger.success(f'Extraction with layer: {model["model_name"]}.{model_layer} has been completed in {round(end_time - start_time, 2)} seconds!')

        logger.success(f'Extraction with model: {model["model_name"]} is complete')
        del extractor
        torch.cuda.empty_cache()

def _apply_imputation_methods(imputation_methods, dataset_path, output_paths, models, modality):  #new method
    """
    Apply imputation methods to missing data.
    Args: imputation_methods (List[Dict]): List of imputation method configurations.
          dataset_path (str): Base path to the dataset.
          output_path (list): Path to saved embeddings.
    Returns: None
    """
    if not imputation_methods:  #non fa nessuna operazione se non c'è imputation:[] nel file config.yml
        return 
    logger.info(f"Applying {len(imputation_methods)} imputation method(s)")

    for model in models:
        model_name = model['model_name']
        print(f'------modelname-------', model_name)

        if type(output_paths) == list:
           output_path_visual = os.path.join(output_paths[0], model['backend'][0], model_name, str(model['output_layers'][0]))
           print(f'--------outputpathvisual----------', output_path_visual)
           output_path_textual = os.path.join(output_paths[1], model['backend'][0], model_name, str(model['output_layers'][0]))
           print(f'--------outputpathtextual----------', output_path_textual)
           model_path = [output_path_visual, output_path_textual]
           print(f'-------modelpath------:', model_path)
        else: 
           output_path = output_paths
           # Se contiene un path, estrae solo il nome del file senza estensione
           '''
           if '/' in model_name or '\\' in model_name:
                   model_name = os.path.splitext(os.path.basename(model_name))[0]
                   model_path = os.path.join(output_path, model['backend'][0], model_name, str(model['output_layers'][0]))
           else:
           '''
           model_path = os.path.join(output_path, model['backend'][0], model_name, str(model['output_layers'][0]))   
           print(f'-------modelpath------:', model_path)

        for method in imputation_methods:
                method_name = method['method_name']   
                
                # traditional imputation methods
                if method_name == 'zeros':
                    missing_file = method.get('missing_file')
                    logger.info(f"Applying zeros imputation")
                    imputer = ZerosImputator(dataset_path, missing_file, model_path)
                    imputer.impute()
                    logger.success(f"Zeros imputation completed.")

                elif method_name == 'random':
                    missing_file = method.get('missing_file')
                    seed = method.get('seed', 42)
                    logger.info(f"Applying random imputation")
                    imputer = RandomImputator(dataset_path, missing_file, model_path, seed)
                    imputer.impute()
                    logger.success(f"Random imputation completed.")

                elif method_name == 'mean':
                    missing_file = method.get('missing_file')
                    logger.info(f"Applying mean imputation")
                    imputer = MeanImputator(dataset_path, missing_file, model_path)
                    imputer.impute()
                    logger.success(f"Mean imputation completed.")

                # graph-based imputation methods
                elif method_name == 'neighmean':
                    missing_file = method.get('missing_file')
                    train_path = method.get('train_path')
                    top_k = method.get('top_k', 20)
                    logger.info(f"Applying neighmean imputation")
                    imputer = NeighmeanImputator(dataset_path, missing_file, model_path, train_path, top_k, modality)
                    imputer.impute()
                    logger.success(f"Neighmean imputation completed.")

                elif method_name == 'featprop':
                    missing_file = method.get('missing_file')
                    train_path = method.get('train_path')
                    top_k = method.get('top_k', 20)
                    layers = method.get('layers', 2)
                    logger.info(f"Applying featprop imputation")
                    imputer = FeatpropImputator(dataset_path, missing_file, model_path, train_path, top_k, layers, modality)
                    imputer.impute()
                    logger.success(f"Featprop imputation completed.")
                
                elif method_name == 'perspagerank':
                    missing_file = method.get('missing_file')
                    train_path = method.get('train_path')
                    top_k = method.get('top_k', 20)
                    layers = method.get('layers', 2)
                    alpha = method.get('alpha', 1)
                    logger.info(f"Applying perspagerank imputation")
                    imputer = PerspagerankImputator(dataset_path, missing_file, model_path, train_path, top_k, layers, alpha, modality)
                    imputer.impute()
                    logger.success(f"Perspagerank imputation completed.")
            
                elif method_name == 'heat':
                    missing_file = method.get('missing_file')
                    train_path = method.get('train_path')
                    top_k = method.get('top_k', 20)
                    layers = method.get('layers', 2)
                    time = method.get('time', 1)
                    logger.info(f"Applying heat imputation")
                    imputer = HeatImputator(dataset_path, missing_file, model_path, train_path, top_k, layers, time, modality)
                    imputer.impute()
                    logger.success(f"Heat imputation completed.")
                
                # autoencoder-based imputation methods
                elif method_name == 'autoenc':
                    missing_visual_file = method.get('missing_visual_file')
                    missing_textual_file = method.get('missing_textual_file')
                    logger.info(f"Applying autoenc imputation.")
                    imputer = AutoencImputator(dataset_path, missing_visual_file, missing_textual_file, v_emb_path = output_path_visual, t_emb_path = output_path_textual)
                    imputer.impute()
                    logger.success(f"Autoenc imputation completed.")
                
                elif method_name == 'graphautoenc':
                    missing_visual_file = method.get('missing_visual_file')
                    missing_textual_file = method.get('missing_textual_file')
                    train_path = method.get('train_path')
                    top_k = method.get('top_k', 20)
                    logger.info(f"Applying graphautoenc imputation.")
                    imputer = GraphautoencImputator(dataset_path, missing_visual_file, missing_textual_file, output_path_visual, output_path_textual, train_path, top_k)
                    imputer.impute()
                    logger.success(f"Graphautoenc imputation completed.")
                
                # new imputation method VIGAN
                elif method_name == 'vigan':
                    missing_visual_file = method.get('missing_visual_file')
                    missing_textual_file = method.get('missing_textual_file')
                    visual_dim = 2048
                    textual_dim = 768
                    gpu_ids=[]
                    logger.info(f"Applying vigan imputation.")
                    imputer = ViganImputator(dataset_path, missing_visual_file, missing_textual_file, visual_dim, textual_dim, gpu_ids)
                    imputer.impute()
                    logger.success(f"Vigan imputation completed.")       
                
                #estendere per gli altri metodi
                else: 
                    logger.warning(f"Unknown imputation method: {method_name}") 

                  
class MultimodalFeatureExtractor:

    def __init__(self, config_file_path='./config/config.yml', argv=None):
        """
        Initialize the MultimodalFeatureExtractor.

        Args:
            config_file_path (str): The path to the configuration file (default is './config/config.yml').
                                    It can be either an absolute path or a path relative to the folder of the config file.
            argv (Optional[List[str]]): Additional arguments (default is None).

        Returns:
            None
        """

        if not os.path.exists('./local/logs/'):
            os.makedirs('./local/logs/')

        log_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # logging.basicConfig(
        #     level=logging.INFO,
        #     format="%(asctime)s [%(levelname)s] %(message)s",
        #     datefmt='%Y-%m-%d-%H:%M:%S',
        #     handlers=[
        #         logging.FileHandler(filename=f'./local/logs/{log_file}.log'),
        #         logging.StreamHandler()
        #     ]
        # )

        logger.add(f"./local/logs/{log_file}.log", level="INFO",
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")
        # logging.add(lambda msg: print(msg, end=""), level="INFO")  # Stream handler for console output # Deprecated

        logger.info('\n' + banner)
        logger.log("WELCOME",
                   '***  Ducho 2.0: Towards a More Up-to-Date Unified Framework for the Extraction of Multimodal Features in Recommendation ***')
        logger.log("WELCOME",
                   '*** Brought to you by: SisInfLab, Politecnico di Bari, Italy (https://sisinflab.poliba.it) ***\n')
        self._config = Config(config_file_path, argv)
        # Get the operating system name
        os_name = platform.system()
        # set gpu to use
        os.environ['CUDA_VISIBLE_DEVICES'] = self._config.get_gpu()
        if os_name == 'Darwin':
            if torch.backends.mps.is_available():
                logger.info(
                    f'PYTORCH: Your torch version ({torch.__version__}) and system are compatible with MPS acceleration!')
            else:
                logger.warning(
                    f'PYTORCH: Your torch version ({torch.__version__}) and/or system may not be compatible with MPS acceleration!')
        else:
            logger.info('Checking if CUDA version is compatible with TensorFlow and PyTorch...')
            if len(tf.config.list_physical_devices("GPU")) > 0:
                logger.info(f'TENSORFLOW: Your tf version ({tf.__version__}) is compatible with your CUDA version!')
            else:
                logger.error(
                    f'TENSORFLOW: Your tf version ({tf.__version__}) is not compatible with your CUDA version!')
            if torch.cuda.is_available():
                logger.info(f'PYTORCH: Your torch version ({torch.__version__}) is compatible with your CUDA version!')
            else:
                logger.error(
                    f'PYTORCH: Your torch version ({torch.__version__}) is not compatible with your CUDA version!')

    def execute_extractions(self):
        """
        It executes all the extraction that have to be done
        """
        logger.info('Extraction is starting...')
        extractions_dict = self._config.get_extractions()
        for modality, modality_extractions in extractions_dict.items():
            for source, source_extractions in modality_extractions.items():
                self.do_extraction(modality, source)
        
        logger.success(f'Extraction is complete, it\'s coffee break! ☕️')


    def do_extraction(self, modality, source):
        """
        Executes extraction and applies imputation methods if configured 
        """
        logger.info(f'Extraction on {source} for {modality} modality')

        # instantiate dataset and extractor
        if modality == 'visual_textual':
            # get paths and models
            working_paths = self._config.paths_for_multiple_extraction(source, modality)
            models = self._config.get_models_list(source, modality)
            dataset = getattr(
                importlib.import_module(f"ducho.multimodal.multiple.{modality}.{camel_case(modality)}Dataset"),
                f"{camel_case(modality)}Dataset"
                )(working_paths['input_path'],
                  working_paths['output_path'],
                  columns=self._config.get_columns(modality))
            dataset._textual_dataset.set_type_of_extraction(source)
            extractor = getattr(importlib.import_module(
                f"ducho.multimodal.multiple.{modality}.{camel_case(modality)}FeatureExtractor"),
                f"{camel_case(modality)}FeatureExtractor"
            )
        else:
            # get paths and models
            working_paths = self._config.paths_for_extraction(source, modality)
            models = self._config.get_models_list(source, modality)
            if modality == 'textual':
                dataset = getattr(importlib.import_module(f"ducho.multimodal.{modality}.{camel_case(modality)}Dataset"),
                                  f"{camel_case(modality)}Dataset"
                                  )(working_paths['input_path'],
                                    working_paths['output_path'],
                                    columns=self._config.get_columns(modality))
                dataset.set_type_of_extraction(source)
            else:
                dataset = getattr(importlib.import_module(f"ducho.multimodal.{modality}.{camel_case(modality)}Dataset"),
                                  f"{camel_case(modality)}Dataset"
                                  )(working_paths['input_path'],
                                    working_paths['output_path'])
            extractor = getattr(importlib.import_module(
                f"ducho.multimodal.{modality}.{camel_case(modality)}FeatureExtractor"),
                f"{camel_case(modality)}FeatureExtractor"
            )

        # execute extraction
        # logger.info('Extraction is starting...')
        _execute_extraction_from_models_list(models=models,
                                             extractor_class=extractor,
                                             gpu=self._config.get_gpu(),
                                             dataset=dataset)
        # logger.success(f'Extraction is complete, it\'s coffee break! ☕️')

        #nuova sezione che implementa l'imputazione 
        imputation_methods = self._config.get_imputation_methods(source, modality) #chiama metodo in Config.py, controlla se ci sono metodi di imputazione definiti per questa modality e questa source
        if imputation_methods:
            logger.info(f'Applying imputation methods for modality {modality} on {source}')
                
            if modality == 'visual_textual': #se abbiamo visual_textual abbiamo path multipli
               output_paths = list(working_paths['output_path'].values())   # passare la lista   #output_path è un dict
               print(f'----------outputpaths-----------', output_paths)
               input_paths = list(working_paths['input_path'].values())
               print(f'----------inputpaths-----------', input_paths)
            else:
               output_paths = working_paths['output_path']
               print(f'----------outputpath-----------', output_paths)
               input_paths = working_paths['input_path']
               print(f'----------inputpath-----------', input_paths)
       
            _apply_imputation_methods(imputation_methods=imputation_methods, dataset_path=self._config._data_dict['dataset_path'], output_paths=output_paths, models=models, modality=modality)
       
            # creare un file IMPUTAZIONE.txt con i dettagli delle imputazioni effettuate
            with open("IMPUTAZIONE.txt", 'w', encoding='utf-8') as f:   
              f.write("DETTAGLI IMPUTAZIONE\n")   #se è avvenuta l'imputazione indica di quali item, per quale modalità, con quale metodo
              f.write("------------------------\n")
              f.write(f"Modalità mancante: {modality}\n")
              for m in imputation_methods: 
                f.write(f"Metodo d'imputazione scelto: {m['method_name']}\n")
                if m['method_name'] not in ['autoenc', 'graphautoenc', 'vigan']:
                    f.write(f"Items con modalità mancante:\n")
                    filepath = 'local/data/demo_giulia/' + m['missing_file']
                    with open(filepath, 'r', encoding='utf-8') as f_m:
                        righe = f_m.readlines()
                    f.writelines(righe)
                    f.write("\n")
                else: 
                    visual_filepath = 'local/data/demo_giulia/' + m['missing_visual_file']  
                    with open(visual_filepath, 'r', encoding='utf-8') as f_m:
                        f.write(f"Items con modalità visuale mancante:\n")
                        righe = f_m.readlines()
                    f.writelines(righe)
                    f.write("\n")
                    textual_filepath = 'local/data/demo_giulia/' + m['missing_textual_file']  
                    with open(textual_filepath, 'r', encoding='utf-8') as f_m:
                        f.write(f"Items con modalità testuale mancante:\n")
                        righe = f_m.readlines()
                    f.writelines(righe)
                    f.write("\n")

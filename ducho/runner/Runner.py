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

# Import imputators
from ducho.imputation.zeros import ZerosImputator
from ducho.imputation.random import RandomImputator
from ducho.imputation.mean import MeanImputator

from ducho.imputation.neighmean import NeighmeanImputator
from ducho.imputation.perspagerank import PerspagerankImputator
from ducho.imputation.heat import HeatImputator
from ducho.imputation.featprop import FeatpropImputator

from ducho.imputation.autoenc import AutoencImputator
from ducho.imputation.graphautoenc import GraphautoencImputator
from ducho.imputation.varautoenc import VarautoencImputator
from ducho.imputation.vargraphautoenc import VargraphautoencImputator

from ducho.imputation.vigan_imputation.vigan import ViganImputator 
from ducho.imputation.smil_imputation.smil import SmilImputator 
from ducho.imputation.mmin_imputation.mmin import MminImputator
from ducho.imputation.dicmor_imputation.dicmor import DicmorImputator

from prettytable import PrettyTable 

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

def _apply_imputation_methods(methods, dataset_path, input_paths, output_paths, modality):
    """
    Apply imputation methods to reconstruct missing data.

    Args: 
        methods (List[Dict]): List of dictionaries containg imputation method configurations.
        dataset_path (str): Base path to the dataset.
        input_paths (list): Path/s to saved embeddings.
        output_paths (list): Path/s where to save embeddings.
        modality (str): Considered modality.

    Returns:
        methods_errors or None
    """
    methods_errors = {}
    for method in methods:
                method_name = method['method_name']   

                # Traditional imputation methods
                if method_name == 'zeros':
                    missing_file = method.get('missing_file', None)
                    missing_percentage = method.get('missing_percentage', None)
                    items_file = method.get('items_file', None)
                    seed = method.get('seed', 42)
                    logger.info(f"Applying zeros imputation")
                    imputer = ZerosImputator(dataset_path, missing_file, input_paths, output_paths, 
                                             missing_percentage, items_file, seed)
                    errors = imputer.impute()
                    logger.success(f"Zeros imputation completed.")
                    methods_errors['zeros'] = errors

                elif method_name == 'random':
                    missing_file = method.get('missing_file', None)
                    missing_percentage = method.get('missing_percentage', None)
                    items_file = method.get('items_file', None)
                    seed = method.get('seed', 42)
                    logger.info(f"Applying random imputation")
                    imputer = RandomImputator(dataset_path, missing_file, input_paths, output_paths, 
                                              missing_percentage, items_file, seed)
                    errors = imputer.impute()
                    logger.success(f"Random imputation completed.")
                    methods_errors['random'] = errors

                elif method_name == 'mean':
                    missing_file = method.get('missing_file', None)
                    missing_percentage = method.get('missing_percentage', None)
                    items_file = method.get('items_file', None)
                    seed = method.get('seed', 42)
                    logger.info(f"Applying mean imputation")
                    imputer = MeanImputator(dataset_path, missing_file, input_paths, output_paths, 
                                            missing_percentage, items_file, seed)
                    errors = imputer.impute()
                    logger.success(f"Mean imputation completed.")
                    methods_errors['mean'] = errors

                # Graph-based imputation methods
                elif method_name == 'neighmean':
                    missing_file = method.get('missing_file', None)
                    missing_percentage = method.get('missing_percentage', None)
                    items_file = method.get('items_file', None)
                    seed = method.get('seed', 42)
                    train_path = method.get('train_path')
                    top_k = method.get('top_k', 20)
                    logger.info(f"Applying neighmean imputation")
                    imputer = NeighmeanImputator(dataset_path, missing_file, input_paths, output_paths, 
                                                 train_path, top_k, modality, missing_percentage, items_file, seed)
                    errors =  imputer.impute()
                    logger.success(f"Neighmean imputation completed.")
                    methods_errors['neighmean'] = errors

                elif method_name == 'featprop':
                    missing_file = method.get('missing_file', None)
                    missing_percentage = method.get('missing_percentage', None)
                    items_file = method.get('items_file', None)
                    seed = method.get('seed', 42)
                    train_path = method.get('train_path')
                    top_k = method.get('top_k', 10)
                    layers = method.get('layers', 1)
                    logger.info(f"Applying featprop imputation")
                    imputer = FeatpropImputator(dataset_path, missing_file, input_paths, output_paths, 
                                                train_path, top_k, layers, modality, missing_percentage, items_file, seed)
                    errors = imputer.impute()
                    logger.success(f"Featprop imputation completed.")
                    methods_errors['featprop'] = errors
                
                elif method_name == 'perspagerank':
                    missing_file = method.get('missing_file', None)
                    missing_percentage = method.get('missing_percentage', None)
                    items_file = method.get('items_file', None)
                    seed = method.get('seed', 42)
                    train_path = method.get('train_path')
                    top_k = method.get('top_k', 10)
                    layers = method.get('layers', 1)
                    alpha = method.get('alpha', 0.1)
                    logger.info(f"Applying perspagerank imputation")
                    imputer = PerspagerankImputator(dataset_path, missing_file, input_paths, output_paths, 
                                                    train_path, top_k, layers, alpha, modality, missing_percentage, items_file, seed)
                    errors = imputer.impute()
                    logger.success(f"Perspagerank imputation completed.")
                    methods_errors['perspagerank'] = errors
            
                elif method_name == 'heat':
                    missing_file = method.get('missing_file', None)
                    missing_percentage = method.get('missing_percentage', None)
                    items_file = method.get('items_file', None)
                    seed = method.get('seed', 42)
                    train_path = method.get('train_path')
                    top_k = method.get('top_k', 10)
                    layers = method.get('layers', 1)
                    time = method.get('time', 5.0)
                    logger.info(f"Applying heat imputation")
                    imputer = HeatImputator(dataset_path, missing_file, input_paths, output_paths, 
                                            train_path, top_k, layers, time, modality, missing_percentage, items_file, seed)
                    errors = imputer.impute()
                    logger.success(f"Heat imputation completed.")
                    methods_errors['heat'] = errors
                
                # Autoencoder-based imputation methods
                elif method_name == 'autoenc':
                    missing_visual_file = method.get('missing_visual_file', None)
                    missing_textual_file = method.get('missing_textual_file', None)
                    missing_percentage_visual = method.get('missing_percentage_visual', None)
                    missing_percentage_textual = method.get('missing_percentage_textual', None)
                    items_file = method.get('items_file', None)
                    seed = method.get('seed', 42)
                    num_epochs = method.get('num_epochs', 1000)
                    learning_rate = method.get('learning_rate', 0.01)
                    batch_size = method.get('batch_size', 32)
                    logger.info(f"Applying autoenc imputation.")
                    imputer = AutoencImputator(dataset_path, missing_visual_file, missing_textual_file, input_paths, output_paths, seed, 
                                               missing_percentage_visual, missing_percentage_textual, items_file, num_epochs, learning_rate, batch_size)
                    errors = imputer.impute()
                    logger.success(f"Autoenc imputation completed.") 
                    methods_errors['autoenc'] = errors
                
                elif method_name == 'graphautoenc':
                    missing_visual_file = method.get('missing_visual_file', None)
                    missing_textual_file = method.get('missing_textual_file', None)
                    missing_percentage_visual = method.get('missing_percentage_visual', None)
                    missing_percentage_textual = method.get('missing_percentage_textual', None)
                    items_file = method.get('items_file', None)
                    train_path = method.get('train_path')
                    top_k = method.get('top_k', 20)
                    seed = method.get('seed', 42)
                    num_epochs = method.get('num_epochs', 1000)
                    learning_rate = method.get('learning_rate', 1e-4)
                    batch_size = method.get('batch_size', 32)
                    logger.info(f"Applying graphautoenc imputation.")
                    imputer = GraphautoencImputator(dataset_path, missing_visual_file, missing_textual_file, input_paths, output_paths, train_path, top_k, seed, 
                                                    missing_percentage_visual, missing_percentage_textual, items_file, num_epochs, learning_rate, batch_size)
                    errors = imputer.impute()
                    logger.success(f"Graphautoenc imputation completed.")
                    methods_errors['graphautoenc'] = errors
                
                elif method_name == 'varautoenc':
                    missing_visual_file = method.get('missing_visual_file', None)
                    missing_textual_file = method.get('missing_textual_file', None)
                    missing_percentage_visual = method.get('missing_percentage_visual', None)
                    missing_percentage_textual = method.get('missing_percentage_textual', None)
                    items_file = method.get('items_file', None)
                    seed = method.get('seed', 42)
                    num_epochs = method.get('num_epochs', 1000)
                    learning_rate = method.get('learning_rate', 0.01)
                    batch_size = method.get('batch_size', 32)
                    beta_visual = method.get('beta_visual', 0.01)
                    beta_textual = method.get('beta_textual', 1.0)
                    logger.info(f"Applying varautoenc imputation.")
                    imputer = VarautoencImputator(dataset_path, missing_visual_file, missing_textual_file, input_paths, output_paths, seed, 
                                                  missing_percentage_visual, missing_percentage_textual, items_file, num_epochs, learning_rate, batch_size, beta_visual, beta_textual)
                    errors = imputer.impute()
                    logger.success(f"Varautoenc imputation completed.") 
                    methods_errors['varautoenc'] = errors
                
                elif method_name == 'vargraphautoenc':
                    missing_visual_file = method.get('missing_visual_file', None)
                    missing_textual_file = method.get('missing_textual_file', None)
                    missing_percentage_visual = method.get('missing_percentage_visual', None)
                    missing_percentage_textual = method.get('missing_percentage_textual', None)
                    items_file = method.get('items_file', None)
                    train_path = method.get('train_path')
                    top_k = method.get('top_k', 20)
                    seed = method.get('seed', 42)
                    num_epochs = method.get('num_epochs', 1000)
                    learning_rate = method.get('learning_rate', 1e-4)
                    batch_size = method.get('batch_size', 32)
                    beta_visual = method.get('beta_visual', 1e-3)
                    beta_textual = method.get('beta_textual', 1e-3)
                    logger.info(f"Applying vargraphautoenc imputation.")
                    imputer = VargraphautoencImputator(dataset_path, missing_visual_file, missing_textual_file, input_paths, output_paths, train_path, top_k, seed, 
                                                       missing_percentage_visual, missing_percentage_textual, items_file, num_epochs, learning_rate, batch_size, beta_visual, beta_textual)
                    errors = imputer.impute()
                    logger.success(f"Vargraphautoenc imputation completed.")
                    methods_errors['vargraphautoenc'] = errors
                
                # GAN-based imputation method
                elif method_name == 'vigan':
                    missing_visual_file = method.get('missing_visual_file', None)
                    missing_textual_file = method.get('missing_textual_file', None)
                    missing_percentage_visual = method.get('missing_percentage_visual', None)
                    missing_percentage_textual = method.get('missing_percentage_textual', None)
                    items_file = method.get('items_file', None)
                    visual_dim = 2048
                    textual_dim = 768
                    gpu_ids = []
                    seed = method.get('seed', 42)
                    num_epochs_cyclegan = method.get('num_epochs_cyclegan', 5)
                    num_epochs_ae = method.get('num_epochs_ae', 5)
                    num_epochs_joint = method.get('num_epochs_joint', 40)
                    learning_rate = method.get('learning_rate', 0.0002)
                    batch_size = method.get('batch_size', 64) 
                    beta1 = method.get('beta1', 0.5)
                    lambda_A = method.get('lambda_A', 10.0)
                    lambda_B = method.get('lambda_B', 10.0)
                    lambda_idt = method.get('lambda_idt', 0.5)
                    lambda_C = method.get('lambda_C', 1.0)
                    lambda_D = method.get('lambda_D', 1.0)
                    n_iter_G = method.get('n_iter_G', 2)
                    n_iter_D = method.get('n_iter_D', 1)
                    logger.info(f"Applying vigan imputation.")
                    imputer = ViganImputator(dataset_path, missing_visual_file, missing_textual_file, input_paths, output_paths, visual_dim, textual_dim, gpu_ids, seed, 
                                             missing_percentage_visual, missing_percentage_textual, items_file, num_epochs_cyclegan, num_epochs_ae, num_epochs_joint,
                                             learning_rate, batch_size, beta1, lambda_A, lambda_B, lambda_idt, lambda_C, lambda_D, n_iter_G, n_iter_D)
                    errors = imputer.impute()
                    logger.success(f"Vigan imputation completed.")  
                    methods_errors['vigan'] = errors

                # Bayesian meta-Learning-based imputation method
                elif method_name == 'smil':
                    missing_visual_file = method.get('missing_visual_file', None)
                    missing_textual_file = method.get('missing_textual_file', None)
                    missing_percentage_visual = method.get('missing_percentage_visual', None)
                    missing_percentage_textual = method.get('missing_percentage_textual', None)
                    items_file = method.get('items_file', None)
                    visual_dim=2048
                    textual_dim=768
                    gpu_ids=[]
                    seed = method.get('seed', 42)
                    num_epochs = method.get('num_epochs', 60)
                    learning_rate = method.get('learning_rate', 0.001)
                    batch_size = method.get('batch_size', 32)
                    alpha = method.get('alpha', 1.0)
                    beta = method.get('beta', 1.0)
                    logger.info(f"Applying smil imputation.")
                    imputer = SmilImputator(dataset_path, missing_visual_file, missing_textual_file, input_paths, output_paths, visual_dim, textual_dim, gpu_ids, seed, 
                                            missing_percentage_visual, missing_percentage_textual, items_file, num_epochs, learning_rate, batch_size, alpha, beta)
                    errors = imputer.impute() 
                    logger.success(f"Smil imputation completed.")
                    methods_errors['smil'] = errors
                
                # Residual autoencoder-based imputation method 
                elif method_name == 'mmin':
                    missing_visual_file = method.get('missing_visual_file', None)
                    missing_textual_file = method.get('missing_textual_file', None)
                    missing_percentage_visual = method.get('missing_percentage_visual', None)
                    missing_percentage_textual = method.get('missing_percentage_textual', None)
                    items_file = method.get('items_file', None)
                    visual_dim=2048
                    textual_dim=768
                    embd_size_v=128  #
                    embd_size_t=128  #
                    gpu_ids=[]
                    seed = method.get('seed', 42)
                    num_epochs = method.get('num_epochs', 100)
                    learning_rate = method.get('learning_rate', 0.0002)
                    batch_size = method.get('batch_size', 128)
                    n_blocks = method.get('n_blocks', 3)
                    dropout = method.get('dropout', 0.3)
                    mse_weight = method.get('mse_weight', 1.0)
                    cycle_weight = method.get('cycle_weight', 1.0)
                    logger.info(f"Applying mmil imputation.")
                    imputer = MminImputator(dataset_path, missing_visual_file, missing_textual_file, input_paths, output_paths, visual_dim, textual_dim, embd_size_v, embd_size_t, gpu_ids, seed, 
                                            missing_percentage_visual, missing_percentage_textual, items_file, num_epochs, learning_rate, batch_size, n_blocks, dropout, mse_weight, cycle_weight)
                    errors = imputer.impute()
                    logger.success(f"Mmin imputation completed.")
                    methods_errors['mmin'] = errors
                
                # Distribution-based imputation method
                elif method_name == 'dicmor':
                    missing_visual_file = method.get('missing_visual_file', None)
                    missing_textual_file = method.get('missing_textual_file', None)
                    missing_percentage_visual = method.get('missing_percentage_visual', None)
                    missing_percentage_textual = method.get('missing_percentage_textual', None)
                    items_file = method.get('items_file', None)
                    visual_dim=2048
                    textual_dim=768
                    embd_size_v=64 #128
                    embd_size_t=64 #128
                    gpu_ids=[]
                    seed = method.get('seed', 42)
                    num_epochs = method.get('num_epochs', 8)
                    learning_rate = method.get('learning_rate', 0.0001)
                    batch_size = method.get('batch_size', 128) #16
                    log_p_weight = method.get('log_p_weight', 0.1)
                    n_flow = method.get('n_flow', 32)
                    n_block = method.get('n_block', 1)
                    rcan_blocks = method.get('rcan_blocks', 20)
                    logger.info(f"Applying dicmor imputation.")
                    imputer = DicmorImputator(dataset_path, missing_visual_file, missing_textual_file, input_paths, output_paths, visual_dim, textual_dim, embd_size_v, embd_size_t, gpu_ids, seed, 
                                              missing_percentage_visual, missing_percentage_textual, items_file, num_epochs, learning_rate, batch_size, log_p_weight, n_flow, n_block, rcan_blocks)
                    errors = imputer.impute()
                    logger.success(f"Dicmor imputation completed.")
                    methods_errors['dicmor'] = errors

                # Extend with any other imputation methods
                else: 
                    logger.warning(f"Unknown imputation method: {method_name}") 

    return methods_errors

          
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
        extractions_dict = self._config.get_extractions() 
        if extractions_dict == {}:
            logger.info(f'Extraction not required.')
        else:
          logger.info('Extraction is starting...')
          for modality, modality_extractions in extractions_dict.items():
            for source, source_extractions in modality_extractions.items():
                self.do_extraction(modality, source)
          logger.success(f'Extraction is complete, it\'s coffee break! ☕️')

    def do_extraction(self, modality, source):
        """
        Executes extraction.
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

class MultimodalFeatureImputator:

    def __init__(self, config_file_path='./config/config.yml', argv=None):
        """
        Initialize the MultimodalFeatureImputator.
        
        Args:
            config_file_path (str): The path to the configuration file (default is './config/config.yml').
                                    It can be either an absolute path or a path relative to the folder of the config file.
            argv (Optional[List[str]]): Additional arguments (default is None).
        
        Returns:
            None
        """
        self._config = Config(config_file_path, argv)

    def execute_imputations(self):
        """
        It executes all the imputations that have to be done
        """
        imputations_dict = self._config.get_imputations()
        if imputations_dict == {}:
            logger.info(f'Imputation not required.')
        else:
            logger.info('Imputation is starting...')
            for mode, modalities in imputations_dict['imputation'].items():
                for modality in modalities.keys():
                  self.do_imputation(mode, modality)
            logger.success(f'Imputation is complete!')

    def do_imputation(self, mode, modality): 
        """
        Executes imputation.
        """
        logger.info(f'Apply {mode} imputation for {modality} modality')
        #imputation_methods = self._config.get_imputation_methods(modality) #call a method in Config.py, check if there are imputation methods
        
        # Get paths and methods
        if mode == 'crossmodal':
            working_paths = self._config.paths_for_multiple_imputation(mode, modality)
            output_paths = list(working_paths['output_path'].values())
            input_paths = list(working_paths['input_path'].values())
            methods = self._config.get_methods_list(mode, modality) 
        else:
            working_paths = self._config.paths_for_imputation(mode, modality)
            output_paths = working_paths['output_path']
            input_paths = working_paths['input_path']
            methods = self._config.get_methods_list(mode, modality)

        # Validate imputation methods
        self._config.validate_methods(methods, modality)

        # Apply imputation methods
        if mode == 'unimodal':                                                                                           
           methods_errors = _apply_imputation_methods(methods=methods, dataset_path=self._config._data_dict['dataset_path'], input_paths=input_paths, output_paths=output_paths, modality=modality)    #path to extracted embeddings and path to imputed embeddings
           # Print the imputation summary 
           self.print_imputation_summary(mode, modality, methods, methods_errors)
        else:
           methods_errors = _apply_imputation_methods(methods=methods, dataset_path=self._config._data_dict['dataset_path'], input_paths=input_paths, output_paths=output_paths, modality=modality)    #path to extracted embeddings and path to imputed embeddings
           # Print the imputation summary 
           self.print_imputation_summary(mode, modality, methods, methods_errors)
     
    def print_imputation_summary(self, mode, modality, methods, methods_errors):
        """
        Create an IMPUTATION_SUMMARY.txt file with details of the requested imputations
        """       
        dataset_path = self._config._data_dict['dataset_path']
        with open(f"{dataset_path}/IMPUTATION_SUMMARY.txt", 'w', encoding='utf-8') as f:   
              f.write("IMPUTATION DETAILS\n")   
              # If imputation has been performed, indicate for which items, for which modality, and the imputation method used
              f.write("\n")
              f.write(f"Imputation mode: {mode}\n")
              f.write(f"Modality: {modality}\n")
              for m in methods:
                name = m['method_name']
                f.write("\n")
                f.write(f"Chosen imputation method: {name}\n")
                
                if mode == 'unimodal':
                    if 'items_file' in m.keys():
                        for percentage, (mse, cosine_sim) in methods_errors[name].items():
                            percentage_int = int(percentage*100)
                            f.write(f"Missingness percentage: {percentage_int}%\n")
                            with open(f"{dataset_path}/artificial_missing_file_{percentage}_{name}.tsv", 'r', encoding='utf-8') as f_m:
                               righe = f_m.readlines()
                            riga = ", ".join(r.strip() for r in righe)
                            f.write(f"Items with {modality} modality imputed: {riga}\n")
                            f.write(f"Recontruction errors: MSE = {mse} - COSINE SIMILARITY = {cosine_sim}\n")
                            f.write("\n")
                            # Cleanup
                            os.remove(f"{dataset_path}/artificial_missing_file_{percentage}_{name}.tsv") 
                        
                    else:
                        filepath = f'{dataset_path}/' + m['missing_file']
                        with open(filepath, 'r', encoding='utf-8') as f_m:
                            righe = f_m.readlines()
                        riga = ", ".join(r.strip() for r in righe)
                        f.write(f"Items with {modality} modality imputed: {riga}\n")    

                if mode == 'crossmodal':
                    if 'items_file' in m.keys():
                        for (percentage_visual, mse_visual, cosine_sim_visual), (percentage_textual, mse_textual, cosine_sim_textual) in zip(methods_errors[name]['visual'], methods_errors[name]['textual']):
                            percentage_visual_int = int(percentage_visual*100)
                            percentage_textual_int = int(percentage_textual*100)
                            f.write(f"Visual missingness percentage: {percentage_visual_int}%\n")
                            f.write(f"Textual missingness percentage: {percentage_textual_int}%\n")
                            with open(f"{dataset_path}/artificial_missing_visual_file_{percentage_visual}_{name}.tsv", 'r', encoding='utf-8') as f_m:
                               righe = f_m.readlines()
                            riga = ", ".join(r.strip() for r in righe)
                            f.write(f"Items with visual modality imputed: {riga}\n")  
                            with open(f"{dataset_path}/artificial_missing_textual_file_{percentage_textual}_{name}.tsv", 'r', encoding='utf-8') as f_m:
                               righe = f_m.readlines()
                            riga = ", ".join(r.strip() for r in righe)
                            f.write(f"Items with textual modality imputed: {riga}\n")  
                            f.write(f"Visual reconstruction errors: MSE = {mse_visual} - COSINE SIMILARITY = {cosine_sim_visual}\n")
                            f.write(f"Textual reconstruction errors: MSE = {mse_textual} - COSINE SIMILARITY = {cosine_sim_textual}\n")
                            f.write("\n")
                            # Cleanup
                            os.remove(f"{dataset_path}/artificial_missing_visual_file_{percentage_visual}_{name}.tsv") 
                            os.remove(f"{dataset_path}/artificial_missing_textual_file_{percentage_textual}_{name}.tsv") 
                        
                    else:
                        visual_filepath = f'{dataset_path}/' + m['missing_visual_file']  
                        with open(visual_filepath, 'r', encoding='utf-8') as f_m:
                            righe = f_m.readlines() 
                        riga = ", ".join(r.strip() for r in righe)
                        f.write(f"Items with visual modality imputed: {riga}\n") 
                   
                        textual_filepath = f'{dataset_path}/' + m['missing_textual_file']  
                        with open(textual_filepath, 'r', encoding='utf-8') as f_m:
                             righe = f_m.readlines()
                        riga = ", ".join(r.strip() for r in righe)
                        f.write(f"Items with textual modality imputed: {riga}\n")    
                        f.write("\n")
                
              # Create summary table
              self._write_imputation_summary_tables_by_missingness(f=f, mode=mode, modality=modality, methods=methods, methods_errors=methods_errors)
          
    def _write_imputation_summary_tables_by_missingness(self, f, mode, modality, methods, methods_errors):
        """
        Create summary tables of reconstruction error metrics using PrettyTable 
        """            
        if all(v is None for v in methods_errors.values()):
           return
    
        def draw_table(title, field_names, rows):
            if not rows:
              return
            table = PrettyTable()
            table.field_names = field_names
            for r in rows:
               table.add_row(r)
            f.write("\n")
            f.write(title + "\n")
            f.write(str(table))
            f.write("\n")

        if mode == 'unimodal':
          # Retrieve all missingness percentages (sorted)
          missingness_values = sorted({int(p * 100) for errors in methods_errors.values() for p in errors.keys()})
          for miss in missingness_values:
            rows = []
            for m in methods:
                name = m['method_name']
                if name not in methods_errors:
                   continue
                # Find the correct percentage
                for p, (mse, cosine_sim) in methods_errors[name].items():
                    if int(p * 100) == miss:
                       rows.append([name, f"{mse:.17f}", f"{cosine_sim:.17f}"])
           
            draw_table(title=f"SUMMARY TABLE - Missingness {miss}% ({modality})", 
                       field_names=["Method", "MSE", "Cosine Similarity"], rows=rows)

        elif mode == 'crossmodal':  
            # Collects all configurations (missing percentage visual, missing percentage textual)
            configurations = set()
            for m in methods:
                    name = m['method_name']
                    visual_list = methods_errors[name]['visual']
                    textual_list = methods_errors[name]['textual']
                    # Zip to correctly match visual/textual
                    for (p_v, _, _), (p_t, _, _) in zip(visual_list, textual_list):
                        configurations.add( (p_v, p_t) )
            # Sort by visual, then textual
            configurations = sorted(configurations, key=lambda x: (x[0], x[1]))
            for p_visual, p_textual in configurations:
                rows = [] 
                for m in methods:
                    name = m['method_name']
                    if name not in methods_errors:
                       continue   
                    # Find the corresponding configuration
                    visual_list = methods_errors[name]['visual']
                    textual_list = methods_errors[name]['textual'] 
                    found = False
                    for (v_p, mse_v, cos_v), (t_p, mse_t, cos_t) in zip(visual_list, textual_list):
                        if v_p == p_visual and t_p == p_textual:
                           rows.append([name, f"{mse_v:.17f}", f"{cos_v:.17f}", f"{mse_t:.17f}", f"{cos_t:.17f}"])
                           found = True
                           break
                    if not found:
                        # The method does not have this configuration
                        rows.append([name, "-", "-", "-", "-"])   
                draw_table(title=(f"SUMMARY TABLE - "f"Missingness VISUAL {int(p_visual*100)}% / TEXTUAL {int(p_textual*100)}%"), 
                           field_names=["Method", "MSE Visual", "Cosine Similarity Visual", "MSE Textual", "Cosine Similarity Textual"], rows=rows)
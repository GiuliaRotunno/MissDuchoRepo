from ducho.runner.Runner import MultimodalFeatureExtractor, MultimodalFeatureImputator
import torch
import torchvision.models as models
from collections import OrderedDict
import os
import gdown
import numpy as np
import random

def set_seed(seed = 42):
    """Set all seeds to make results reproducible (deterministic mode).
       When seed is None, disables deterministic mode.
    :param seed: an integer to your choosing
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"


def main():
    set_seed()
    extractor_obj = MultimodalFeatureExtractor(config_file_path='./demos/demo_elliot/config.yml')
    extractor_obj.execute_extractions()
    imputator_obj = MultimodalFeatureImputator(config_file_path='./demos/demo_elliot/config.yml')
    imputator_obj.execute_imputations()                                                          

if __name__ == '__main__':
    main()

# Command to run the code:  PYTHONPATH=. python demos/demo_elliot/run.py
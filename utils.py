import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Фиксирует seed для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # для datasets/huggingface
    # import os
    # os.environ["PYTHONHASHSEED"] = str(seed)
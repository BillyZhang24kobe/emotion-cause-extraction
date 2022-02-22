import torch
import random
import numpy as np
import config

def set_seed(args):
    args.seed = config.SEED
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
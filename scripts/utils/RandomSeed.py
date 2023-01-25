"""
Fix random seed for program reproducible.
python scripts/utils/RandomSeed.py <seed>
"""

import random
import numpy as np
import torch
import argparse


def setup_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return seed

def main(args):
    setup_random_seed(args.seed)

    random_num = np.random.normal(0, 0.1, 10)
    print(random_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2023, help='Random seed number. Default:2023')
    args = parser.parse_args()

    main(args)
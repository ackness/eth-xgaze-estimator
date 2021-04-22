import argparse

from solvers import get_solver
from utils.config import get_config

if __name__ == '__main__':
    # --------------  args  -----------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg', type=str, help="path to config file",
    )
    args = parser.parse_args()
    # --------------  args  -----------------

    CONFIG = get_config(args.cfg)

    solver = get_solver(CONFIG)

    if CONFIG.mode == "train":
        solver.train()
    elif CONFIG.mode == "test":
        solver.test()
    else:
        raise KeyError

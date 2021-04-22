import importlib
import os

# from solvers import *

__all__ = ['get_solver']


def get_solver(config):
    all_solver_list = [f[:-3] for f in os.listdir(os.path.dirname(__file__))]
    # print(all_solver_list)
    if config.solver in all_solver_list:
        print(f"Using Solver {config.solver}")
        module = importlib.import_module(f".{config.solver}", package="solvers")
        # module = eval(f"{config.solver}")
        solver = module.Solver(config)
    else:
        raise NotImplementedError(f"Not find a proper solver={config.solver}!")
    return solver

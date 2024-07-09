from .spc import SPCL2L2Solver


SOLVERS = {
    "SPC": SPCL2L2Solver
}


def get_solver(acquisition_model):
    return SOLVERS[acquisition_model.__class__.__name__]
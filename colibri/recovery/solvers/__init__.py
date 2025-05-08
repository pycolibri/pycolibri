from .spc import L2L2SolverSPC
from .modulo import L2L2SolverModulo


SOLVERS = {
    "SPC": L2L2SolverSPC,
    "Modulo": L2L2SolverModulo,
}


def get_solver(acquisition_model):
    return SOLVERS[acquisition_model.__class__.__name__]
from .spc import L2L2SolverSPC
from .modulo import L2L2SolverModulo
from .sd_cassi import L2L2SolverSDCASSI

SOLVERS = {
    "SPC": L2L2SolverSPC,
    "Modulo": L2L2SolverModulo,
    "SD_CASSI": L2L2SolverSDCASSI,
}


def get_solver(acquisition_model):
    return SOLVERS[acquisition_model.__class__.__name__]
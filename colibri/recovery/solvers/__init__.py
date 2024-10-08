from .spc import L2L2SolverSPC


SOLVERS = {
    "SPC": L2L2SolverSPC
}


def get_solver(acquisition_model):
    return SOLVERS[acquisition_model.__class__.__name__]
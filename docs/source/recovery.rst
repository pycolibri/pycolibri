Recovery
========

The recovery module provides an implementation of state-of-the-art recovery algorithms widely used 
in image restoration on inverse problems.

.. math::

    \textbf{x}^{*} \in  \underset{\textbf{x}}{ \argmin } \; f(\mathbf{x})+ \lambda g(\mathbf{x})

where :math:`f(\mathbf{x})` is the fidelity term ( e.g., :math:`\Vert \textbf{y} - \textbf{H}(\textbf{x}) \Vert_2^2` ) and :math:`g(\mathbf{x})` is the prior term.

List of Algorithms
--------------------

.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.recovery.fista.Fista
    colibri.recovery.pnp.PnP_ADMM
    colibri.recovery.lfsi.LFSI

    

List of Solvers
--------------------

.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.recovery.solvers.core.L2L2Solver
    colibri.recovery.solvers.spc.L2L2SolverSPC
    colibri.recovery.solvers.modulo.L2L2SolverModulo



Fidelity Terms
--------------------
The module contains differents fidelity terms :math:`f(\mathbf{x})` that can be used in the recovery algorithms.


.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.recovery.terms.fidelity.L2
    colibri.recovery.terms.fidelity.L1



Prior Terms
--------------------
The module contains differents prior terms :math:`g(\mathbf{x})` that can be used in the recovery algorithms.


    
.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.recovery.terms.prior.Sparsity
    colibri.recovery.terms.prior.LearnedPrior
    
Transormation
--------------------

The module contains differents signal transforms that can be used in the recovery algorithms.

    
.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.recovery.terms.transforms.DCT2D
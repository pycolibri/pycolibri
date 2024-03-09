Optics
======

The PyColibri optics module encompasses a collection of optical systems extensively employed in the computational imaging research for spectral, depth, phase and other imaging applications. This module facilitates the simulation of optical propagation and the design of optical elements inside the systems, offering researchers and developers a powerful toolkit for advancing imaging technologies.


Mathematically, all implemented optical systems  are of the form

.. math::

    y = \forwardLinear_{\learnedOptics}(x) + \noise

where :math:`x\in\xset` is the input optical field, :math:`y\in\yset` are the acquired signal,
:math:`\forwardLinear:\xset\mapsto \yset` is a deterministic (linear or non-linear) optics model of the acquisition system, 
:math:`\learnedOptics` is a set of learnable parameters characterizing the optical system,
and :math:`\noise` is a stochastic mapping which characterizes the noise affecting the measurements.



Spectral Imaging systems
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri_hdsp.optics.cassi.CASSI
    colibri_hdsp.optics.spc.SPC


Functional miscellanea of the optics module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: stubs
    :template: methods_template.rst
    :nosignatures:

    colibri_hdsp.optics.functional.prism_operator
    colibri_hdsp.optics.functional.forward_color_cassi
    colibri_hdsp.optics.functional.backward_color_cassi
    colibri_hdsp.optics.functional.forward_dd_cassi
    colibri_hdsp.optics.functional.backward_dd_cassi
    colibri_hdsp.optics.functional.forward_cassi
    colibri_hdsp.optics.functional.backward_cassi
    colibri_hdsp.optics.functional.forward_spc
    colibri_hdsp.optics.functional.backward_spc



.. _optics:

Optics
======

The PyColibri optics module encompasses a collection of optical systems extensively employed in the computational imaging research for spectral, depth, phase and other imaging applications. This module facilitates the simulation of optical propagation and the design of optical elements inside the systems, offering researchers and developers a powerful toolkit for advancing imaging technologies.


Mathematically, all implemented optical systems are of the form

.. math::

    \mathbf{y} = \forwardLinear_{\learnedOptics}(\mathbf{x}) + \noise

where :math:`\mathbf{x}\in\xset` is the input optical field, :math:`\mathbf{y}\in\yset` are the acquired signal,
:math:`\forwardLinear:\xset\mapsto \yset` is a deterministic (linear or non-linear) optics model of the acquisition system, 
:math:`\learnedOptics` is a set of learnable parameters characterizing the optical system,
and :math:`\noise` is a stochastic mapping which characterizes the noise affecting the measurements.



Spectral Imaging systems
------------------------------

.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.optics.cassi.SD_CASSI
    colibri.optics.cassi.DD_CASSI
    colibri.optics.cassi.C_CASSI
    colibri.optics.spc.SPC
    colibri.optics.doe.SingleDOESpectral
    colibri.optics.modulo.Modulo

Phase Imaging systems
------------------------------
.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.optics.phase_imaging.CodedPhaseImaging

Functional operators of the optical systems
---------------------------------------------

.. autosummary::
    :toctree: stubs
    :template: methods_template.rst
    :nosignatures:

    colibri.optics.functional.coded_phase_imaging_forward
    colibri.optics.functional.coded_phase_imaging_backward
    colibri.optics.functional.forward_color_cassi
    colibri.optics.functional.backward_color_cassi
    colibri.optics.functional.forward_dd_cassi
    colibri.optics.functional.backward_dd_cassi
    colibri.optics.functional.forward_sd_cassi
    colibri.optics.functional.backward_sd_cassi
    colibri.optics.functional.forward_spc
    colibri.optics.functional.backward_spc
    colibri.optics.functional.psf_single_doe_spectral 
    colibri.optics.functional.convolutional_sensing   
    colibri.optics.functional.fourier_conv
    colibri.optics.functional.signal_conv
    colibri.optics.functional.wiener_filter     
    colibri.optics.functional.gaussian_noise  
    colibri.optics.functional.modulo

Functional operators of optical elements
-------------------------------------------

.. autosummary::
    :toctree: stubs
    :template: methods_template.rst
    :nosignatures:

    colibri.optics.functional.prism_operator
    colibri.optics.functional.circular_aperture   
    colibri.optics.functional.height2phase    
    colibri.optics.functional.ideal_panchromatic_sensor  
    colibri.optics.functional.wave_number 
    colibri.optics.functional.get_spatial_coords    

Functional operators of light propagation
-------------------------------------------

.. autosummary::
    :toctree: stubs
    :template: methods_template.rst
    :nosignatures:

    colibri.optics.functional.scalar_diffraction_propagation
    colibri.optics.functional.transfer_function_angular_spectrum    
    colibri.optics.functional.transfer_function_fresnel
    colibri.optics.functional.fraunhofer_propagation
    colibri.optics.functional.fraunhofer_inverse_propagation


state-of-the-art optical elements
-------------------------------------------
.. autosummary::
    :toctree: stubs
    :template: methods_template.rst
    :nosignatures:

    colibri.optics.sota_does.conventional_lens
    colibri.optics.sota_does.spiral_doe



utils
-------------------------------------------
.. autosummary::
    :toctree: stubs
    :template: methods_template.rst
    :nosignatures:

    colibri.optics.functional.add_pad
    colibri.optics.functional.unpad
    colibri.optics.functional.fft
    colibri.optics.functional.ifft
    



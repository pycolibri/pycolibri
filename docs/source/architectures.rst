Architectures
========================

End-to-End (E2E) represents a collection of advanced architectures for computational imaging that integrates optical systems with deep learning models. These architectures are designed for flexibility and performance across a variety of imaging applications. In E2E models, the optical layer is modeled as a neural network layer, denoted as :math:`\mathcal{M}_\phi`, while the decoder is represented by another neural network layer, :math:`\mathcal{N}_\theta`. The optimization problem is formulated as following:

.. math::
    \begin{equation}
        \begin{aligned}
            \{ \phi^*, \theta^* \} &= \argmin_{\phi, \theta} \mathcal{L}(\phi, \theta) \\
            & \coloneqq \sum_k \mathcal{L}_{\text{task}} ( \mathcal{N}_\theta(\mathcal{M}_\phi(x_k)), x_k) + \rho R_\rho(\phi) + \sigma R_\sigma(\theta)
        \end{aligned}
    \end{equation},


:math:`\text{            }` where the optimal parameters of the optical system and the decoder are represented by :math:`\{\phi^*, \theta^*\}`. The training dataset consists of input-output pairs :math:`\{x_k, y_k\}_{k=1}^K`. The loss function :math:`\mathcal{L}_{\text{task}}` is selected based on the task, usually using mean square error (MSE) for reconstruction and cross-entropy for classification. Regularization functions :math:`R_\rho(\phi)` and :math:`R_\sigma(\theta)`, with parameters :math:`\rho` and :math:`\sigma`, are used to prevent overfitting in the optical system and the decoder.

e2e
~~~~~~~~~
.. autoclass:: colibri_hdsp.archs.e2e.E2E


Models
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri_hdsp.models.autoencoder.Autoencoder
    colibri_hdsp.models.unet.Unet
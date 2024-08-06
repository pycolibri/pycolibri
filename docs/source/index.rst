Computational Optical Learning Library (Colibri) Documentation
==================

|Test Status| |Docs Status| |Python 3.8+| |colab|


Colibri is a deep learning based library specialized in optimizing the key parameters of optical systems that can be learned from data to improve the performance of the system. 

In Colibri, optical systems, neural networks, model based recovery algorithms, and datasets are implemented to be easily used or modified for new research ideas. The purpose of Colibri is to boost the research-related areas where optics and networks are required and introduce new researchers to state-of-the-art algorithms in a straightforward and friendly manner.



🔗 Relevant Links
------------------

* Source code: `https://github.com/pycolibri/pycolibri <https://github.com/pycolibri/pycolibri>`_
* Documentation: `https://pycolibri.github.io/pycolibri/ <https://pycolibri.github.io/pycolibri/>`_
* Video tutorials: `Link <Link>`_


🥅 Goals
------------------

- Easy to use, customize and add modules.
- Comprehensive documentation and examples.
- High-quality code and tests.
- Fast and efficient algorithms.
- Wide range of optical systems, neural networks, recovery algorithms, and datasets.
- Support for the latest research in the field.


💿 Installation
------------------

To get started with Colibri, install the library using the following steps:

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/pycolibri/pycolibri.git

2. Create a virtual environment with conda:

.. code-block:: bash

    conda create -n colibri python=3.10
    conda activate colibri

3. Install the requirements:

.. code-block:: bash

    pip install -r requirements.txt

4. Enjoy! 😄

🚀 Quick Start
------------------

Check out the demo list in the `examples <https://pycolibri.github.io/pycolibri/auto_examples/index.html>`_ folder to get started with Colibri.

* `Demo Colibri <https://pycolibri.github.io/pycolibri/auto_examples/demo_datasets.html#sphx-glr-auto-examples-demo-datasets-py>`_
* `Demo PnP <https://pycolibri.github.io/pycolibri/auto_examples/demo_pnp.html#sphx-glr-auto-examples-demo-pnp-py>`_
* `Demo FISTA <https://pycolibri.github.io/pycolibri/auto_examples/demo_fista.html#sphx-glr-auto-examples-demo-fista-py>`_
* `Demo DOEs <https://pycolibri.github.io/pycolibri/auto_examples/demo_does.html#sphx-glr-auto-examples-demo-does-py>`_
* `Demo Datasets <https://pycolibri.github.io/pycolibri/auto_examples/demo_datasets.html#sphx-glr-auto-examples-demo-datasets-py>`_

🧰 Available Modules
------------------

📷 **Optical Systems**

* Spectral Imaging

      -  `Single Pixel Camera (SPC) <https://pycolibri.github.io/pycolibri/stubs/colibri.optics.spc.SPC.html>`_
      -  `Single Disperser CASSI (SD-CASSI) <https://pycolibri.github.io/pycolibri/stubs/colibri.optics.cassi.SD_CASSI.html#colibri.optics.cassi.SD_CASSI>`_
      -  `Dual Disperser CASSI (DD-CASSI) <https://pycolibri.github.io/pycolibri/stubs/colibri.optics.cassi.DD_CASSI.html>`_
      -  `Color CASSI (C-CASSI) <https://pycolibri.github.io/pycolibri/stubs/colibri.optics.cassi.C_CASSI.html>`_
      -  `Diffractive Optical Element (DOE) <https://pycolibri.github.io/pycolibri/stubs/colibri.optics.doe.SingleDOESpectral.html>`_
      
📈 **Regularizers**

* Binary Regularizers

   - `Values <https://pycolibri.github.io/pycolibri/stubs/colibri.regularizers.Reg_Binary.html#colibri.regularizers.Reg_Binary>`_
   - `Transmitance <https://pycolibri.github.io/pycolibri/stubs/colibri.regularizers.Reg_Transmittance.html#colibri.regularizers.Reg_Transmittance>`_

* Stochastic Regularizers

   - `Correlation <https://pycolibri.github.io/pycolibri/stubs/colibri.regularizers.Correlation.html#colibri.regularizers.Correlation>`_
   - `Kullback-Leibler Divergence <https://pycolibri.github.io/pycolibri/stubs/colibri.regularizers.KLGaussian.html#colibri.regularizers.KLGaussian>`_
   - `Minimal Variance <https://pycolibri.github.io/pycolibri/stubs/colibri.regularizers.MinVariance.html#colibri.regularizers.MinVariance>`_

💻️ **Deep Neural Networks**

    - `Autoencoder <https://pycolibri.github.io/pycolibri/models.html>`_
    - `Unet <https://pycolibri.github.io/pycolibri/models.html>`_

🖥 **Recovery Algorithms**

* Algorithms

   - `Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) <https://pycolibri.github.io/pycolibri/recovery.html>`_
   - `Alternating Direction Method of Multipliers Plug and Play (ADMM-PnP) <https://pycolibri.github.io/pycolibri/stubs/colibri.recovery.pnp.PnP_ADMM.html#colibri.recovery.pnp.PnP_ADMM>`_

* Solvers

   - `L2L2Solver <https://pycolibri.github.io/pycolibri/stubs/colibri.recovery.solvers.core.L2L2Solver.html#colibri.recovery.solvers.core.L2L2Solver>`_
* Fidelity Terms

   - `L2 <https://pycolibri.github.io/pycolibri/stubs/colibri.recovery.terms.fidelity.L2.html#colibri.recovery.terms.fidelity.L2>`_
   - `L1 <https://pycolibri.github.io/pycolibri/stubs/colibri.recovery.terms.fidelity.L1.html#colibri.recovery.terms.fidelity.L1>`_

* Priors
   
      - `Sparsity <https://pycolibri.github.io/pycolibri/stubs/colibri.recovery.terms.prior.Sparsity.html#colibri.recovery.terms.prior.Sparsity>`_

* Transforms
   
      - `DCT <https://pycolibri.github.io/pycolibri/stubs/colibri.recovery.terms.transforms.DCT2D.html#colibri.recovery.terms.transforms.DCT2D>`_

🎆 **Frameworks**

* Coupled Optimization for Optics and Recovery 

   - `End-to-end framework <https://pycolibri.github.io/pycolibri/stubs/colibri.misc.e2e.E2E.html#colibri.misc.e2e.E2E>`_ 


💡 Contributing
------------------

Information about contributing to Colibri can be found in the `CONTRIBUTING.md <https://github.com/pycolibri/pycolibri/blob/main/README.md>`_ guide. and in the guide `How to Contribute <https://pycolibri.github.io/pycolibri/contributing.html>`_

🛡️ License
------------------

.. code-block:: text

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contents:

   architectures
   models
   optics
   datasets
   regularizers
   auto_examples/index
   recovery
   contributing

.. |Test Status| image:: https://github.com/pycolibri/pycolibri/actions/workflows/test.yml/badge.svg
   :target: https://github.com/pycolibri/pycolibri/actions/workflows/test.yml
.. |Docs Status| image:: https://github.com/pycolibri/pycolibri/actions/workflows/documentation.yml/badge.svg
   :target: https://github.com/pycolibri/pycolibri/actions/workflows/documentation.yml
.. |Python 3.8+| image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://www.python.org/downloads/release/python-380/
.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/pycolibri/pycolibri/blob/main/main.ipynb
.. |Github| image:: https://img.shields.io/badge/github-%23121011.svg
   :target: https://github.com/pycolibri/pycolibri

Optics
======

The optics module of our library encompasses a variety of optical systems and components designed to extend the capabilities of computational imaging. These systems are integral for capturing complex visual information, paving the way for advanced image reconstruction and analysis techniques.

cassi
~~~~~
Coded Aperture Snapshot Spectral Imaging (CASSI) systems allow for the capture of spectral information across multiple wavelengths simultaneously, without the need for traditional scanning methods. This technology is pivotal in applications requiring spectral discrimination and analysis.

.. autoclass:: colibri_hdsp.optics.cassi.CASSI
    :members:


spc
~~~
Single Pixel Cameras (SPC) represent a paradigm shift in imaging technology, relying on compressive sensing to reconstruct images from a single detection element. This approach is highly beneficial for imaging in challenging conditions, including low light and non-visible spectra.

.. autoclass:: colibri_hdsp.optics.spc.SPC
    :members:

============
Installation
============


The package can be installed directly from the Python Package Index or GitHub.
The version on GitHub might be more recent, as only stable versions are
uploaded to the Python Package Index.

PyPI
----

The version from PyPI can directly be installed using pip

.. code-block:: bash

    pip install scikit-gstat


GitHub
------

The most recent version from GitHub can be installed like:

.. code-block:: bash

    git clone git@github.com:mmaelicke/scikit-gstat
    cd scikit-gstat
    pip install -e .


Conda-Forge
-----------

Since version `0.5.5`, SciKit-GStat is available on Conda-Forge.
You can install it like:

.. code-block:: bash

    conda install -c conda-forge scikit-gstat

Note
----

On Windows, you might run into problems installing all requirements
in a clean Python environment, especially if C++ redistributables are missing. 
This can happen i.e. on *bare* VMs and the compilation of libraries required by
scipy, numpy or numba package are the ones failing.
In these cases, install the libraries first, and then SciKit-GStat or move to
the conda-forge package 

.. code-block:: bash

    conda install numpy, scipy, numba
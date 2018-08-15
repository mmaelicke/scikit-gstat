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

    git clone https://github.com/mmaelicke/scikit-gstat.git
    cd scikit-gstat
    pip install -r requirements.txt
    python setup.py install


Note
----

Depending on you OS, you might run into problems installing all requirements
in a clean Python environment. These problems are usually caused by the scipy
and numba package, which might need to be compiled. From our experience, no
problems should occur, when an environment manager like anaconda is used.
Then, the requirements can be installed like:

.. code-block:: bash

    conda install numpy, scipy, numba
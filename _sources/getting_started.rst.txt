===============
Getting Started
===============


Load the class and data
-----------------------

The main class of scikit-gstat is the Variogram. It can directly be imported
from the module, called skgstat. The main class can easily be demonstrated on
random data.

.. ipython:: python
    :okwarning:

    from skgstat import Variogram
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    np.random.seed(42)
    coordinates = np.random.gamma(20, 5, (50,2))
    np.random.seed(42)
    values = np.random.normal(20, 5, 50)

The Variogram needs at least an array of coordinates and an array of values
on instantiation.

.. ipython:: python

    V = Variogram(coordinates=coordinates, values=values)
    print(V)


Plot
----

The Variogram class has its own plotting method.

.. ipython:: python
    :okwarning:

    @savefig default_variogram.png width=7in
    V.plot()

With version 0.2, the histogram plot can also be disabled. This is most
useful, when the binning method for the lag classes is changed from `'even'`
step classes to `'uniform'` distribution in the lag classes.

.. ipython:: python
    :okwarning:

    V.set_bin_func('uniform')
    @savefig variogram_uniform.png width=7in
    V.plot(hist=False)


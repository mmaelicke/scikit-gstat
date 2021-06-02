===============
Getting Started
===============


Load the class and data
-----------------------

The main class of scikit-gstat is the Variogram. It can directly be imported
from the module, called skgstat. The main class can easily be demonstrated on
the data module available with version `>=0.5.5`.

.. ipython:: python
    :okwarning:

    import skgstat as skg
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    data = skg.data.pancake(N=500, seed=42)
    print(data.get('origin'))
    coordinates, values = data.get('sample')

The Variogram needs at least an array of coordinates and an array of values
on instantiation.

.. ipython:: python

    V = skg.Variogram(coordinates=coordinates, values=values)
    print(V)


Plot
----

The Variogram class has its own plotting method.

.. ipython:: python
    :okwarning:

    @savefig default_variogram.png width=7in
    V.plot()
    plt.close()

With version 0.2, the histogram plot can also be disabled. This is most
useful, when the binning method for the lag classes is changed from `'even'`
step classes to `'uniform'` distribution in the lag classes.

.. ipython:: python
    :okwarning:

    V.set_bin_func('uniform')
    @savefig variogram_uniform.png width=7in
    V.plot(hist=False)
    plt.close()

Mutating
--------

One of the main strenghs of :class:`Variogram <skgstat.Variogram>` is its 
ability to change arguments in place. Any dependent result or parameter
will be invalidated and re-caluculated.
You can i.e. increase the number of lag classes:

.. ipython:: python
    :okwarning:

    V.n_lags = 25
    V.maxlag = 500
    V.bin_func = 'kmeans'

    @savefig default_variogram_25lag.png width=7in
    V.plot()
    plt.close()

Note, how the experimental variogram was updated and the model was
fitted to the new data automatically.


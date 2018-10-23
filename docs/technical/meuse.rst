==============
Meuse data set
==============


General
=======


blabla

Loading data
~~~~~~~~~~~~

The data set might be familiar to you, it's the Meuse data set. It contains
various measurements of heavy metals and some of them show quite nice
covariance fields.

.. ipython:: python

    import pandas as pd
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    from skgstat import Variogram

    df = pd.read_csv('data/meuse.csv', sep='\t')
    df.head()

    @savefig copper.png width=7in
    plt.scatter(df.x.values, df.y.values, df.copper.values)


Apply the variogram
~~~~~~~~~~~~~~~~~~~

.. ipython:: python

    V = Variogram(df[['x', 'y']], df.copper, n_lags=15, maxlag='median')
    @savefig copper_vario.png width=7in
    V.plot()
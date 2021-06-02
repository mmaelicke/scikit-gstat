=================
Binning functions
=================

SciKit-GStat implements a large amount of binning functions,
which can be used to spatially aggregate the distance matrix 
into lag classes, or bins.
There are a number of functions available, which usually accept
more than one method identifier:


.. autofunction:: skgstat.binning.even_width_lags

.. autofunction:: skgstat.binning.uniform_count_lags

.. autofunction:: skgstat.binning.auto_derived_lags

.. autofunction:: skgstat.binning.kmeans

.. autofunction:: skgstat.binning.ward

.. autofunction:: skgstat.binning.stable_entropy_lags


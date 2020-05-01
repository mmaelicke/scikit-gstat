=========
Changelog
=========

Version 0.2.8
=============

- [Variogram] now raises a `Warning` if all input values are the same
- [DOCS] Tutorial added and Dockerfile finalized
- [Variogram] `normalize` default value changed to `normalize=False`
- [Variogram] `harmonize` parameter is removed
- [Variogram] Monotonization (old harmonize par) is available as a new
  theoretical model function. Can be used by setting `model='harmonize'`
- [interfaces] gstools interface implemented. 
  :func:`gstools_cov_model <skgstat.interfaces.gstools.gstools_cov_model>`
  takes a :class:`skgstat.Variogram` instance and returns a **fitted** 
  `gstools.CovModel`. 

Version 0.2.7
=============

- [Kriging] Little performance gains due to code cleanup.
- [Variogram] The `normalize=True` default in `__init__` will change to 
  `normalize=False` in a future version. A DeprecationWarning was included.
- [tests] The Variogram class fitting unit tests are now explicitly setting 
  the normalize parameter to handle the future deprecation.
- [tests] More unittests were added to increase coverage
- [interfaces] The new submodule `skgstat.interfaces` is introduced. This 
  submodule collects interfacing classes to use skgstat classes with other 
  Python modules.
- [interfaces] The first interfacing class is the 
  :class:`VariogramEstimator <skgstat.interfaces.VariogramEstimator>`. This 
  is a scikit-learn compatible `Estimator` class that can wrap a `Variogram`. 
  The intended usage is to find variogram hyper-parameters using `GridSearchCV`.
  This is also the only usecase covered in the unit tests.
- [interfaces] Implemented 
  :func:`pykrige_as_kwargs <skgstat.interfaces.pykrige.pykrige_as_kwargs>`. 
  Pass a :class:`Variogram <skgstat.Variogram>` object and a dict of parameters 
  is returned that can be passed to pykrige Kriging classes using the double 
  star operator.
- Added Dockerfile. You can now build a docker container with scikit-gstat 
  installed in a miniconda environment. On run, a jupyter server is exposed on
  Port 8888. In a future release, this server will serve tutorial notebooks.
- [stmodels] small bugfix in product model
- [stmodels] removed variogram wrapper and added stvariogram wrapper to 
  correctly detect space and time lags

Version 0.2.6
=============
- [OrdinaryKriging]: widely enhanced the class in terms of performance, code
  coverage and handling.

    - added `mode` property: The class can derive exact solutions or estimate
      the kriging matrix for high performance gains
    - multiprocessing is supported now
    - the `solver` property can be used to choose from 3 different solver for
      the kriging matrix.

- [OrdinaryKriging]: calculates the kriging variance along with the estimation itself.
  The Kriging variance can be accessed after a call to 
  :func:`OrdinaryKriging.transform <skgstat.OrdinaryKriging.transform>` and can be 
  accessed through the `OrdinaryKriging.sigma` attribute. 
- [Variogram] deprecated
  :func:`Variogram.compiled_model <skgstat.Variogram.compiled_model>`. Use
  :func:`Variogram.fitted_model <skgstat.Variogram.fitted_model>` instead.
- [Variogram] added a new and much faster version of the parameterized model:
  :func:`Variogram.fitted_model <skgstat.Variogram.fitted_model>`
- [Variogram] minor change in the cubic model. This made the adaption of the 
  associated unit test necessary. 

Version 0.2.5
=============
- added :class:`OrdinaryKriging <skgstat.OrdinaryKriging>` for using a
  :class:`Variogram <skgstat.Variogram>` to perform an interpolation.

Version 0.2.4
=============

- added :class:`SpaceTimeVariogram <skgstat.SpaceTimeVariogram>` for
  calculating dispersion functions depending on a space and a time lag.

Version 0.2.3
=============

- **[severe bug]** A severe bug was in
  :func:`Variogram.__vdiff_indexer <skgstat.Variogram.__vdiff_indexer>` was
  found and fixed. The iterator was indexing the
  :func:`Variogram._diff <skgstat.Variogram._diff>` array different from
  :func:`Variogram.distance <skgstat.Variogram.distance>`. **This lead to
  wrong semivariance values for all versions > 0.1.8!**. Fixed now.
- [Variogram] added unit tests for parameter setting
- [Variogram] fixed ``fit_sigma`` setting of ``'exp'``: changed the formula
  from :math:`e^{\left(\frac{1}{x}\right)}` to
  :math:`1. - e^{\left(\frac{1}{x}\right)}` in order to increase with
  distance and, thus, give less weight to distant lag classes during fitting.

Version 0.2.2
=============

- added DirectionalVariogram class for direction-dependent variograms
- [Variogram] changed default values for `estimator` and `model` from
  function to string

Version 0.2.1
=============

- added various unittests

Version 0.2.0
=============

- completely rewritten Variogram class compared to v0.1.8
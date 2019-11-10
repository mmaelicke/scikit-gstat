=========
Changelog
=========

Version 0.2.7
=============
- 

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
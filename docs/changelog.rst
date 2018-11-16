=========
Changelog
=========

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
=========
Changelog
=========

Version 0.5.0
=============
- [MetricSpace] A new class :class:`MetricSpace <skgstat.MetricSpace>` was introduced. This class can be passed
  to any class that accepted coordinates so far. This wrapper can be used to pre-calculate large distance
  matrices and pass it to a lot of Variograms. 
- [MetricSpacePair] A new class :class:`MetricSpacePair <skgstat.MetricSpacePair>` was introduced.
  This is a pair of two :class:`MetricSpaces <skgstat.MetricSpace>` and pre-calculates all distances between
  the two spaces. This is i.e. used in Kriging to pre-calcualte all distance between the input coordinates and
  the interpolation grid only once.

Version 0.4.4
=============
- [models] the changes to :func:`matern <skgstat.models.matern>` introduced in `0.3.2` are reversed. 
  The Matérn model does not adapt the smoothness scaling to effective range anymore, as the behavior was too
  inconsistent.
- [interface] minor bugfix of circular import in `variogram_estimator` interface
- [models] :func:`matern(0, ...) <skgstat.models.matern>` now returns the nugget instead of `numpy.NaN`
- [models] :func:`stable(0, ...) <skgstat.models.stable>` now returns the nugget instead of `numpy.NaN` or a 
  `ZeroDivisionError`.

Version 0.4.3
=============
- [Variogram] :func:`dim <skgstat.Variogram.dim>` now returns the spatial dimensionality of the input data.
- [Variogram] fixed a numpy depreaction warning in `_calc_distances`

Version 0.4.2
=============
- [Variogram] :func:`bins <skgstat.Variogram.bins>` now cases manual setted bin edges automatically
  to a :func:`numpy.array`.
- [Variogram] :func:`get_empirical <skgstat.Variogram.get_empirical>` returns the empirical variogram.
  That is a tuple of the current :func:`bins <skgstat.Variogram.bins>` and 
  :func:`experimental <skgstat.Variogram.experimental>` arrays, with the option to move the bin to the
  lag classes centers.

Version 0.4.1
=============
- [Variogram] moved the bin function setting into a wrapper instance method, which was an anonymous lambda before.
  This makes the Variogram serializable again.
- [Variogram] a list of pylint errors were solved. Still enough left.

Version 0.4.0
=============
- [binning] added `'stable_entropy'` option that will optimize the lag class edges to be of comparable Shannon Entropy.

Version 0.3.11
==============
- [Variogram] A new method is introduced to calculate fitting weights. Works for all but the manual fit
  method. By setting :func:`fit_sigma='entropy' <skgstat.Variogram.fit_sigma>`, the fitting weights will
  be adjusted according to the lag classes' Shannon entropy. That will ignore lag classes of high
  uncertainty and emphasize lags of low uncertainty.

Version 0.3.10
==============
- [binning] added a median aggregation option to :func:`ward <skgstat.binning.ward>`. This can be 
  enabled by setting `binning_agg_func` to `'median'`. The cluster centroids will be derived from 
  the members median value, instead of mean value.
- [Variogram] added :func:`fit_method='ml' <skgstat.Variogram.fit_method>` - a maximum likelihood fitting 
  procedure to fit the theoretical variogram to the experimental
- [Variogram] added :func:`fit_method='manual' <skgstat.Variogram.fit_method>`. This is a manual fitting 
  method that takes the variogram parameters either at instantiation prefixed by `fit_`, or as 
  keyword arguments by :func:`fit <skgstat.Variogram.fit>`. 
- [Variogram] the manual fitting method will preseve the previous parameters, if the Variogram was 
  fitted before and the fitting parameters are not manually overwritten.


Version 0.3.9
=============
- [binning] added :func:`kmeans <skgstat.binning.kmeans>` and :func:`ward <skgstat.binning.ward>` for forming
  non-equidistant lag classes based on a distance matrix clustering
- [Kriging] Kriging now stores the last interpolated field as `z`. This is the first of a few changes
  in future releases, which will ultimately add some plotting methods to Kriging.

Version 0.3.8
=============
- [plotting] minor bugfixes in plotting routines (wrong arguments, pltting issues)
- [docs] added a tutorial about plotting
- [binning] added :func:`auto_derived_lags <skgstat.binning.auto_derived_lags>` for a variety
  of different methods that find a good estimate for either the number of lag classes or the 
  lag class width. These can be used by passing the method name as :func:`bin_func <skgstat.Variogram.set_bin_func>` 
  parameter: Freedman-Diaconis (`'fd'`), Sturge's rule (`'sturges'`), Scott's rule (`'scott'`) and 
  Doane's extension to Sturge's rule (`'doane'`). 
  Uses `histogram_bin_edges <numpy.histogram_bin_edges>` internally.

Version 0.3.7
=============
- [Variogram] now accepts arbitary kwargs. These can be used to further specify functional behavior
  of the class. As of Version `0.3.7` this is used to pass arguments down to the 
  :func:`entropy <skgstat.estimators.entropy>` and :func:`percentile <skgstat.estimators.percentile>` 
  estimators.
- [Variogram] the :func:`describe <skgstat.Variogram.describe>` now adds the 
  :func:`init <skgstat.Variogram.__init__>` arguments by default to the output. The method can output 
  the init params as a nested dict inside the output or flatten the output dict.

Version 0.3.6
=============
.. warning:: 
  There is some potential breaking behaviour

- [Variogram] some internal code cleanup. Removed some unnecessary loops
- [Variogram] setting the :func:`n_lags <skgstat.Variogram.n_lags>` property now correctly forces
  a recalculation of the lag groupings. So far they were kept untouches, which might result
  in old experimental variogram values for the changed instance.
  **This is a potential breaking change**.
- [Variogram] The :func:`lag_classes <skgstat.Variogram.lag_classes>` generator now yields empty 
  arrays for unoccupied lag classes. This will result in :class:`NaN <numpy.NaN>` values for the 
  semi-variance. This is actually a bug-fix.
  **This is a potential breaking change**

Version 0.3.5
=============
- [plotting] The :func:`location_trend <skgstat.Variogram.location_trend>` can now add 
  trend model lines to the scatter plot for the `'plotly'` backend and calculate the 
  R² for the trend model.
- [Variogram] the *internal* attribute holding the name of the current distance function
  was renamed from `_dict_func` to `_dist_func_name`

Version 0.3.4
=============
- [plotting] The :func:`scattergram <skgstat.Variogram.scattergram>` 
  functions color the plotted points with respect to the lag bin they
  are originating from. For `matplotlib`, this coloring is suppressed, but can activated by 
  passing the argument ``scattergram(single_color=False)``.

Version 0.3.3
=============

- [plotting] a new submodule is introduced: :py:mod:`skgstat.plotting`. This contains all plotting functions. 
  The plotting behavior is not changed, but using :func:`skgstat.plotting.backend`, the used plotting library
  can be switched from `matplotlib` to `plotly`
- [stmodels] some code cleanup
- [SpaceTimeVariogram] finally can fit the product-sum model to the experimental variogram

Version 0.3.2
=============
- [models] Matérn model now adapts effective range to smoothness parameter
- [models] Matérn model documentation updated
- [models] some minor updates to references in the docs

Version 0.3.1
=============

- [Variogram] - internal distance calculations were refactored, to speed things up
- [Kriging] - internal distance calculations were refactored, to speed things up

Version 0.3.0
=============

- [Variogram] some internal calculations were changed.
- [DirectionalVariogram] - the circular search are is removed and raises a NotImplementedError
- [DirectionalVariogram] - direction mask data is calculated way faster and without shapely involved.
- shapely is not a dependency anymore
- [unittests] - more unittests were added.

Version 0.2.8
=============

- [Variogram] is now ``pickle.dump()``-able, by removing ``lambda`` usage (thanks to @redhog!)
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
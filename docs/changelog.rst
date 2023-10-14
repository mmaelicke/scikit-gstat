=========
Changelog
=========

Version 1.1-rc
==============

Version 1.0.11
--------------
- [models] the models now use `@numba.jit(nopython=True)` as this will be the default behaviour starting with `numba==0.59.0`

Version 1.0.10
--------------
- [MetricSpace] now supports the Mahalanobis distance as well

Version 1.0.9
-------------
- [binning] fixed a deprecation warning raised by a future change to KMeans to preserve the current behavior
- [data] Added a data generator for creating :func:`random multivariate data <skgstat.data.corr_variable>`
- [tests] Added tests for :func:`random multivariate data <skgstat.data.corr_variable>`

Version 1.0.8
-------------
- [util] added :func:`cross_variograms <skgstat.cross_variograms>` for calculating cross-variograms for
  all combinations of ``N`` input variables. Variograms are returned in a 2D List (matrix) with all
  primary variograms on the diagonal.
- [util] added support for :class:`DirectionalVariogram <skgstat.DirectionalVariogram>` in case azimuth,
  tolerance or bandwidth are passed as keyword arguments to :func:`cross_variograms <skgstat.cross_variograms>`
- [util] added tests for new cross_variograms function to be implemented.

Version 1.0.7
-------------
- [Variogram] The Variogram instance now removed all usages of the deprecated ``Variogram.residuals`` property
- [tests] All warnings in the tests have been resolved, mostly by updating imports
- [tests] Expected warnings in unittests are ignored now.

Version 1.0.6
-------------
This is technically the same as the 1.0.5 as I screwed up and uploaded to PyPI without merging the changes.


Version 1.0.5
-------------
- [Variogram] The variogram now accepts 2D values in ``(n_samples, 2)`` shape. The second column will be
  interpreted as a co-variable co-located to the observations and the Variogram then calculates the
  **cross-variogram** for ``values[:, 0] ~ values[:, 1]``
  Note that this only implements the basic calculation of cross-variograms. Utility functions and
  plotting still need to be updated. A tutorial will also be added in future.

Version 1.0.4
-------------
- [Variogram] added :func:`Variogram.pairwise_diffs <skgstat.Variogram.pairwise_diffs>` property for accessing
  :func:`Variogram._diff <skgstat.Variogram._diff>`
- [Variogram] added :func:`Variogram.model_residuals <skgstat.Variogram.model_residuals>` as a replacement for
  :func:`Variogram.residuals <skgstat.Variogram.residuals>`
- [Variogram] deprecated :func:`Variogram.residuals <skgstat.Variogram.residuals>` as it could be confused with
  :func:`Variogram.pairwise_diffs <skgstat.Variogram.pairwise_diffs>`, which are also residuals


Version 1.0.3
-------------
- [Variogram] :func:`Variogram.pairwise_diffs <skgstat.Variogram.pairwise_diffs>` wraps around old ``_diff``
  and should be used instead of directly accessing ``_diff``
- [Variogram] :func:`Variogram.model_residuals <skgstat.Variogram.model_residuals>` will replace
  ``Variogram.residuals`` which has been deprecated

Version 1.0.1
-------------
- [Variogram] documentation added to :func:`use_nugget <skgstat.Variogram.use_nugget>`
- [Variogram] :func:`Variogram.fit(method='manual') <skgstat.Variogram.fit>` now
  implicitly sets :func:`use_nugget=True <skgstat.Variogram.use_nugget>` if a
  nugget is passed to fit.

Version 1.0.0
=============
- [plotting] the 3D surface plot is now handling the opacity settings correctly.
- [utils] the utils now include the likelihood submodule, which includes a
  :func:`get_likelihood <skgstat.util.likelihood.get_likelihood>` function factory.
  The returned function can be minimized using SciPy to perform maximum likelihood fits.

Version 0.6.14
--------------
- [plotting] plotly contour plots are showing a colorbar label by default
- [DirectionalVariogram] the constructor now sets an empty uncertainty array to prevent the
  class from throwing and error if no confidence interval is given.

Version 0.6.13
--------------
- [docs] the sphinx recipe knitting a TeX file from docs is now ignored on fail
  Reason is that the current build is too lage and any kind of buffer is overflowing
- [docs] The jupyter notebook tutorials for the Docker image are now at root level.
- [docs] The documentation tutorials are now sphinx-gallery builds of the notebook
  tutorial examples
- [docs] all tutorials have been updated (spelling, code style etc.)
- [docs] all tutorials now use the data submodule to be reproducible

Version 0.6.12
--------------
- [data] the dataset loader can now return pandas.DataFrame objects
- [Dockerfile] some cleanups for making future tutorials work.

Version 0.6.11
--------------
- [Variogram] The kriging based leave-one-out cross validation is now omitting NaN.

Version 0.6.10
--------------
- [Variogram] The KMeans based binning function is now raising a value error if
  a ConvergenceWarning is found. The reason is, that the original settings for binning
  were not valid if KMeans did not converge and thus, the bins array might not be
  in a well defined state.

Version 0.6.9
-------------
- SciKit-GStat is now tested for Python 3.9. Finally.
  All unittest are now automatically run for Python 3.6, 3.7, 3.8 and 3.9

Version 0.6.8
-------------
- [RasterMetricSpace] small bugfix for Exceptions raised with small sample sizes

Version 0.6.7
-------------
- [RasterMetricSpace] a new class is introduced: :class:`RasterEquidistantMetricSpace <skgstat.RasterEquidistantMetricSpace>`.
  An instance can be passed as `coordinates`. It samples a given Raster image at concentric rings, to derive a
  more uniformly distributed distance matrix.

Version 0.6.6
-------------
- [Variogram] The automatic fitting of a theoretical variogram model is now optional. You can pass `None` as
  `fit_method` parameter, which will suppress the fitting.

Version 0.6.5
-------------
- [Variogram] now supports custom bin edges for the experimental variogram. :func:`Variogram.bins <skgstat.Variogram.bins>`
  now accepts a list or array of upper bin edges.
- [Variogram] has a new property called :func:`bin_count <skgstat.Variogram.bin_count>` which returns the number of
  point pairs within each lag class

Version 0.6.4
-------------
- [Kriging] `OrdinaryKriging.sigma <skgstat.OrdinaryKriging>` is now initialized as a `NaN`-filled array.
- [Kriging] `OrdinaryKriging._estimator <skgstat.OrdinaryKriging>` handles the error variance matrix index
  now correctly. On error during kriging, the index was not incremented, which lead to malformed error variance field output.

Version 0.6.3
-------------
- [interfaces] If any of the gstools interfaces are used, the Variogram will call :func:`fit <skgstat.Variogram.fit>`
  without forcing a full preprocessing cycle. This fixes edge cases, where a parameter was mutated, but the fitting
  not performed before the instance was exported. This should only have happened in very rare occasions.
- [data] added the meuse dataset from the R-package ``'sp'``

Version 0.6.2
-------------
- [Variogram] the fitting method is now implemented as :func:`Variogram.fit_method <skgstat.Variogram.fit_method>`
  property. It will drop fitting parameters if the fit method is changed to something else than ``'manual'``.
- [Variogram] If an invalid :func:`Variogram.fit_method <skgstat.Variogram.fit_method>` is set, an
  :class:`AttributeError` will instantly be raised. Beforehand it was only raised on the next call of
  :func:`fit <skgstat.Variogram.fit>`

Version 0.6.1
-------------
- The Dockerfile was completely rewritten. A user can now specify the used Python version
  at build time of the docker image.
- The Dockerfile is now part of the python package

Version 0.6.0
-------------
- The util and data submodule are now always loaded at top-level
- fixed a potential circular import
- added uncertainty tools to util. This is not yet finished and may change the signature before
  it gets stable with Version 1.0 or 1.1

.. note::
  The current implementation of uncertainty propagation is not stable. It will be changed until
  version 0.7. The entry-point `obs_sigma` will stay stable and persist, but currently the uncertainty
  propagation will not be updated and invalidated as the Variogram instance changes.

Version 0.5.6
-------------
- [Variogram] the internal :class:`MetricSpace <skgstat.MetricSpace>` instance used to calculate the distance matrix
  is now available as the :any:`Variogram.metric_space <skgstat.Variogram.metric_space>` property.
- [Variogram] :any:`Variogram.metric_space <skgstat.Variogram.metric_space>` is now read-only.
- [unittest] two unittests are changed (linting, not functionality)

Version 0.5.5
-------------
- [data] new submodule :any:`data <skgstat.data>` contains sample random fields and methods for sampling
  these fields in a reproducible way at random locations and different sample sizes.

Version 0.5.4
-------------
- [util] added a new `cross_validation` utility module to cross-validate variograms with leave-one-out Kriging
  cross validations.

Version 0.5.3
-------------
- [MetricSpace] new class :class:`ProbabilisticMetricSpace <skgstat.MetricSpace.ProbabilisticMetricSpace>` that
  extends the metric space by a stochastic element to draw samples from the input data, instead of using
  the full dataset.

Version 0.5.2
-------------
- [interface] new interface function added: :func:`to_gs_krige <skgstat.Variogram.to_gs_krige>`. This interface
  will return a :any:`gs.Krige <gstools.Krige>` instance from the fitted variogram.
- some typos were corrected
- some code refactored (mainly linting errors)

Version 0.5.1
-------------
- [plotting] the spatio-temporal 2D and 3D plots now label the axis correctly.
- [plotting] fixed swapped plotting axes for spatio-temporal plots.

Version 0.5.0
-------------
- [MetricSpace] A new class :class:`MetricSpace <skgstat.MetricSpace>` was introduced. This class can be passed
  to any class that accepted coordinates so far. This wrapper can be used to pre-calculate large distance
  matrices and pass it to a lot of Variograms.
- [MetricSpacePair] A new class :class:`MetricSpacePair <skgstat.MetricSpacePair>` was introduced.
  This is a pair of two :class:`MetricSpaces <skgstat.MetricSpace>` and pre-calculates all distances between
  the two spaces. This is i.e. used in Kriging to pre-calcualte all distance between the input coordinates and
  the interpolation grid only once.

Version 0.4.4
-------------
- [models] the changes to :func:`matern <skgstat.models.matern>` introduced in `0.3.2` are reversed.
  The Matérn model does not adapt the smoothness scaling to effective range anymore, as the behavior was too
  inconsistent.
- [interface] minor bugfix of circular import in `variogram_estimator` interface
- [models] :func:`matern(0, ...) <skgstat.models.matern>` now returns the nugget instead of `numpy.NaN`
- [models] :func:`stable(0, ...) <skgstat.models.stable>` now returns the nugget instead of `numpy.NaN` or a
  `ZeroDivisionError`.

Version 0.4.3
-------------
- [Variogram] :func:`dim <skgstat.Variogram.dim>` now returns the spatial dimensionality of the input data.
- [Variogram] fixed a numpy depreaction warning in `_calc_distances`

Version 0.4.2
-------------
- [Variogram] :func:`bins <skgstat.Variogram.bins>` now cases manual set bin edges automatically
  to a :func:`numpy.array`.
- [Variogram] :func:`get_empirical <skgstat.Variogram.get_empirical>` returns the empirical variogram.
  That is a tuple of the current :func:`bins <skgstat.Variogram.bins>` and
  :func:`experimental <skgstat.Variogram.experimental>` arrays, with the option to move the bin to the
  lag classes centers.

Version 0.4.1
-------------
- [Variogram] moved the bin function setting into a wrapper instance method, which was an anonymous lambda before.
  This makes the Variogram serializable again.
- [Variogram] a list of pylint errors were solved. Still enough left.

Version 0.4.0
-------------
- [binning] added `'stable_entropy'` option that will optimize the lag class edges to be of comparable Shannon Entropy.

Version 0.3.11
--------------
- [Variogram] A new method is introduced to calculate fitting weights. Works for all but the manual fit
  method. By setting :func:`fit_sigma-'entropy' <skgstat.Variogram.fit_sigma>`, the fitting weights will
  be adjusted according to the lag classes' Shannon entropy. That will ignore lag classes of high
  uncertainty and emphasize lags of low uncertainty.

Version 0.3.10
--------------
- [binning] added a median aggregation option to :func:`ward <skgstat.binning.ward>`. This can be
  enabled by setting `binning_agg_func` to `'median'`. The cluster centroids will be derived from
  the members median value, instead of mean value.
- [Variogram] added :func:`fit_method-'ml' <skgstat.Variogram.fit_method>` - a maximum likelihood fitting
  procedure to fit the theoretical variogram to the experimental
- [Variogram] added :func:`fit_method-'manual' <skgstat.Variogram.fit_method>`. This is a manual fitting
  method that takes the variogram parameters either at instantiation prefixed by `fit_`, or as
  keyword arguments by :func:`fit <skgstat.Variogram.fit>`.
- [Variogram] the manual fitting method will preserve the previous parameters, if the Variogram was
  fitted before and the fitting parameters are not manually overwritten.


Version 0.3.9
-------------
- [binning] added :func:`kmeans <skgstat.binning.kmeans>` and :func:`ward <skgstat.binning.ward>` for forming
  non-equidistant lag classes based on a distance matrix clustering
- [Kriging] Kriging now stores the last interpolated field as `z`. This is the first of a few changes
  in future releases, which will ultimately add some plotting methods to Kriging.

Version 0.3.8
-------------
- [plotting] minor bugfixes in plotting routines (wrong arguments, pltting issues)
- [docs] added a tutorial about plotting
- [binning] added :func:`auto_derived_lags <skgstat.binning.auto_derived_lags>` for a variety
  of different methods that find a good estimate for either the number of lag classes or the
  lag class width. These can be used by passing the method name as :func:`bin_func <skgstat.Variogram.set_bin_func>`
  parameter: Freedman-Diaconis (`'fd'`), Sturge's rule (`'sturges'`), Scott's rule (`'scott'`) and
  Doane's extension to Sturge's rule (`'doane'`).
  Uses `histogram_bin_edges <numpy.histogram_bin_edges>` internally.

Version 0.3.7
-------------
- [Variogram] now accepts arbitrary kwargs. These can be used to further specify functional behavior
  of the class. As of Version `0.3.7` this is used to pass arguments down to the
  :func:`entropy <skgstat.estimators.entropy>` and :func:`percentile <skgstat.estimators.percentile>`
  estimators.
- [Variogram] the :func:`describe <skgstat.Variogram.describe>` now adds the
  :func:`init <skgstat.Variogram.__init__>` arguments by default to the output. The method can output
  the init params as a nested dict inside the output or flatten the output dict.

Version 0.3.6
-------------
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
-------------
- [plotting] The :func:`location_trend <skgstat.Variogram.location_trend>` can now add
  trend model lines to the scatter plot for the `'plotly'` backend and calculate the
  R² for the trend model.
- [Variogram] the *internal* attribute holding the name of the current distance function
  was renamed from `_dict_func` to `_dist_func_name`

Version 0.3.4
-------------
- [plotting] The :func:`scattergram <skgstat.Variogram.scattergram>`
  functions color the plotted points with respect to the lag bin they
  are originating from. For `matplotlib`, this coloring is suppressed, but can activated by
  passing the argument ``scattergram(single_color-False)``.

Version 0.3.3
-------------

- [plotting] a new submodule is introduced: :py:mod:`skgstat.plotting`. This contains all plotting functions.
  The plotting behavior is not changed, but using :func:`skgstat.plotting.backend`, the used plotting library
  can be switched from `matplotlib` to `plotly`
- [stmodels] some code cleanup
- [SpaceTimeVariogram] finally can fit the product-sum model to the experimental variogram

Version 0.3.2
-------------
- [models] Matérn model now adapts effective range to smoothness parameter
- [models] Matérn model documentation updated
- [models] some minor updates to references in the docs

Version 0.3.1
-------------

- [Variogram] - internal distance calculations were refactored, to speed things up
- [Kriging] - internal distance calculations were refactored, to speed things up

Version 0.3.0
-------------

- [Variogram] some internal calculations were changed.
- [DirectionalVariogram] - the circular search are is removed and raises a NotImplementedError
- [DirectionalVariogram] - direction mask data is calculated way faster and without shapely involved.
- shapely is not a dependency anymore
- [unittests] - more unittests were added.

Version 0.2.8
-------------

- [Variogram] is now ``pickle.dump()``-able, by removing ``lambda`` usage (thanks to @redhog!)
- [Variogram] now raises a `Warning` if all input values are the same
- [DOCS] Tutorial added and Dockerfile finalized
- [Variogram] `normalize` default value changed to `normalize-False`
- [Variogram] `harmonize` parameter is removed
- [Variogram] Monotonization (old harmonize par) is available as a new
  theoretical model function. Can be used by setting `model-'harmonize'`
- [interfaces] gstools interface implemented.
  :func:`gstools_cov_model <skgstat.interfaces.gstools.gstools_cov_model>`
  takes a :class:`skgstat.Variogram` instance and returns a **fitted**
  `gstools.CovModel`.

Version 0.2.7
-------------

- [Kriging] Little performance gains due to code cleanup.
- [Variogram] The `normalize-True` default in `__init__` will change to
  `normalize-False` in a future version. A DeprecationWarning was included.
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
-------------
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
-------------
- added :class:`OrdinaryKriging <skgstat.OrdinaryKriging>` for using a
  :class:`Variogram <skgstat.Variogram>` to perform an interpolation.

Version 0.2.4
-------------

- added :class:`SpaceTimeVariogram <skgstat.SpaceTimeVariogram>` for
  calculating dispersion functions depending on a space and a time lag.

Version 0.2.3
-------------

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
-------------

- added DirectionalVariogram class for direction-dependent variograms
- [Variogram] changed default values for `estimator` and `model` from
  function to string

Version 0.2.1
-------------

- added various unittests

Version 0.2.0
-------------

- completely rewritten Variogram class compared to v0.1.8
